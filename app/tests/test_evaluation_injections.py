"""Opt-in evidence on materialized data; all writes stay in pytest's tmp_path."""

import copy
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import imagehash
import pytest
import yaml
from PIL import Image

from analyzers.duplicates import analyze_duplicates
from analyzers.invalid_boxes import analyze_invalid_boxes
from ingestion.loader import load_dataset
from policies.duplicates import load_duplicate_config
from policies.invalid_boxes import load_invalid_box_config
from policies.models import load_quality_policy
from presentation.contracts import QualityReport
from presentation.gate import build_quality_report

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_EVALUATION_INJECTIONS") != "1",
    reason="Set RUN_EVALUATION_INJECTIONS=1 with the real dataset materialized",
)


def emit(name, **evidence):
    print(json.dumps({"probe": name, **evidence}, ensure_ascii=False, sort_keys=True))


def fingerprints(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


@pytest.fixture(scope="module")
def real_data():
    source = ROOT / "data" / "raw"
    paths = sorted(p for p in source.rglob("*") if p.is_file())
    paths += [APP / "policies" / "quality.yaml"]
    before = fingerprints(paths)
    coco = load_dataset(source / "annotations")
    assert coco.images and len(coco.annotations) >= 2
    yield source, coco
    assert fingerprints(paths) == before, "Source data or normal policy changed"


def test_real_invalid_boxes(real_data):
    _, coco = real_data
    original = coco.model_dump()
    bad = copy.deepcopy(original)
    config = load_invalid_box_config()
    before = analyze_invalid_boxes(original, config)
    assert before.metric_value == 0
    bad["annotations"][0]["bbox"][2] = -1.0
    second = bad["annotations"][1]
    image = next(i for i in bad["images"] if i["id"] == second["image_id"])
    second["bbox"][0] = float(image["width"])
    after = analyze_invalid_boxes(bad, config)
    offenders = {x["annotation_id"]: x for x in after.details["offending_samples"]}
    assert after.metric_value == 2
    assert not after.passed
    assert "non_positive_width" in offenders[bad["annotations"][0]["id"]]["reasons"]
    assert "exceeds_image_width" in offenders[second["id"]]["reasons"]
    assert coco.model_dump() == original
    emit("boxes", before=before.model_dump(), after=after.model_dump(), expected_invalid=2)


def test_real_recompressed_copy(real_data, tmp_path):
    source, coco = real_data
    image = min(coco.images, key=lambda item: item.id)
    content = (source / "images" / image.file_name).read_bytes()
    copy_id = max(i.id for i in coco.images) + 1
    config = load_duplicate_config()
    before = analyze_duplicates({image.id: content}, config)
    assert before.metric_value == 0
    with Image.open(BytesIO(content)) as decoded:
        copy_path = tmp_path / "recompressed.jpg"
        decoded.convert("RGB").save(copy_path, format="JPEG", quality=65)
    recompressed = copy_path.read_bytes()
    assert recompressed != content
    after = analyze_duplicates({image.id: content, copy_id: recompressed}, config)
    with Image.open(BytesIO(content)) as a, Image.open(BytesIO(recompressed)) as b:
        distance = int(imagehash.phash(a, hash_size=8) - imagehash.phash(b, hash_size=8))
    expected_pair = {
        "image_id_a": image.id,
        "image_id_b": copy_id,
        "hamming_distance": distance,
        "similarity": 1 - distance / 64,
    }
    assert after.metric_value == 1
    assert not after.passed
    assert after.details["image_pairs"] == [expected_pair]
    assert expected_pair["similarity"] >= config.threshold
    emit(
        "recompression",
        source=image.file_name,
        quality=65,
        before=before.model_dump(),
        after=after.model_dump(),
        original_sha256=hashlib.sha256(content).hexdigest(),
        recompressed_sha256=hashlib.sha256(recompressed).hexdigest(),
    )


def run_process(command, cwd, env):
    result = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, check=False, timeout=180
    )
    # Only tool output is recorded, never the process environment or credentials.
    emit(
        "process",
        command=shlex.join(command),
        exit_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    return result


def test_impossible_threshold_blocks_real_dvc_downstream(real_data, tmp_path):
    source, coco = real_data
    dvc = Path(os.environ.get("EVALUATION_DVC", ROOT / ".venv-dvc" / "bin" / "dvc"))
    assert dvc.is_file(), "Install the README's separate DVC environment first"
    workspace = tmp_path / "pipeline"
    shutil.copytree(
        APP,
        workspace / "app",
        ignore=shutil.ignore_patterns(
            ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".env", ".env.*"
        ),
    )
    shutil.copytree(source, workspace / "data" / "raw")
    reports = workspace / "reports"
    reports.mkdir()
    policy_path = workspace / "app" / "policies" / "quality.yaml"
    policy = load_quality_policy(policy_path)
    source_report = build_quality_report(
        dataset_dir=source, policy=policy, dataset_version="evaluation-injections"
    )
    # Controlled two-class fixture: do not hide empty categories in the real input.
    fixture = coco.model_dump()
    annotated = {a["category_id"] for a in fixture["annotations"]}
    excluded = [c for c in fixture["categories"] if c["id"] not in annotated]
    fixture["categories"] = [c for c in fixture["categories"] if c["id"] in annotated]
    annotations = workspace / "data" / "raw" / "annotations"
    for path in annotations.glob("*.json"):
        path.unlink()  # Disposable copies only.
    (annotations / "fixture.json").write_text(json.dumps(fixture))
    emit("source_baseline", status=source_report.status, excluded_fixture_categories=excluded)
    baseline = build_quality_report(
        dataset_dir=workspace / "data" / "raw",
        policy=policy,
        dataset_version="evaluation-injections",
    )
    assert baseline.status != "failed", "Baseline must pass before injecting the impossible limit"
    report_path = reports / "quality.json"
    report_path.write_text(baseline.model_dump_json(indent=2), encoding="utf-8")
    stages = yaml.safe_load((ROOT / "dvc.yaml").read_text())["stages"]
    # Actual gate/split definitions, isolated from other stages and remote data.
    selected = {name: copy.deepcopy(stages[name]) for name in ("quality_gate", "split")}
    for name, stage in selected.items():
        script = "dvc_gate_stage.py" if name == "quality_gate" else "dvc_split_stage.py"
        assert stage["cmd"] == f"uv run python {script}"
        stage["cmd"] = shlex.join([sys.executable, script])
    (workspace / "dvc.yaml").write_text(yaml.safe_dump({"stages": selected}))
    env = {
        "PATH": os.defpath,
        "HOME": str(tmp_path / "home"),
        "DVC_NO_ANALYTICS": "1",
        "DVC_SITE_CACHE_DIR": str(tmp_path / "dvc-site"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "REPORTS_DIR": str(reports),
    }
    Path(env["HOME"]).mkdir()
    assert run_process([str(dvc), "init", "--no-scm"], workspace, env).returncode == 0
    assert run_process([str(dvc), "repro", "split"], workspace, env).returncode == 0
    split_path = reports / "splits.json"
    assert split_path.is_file()
    split_path.unlink()  # Only the disposable baseline output, not the real report.
    document = yaml.safe_load(policy_path.read_text())
    document["min_images_per_class"]["threshold"] = len(coco.images) + 1
    policy_path.write_text(yaml.safe_dump(document))
    failed = build_quality_report(
        dataset_dir=workspace / "data" / "raw",
        policy=load_quality_policy(policy_path),
        dataset_version="evaluation-injections",
    )
    assert failed.status == "failed"
    report_path.write_text(failed.model_dump_json(indent=2), encoding="utf-8")
    assert QualityReport.model_validate_json(report_path.read_text()) == failed
    check = next(c for c in failed.checks if c.check_name == "min_images_per_class")
    counts = {
        category: len(
            {a["image_id"] for a in fixture["annotations"] if a["category_id"] == category}
        )
        for category in annotated
    }
    assert check.metric_value == min(counts.values())
    assert check.details["criterion"]["threshold"] == len(coco.images) + 1
    assert not check.passed and check.action == "fail"
    emit(
        "threshold",
        before_status=baseline.status,
        after_status=failed.status,
        original_threshold=policy.min_images_per_class.threshold,
        check=check.model_dump(),
    )
    gate = run_process([sys.executable, "dvc_gate_stage.py"], workspace / "app", env)
    assert gate.returncode == 1
    downstream = run_process([str(dvc), "repro", "split"], workspace, env)
    assert downstream.returncode != 0
    assert "Running stage 'split'" not in downstream.stdout + downstream.stderr
    assert not split_path.exists()
    assert not (reports / ".quality_gate.passed").exists()
    emit("downstream", split_executed=False, split_report_exists=False, marker_exists=False)
