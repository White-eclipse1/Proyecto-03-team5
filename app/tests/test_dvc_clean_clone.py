"""OPS-10 (Fase 0 y M1): `dvc status` de un clon limpio no debe reportar el pipeline cambiado.

Las salidas con `cache: false` no van al cache de DVC: se versionan con git. Si una
está ignorada, un clon limpio no la tiene y `dvc status` marca como cambiadas las
etapas que dependen de ella (le pasaba a `reports/.quality_gate.passed`).
"""

import hashlib
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _uncached_outputs() -> list[Path]:
    pipeline = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    found = []
    for stage in pipeline["stages"].values():
        workdir = REPO_ROOT / stage.get("wdir", ".")
        for kind in ("outs", "metrics"):
            for out in stage.get(kind, []):
                if isinstance(out, dict):
                    ((path, options),) = out.items()
                    if options.get("cache") is False:
                        found.append((workdir / path).resolve())
    return found


def test_every_uncached_dvc_output_is_tracked_by_git():
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.splitlines()
    )
    outputs = _uncached_outputs()
    assert outputs, "dvc.yaml no declara salidas cache: false"

    missing = [p.relative_to(REPO_ROOT).as_posix() for p in outputs]
    missing = [p for p in missing if p not in tracked]

    assert missing == []


def test_the_quality_gate_marker_is_the_one_recorded_in_dvc_lock():
    lock = yaml.safe_load((REPO_ROOT / "dvc.lock").read_text(encoding="utf-8"))
    recorded = {out["path"]: out["md5"] for out in lock["stages"]["quality_gate"]["outs"]}
    marker = REPO_ROOT / "reports" / ".quality_gate.passed"

    assert (
        hashlib.md5(marker.read_bytes()).hexdigest() == recorded["../reports/.quality_gate.passed"]
    )
