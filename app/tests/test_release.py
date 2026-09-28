import json

import pytest

from policies.models import load_quality_policy
from presentation.release import cut_release, diff_releases
from tests._dataset_fixtures import write_coco_dataset

QUALITY_YAML = """
min_images_per_class:
  threshold: 1
  action: fail
max_imbalance_ratio:
  threshold: 999
  action: warn
max_small_object_ratio:
  threshold: 0.99
  width_px: 1
  height_px: 1
  action: warn
degenerate_boxes:
  threshold: 0
  action: fail
cross_split_leakage:
  threshold: 0
  action: fail
duplicate_similarity_threshold:
  threshold: 0.999
  action: warn
min_spatial_dispersion:
  threshold: 0.0
  action: warn
"""


def _policy_path(tmp_path):
    path = tmp_path / "quality.yaml"
    path.write_text(QUALITY_YAML, encoding="utf-8")
    return path


def test_cut_release_writes_quality_splits_and_catalog(tmp_path):
    dataset_dir = write_coco_dataset(tmp_path / "dataset", cats=6, dogs=6)
    reports_dir = tmp_path / "reports"
    policy = load_quality_policy(_policy_path(tmp_path))

    release = cut_release("v0.1.0", dataset_dir=dataset_dir, reports_dir=reports_dir, policy=policy)

    assert release.dataset_version == "v0.1.0"
    quality_path = reports_dir / "releases" / "v0.1.0" / "quality.json"
    splits_path = reports_dir / "releases" / "v0.1.0" / "splits.json"
    assert quality_path.exists()
    assert splits_path.exists()

    catalog = json.loads((reports_dir / "versions.json").read_text(encoding="utf-8"))
    assert catalog["releases"] == [
        {
            "dataset_version": "v0.1.0",
            "quality_file": "releases/v0.1.0/quality.json",
            "splits_file": "releases/v0.1.0/splits.json",
        }
    ]

    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    assert quality["dataset_version"] == "v0.1.0"
    assert quality["status"] != "failed"
    assert len(quality["checks"]) == 7
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    assert splits["total_images"] == 12
    assert (
        splits["splits"]["train"]["image_count"]
        + splits["splits"]["validation"]["image_count"]
        + splits["splits"]["test"]["image_count"]
        == 12
    )


def test_cut_release_rejects_failed_quality_before_writing(tmp_path):
    dataset_dir = write_coco_dataset(tmp_path / "dataset", cats=6, dogs=6)
    reports_dir = tmp_path / "reports"
    policy = load_quality_policy(_policy_path(tmp_path))
    policy.min_images_per_class.threshold = 999

    with pytest.raises(ValueError, match="en failed"):
        cut_release("v0.1.0", dataset_dir=dataset_dir, reports_dir=reports_dir, policy=policy)

    assert not (reports_dir / "versions.json").exists()
    assert not (reports_dir / "releases" / "v0.1.0").exists()


def test_cut_release_rejects_bad_version_format(tmp_path):
    dataset_dir = write_coco_dataset(tmp_path / "dataset", cats=6, dogs=6)
    policy = load_quality_policy(_policy_path(tmp_path))

    with pytest.raises(ValueError, match=r"vMAJOR\.MINOR\.PATCH"):
        cut_release("1.0", dataset_dir=dataset_dir, reports_dir=tmp_path / "reports", policy=policy)


def test_cut_release_rejects_duplicate_version(tmp_path):
    dataset_dir = write_coco_dataset(tmp_path / "dataset", cats=6, dogs=6)
    reports_dir = tmp_path / "reports"
    policy = load_quality_policy(_policy_path(tmp_path))
    cut_release("v0.1.0", dataset_dir=dataset_dir, reports_dir=reports_dir, policy=policy)

    with pytest.raises(ValueError, match="ya existe"):
        cut_release("v0.1.0", dataset_dir=dataset_dir, reports_dir=reports_dir, policy=policy)


def test_diff_releases_compares_two_cut_releases(tmp_path):
    reports_dir = tmp_path / "reports"
    policy = load_quality_policy(_policy_path(tmp_path))

    dataset_a = write_coco_dataset(tmp_path / "dataset-a", cats=6, dogs=6)
    cut_release("v0.1.0", dataset_dir=dataset_a, reports_dir=reports_dir, policy=policy)

    dataset_b = write_coco_dataset(tmp_path / "dataset-b", cats=9, dogs=6)
    cut_release("v0.2.0", dataset_dir=dataset_b, reports_dir=reports_dir, policy=policy)

    diff = diff_releases("v0.1.0", "v0.2.0", reports_dir=reports_dir)

    assert diff["from"] == "v0.1.0"
    assert diff["to"] == "v0.2.0"
    assert diff["images_per_category"]["cat"] == {"from": 6, "to": 9}
    assert diff["images_per_category"]["dog"] == {"from": 6, "to": 6}


def test_diff_releases_rejects_unknown_version(tmp_path):
    reports_dir = tmp_path / "reports"
    policy = load_quality_policy(_policy_path(tmp_path))
    dataset_dir = write_coco_dataset(tmp_path / "dataset", cats=6, dogs=6)
    cut_release("v0.1.0", dataset_dir=dataset_dir, reports_dir=reports_dir, policy=policy)

    with pytest.raises(ValueError, match="no existe"):
        diff_releases("v0.1.0", "v9.9.9", reports_dir=reports_dir)


def test_release_injects_same_policy_and_custom_splits(monkeypatch, tmp_path):
    from presentation import release as module
    from splits.models import SplitsConfig

    dataset_dir = write_coco_dataset(tmp_path / "dataset", cats=6, dogs=6)
    policy = load_quality_policy(_policy_path(tmp_path))
    policy.duplicate_similarity_threshold.threshold = 1.0
    config = SplitsConfig(train=0.5, val=0.25, test=0.25, seed=73)
    from presentation import gate

    original_analyze = gate.analyze_duplicates
    original_split = gate.split_dataset
    original_report = module.build_splits_report
    original_leakage = gate.analyze_cross_split_leakage
    seen = {"analyze": 0, "split": 0}

    def analyze(contents, duplicate_config):
        seen["analyze"] += 1
        assert duplicate_config.threshold == 1.0
        result = original_analyze(contents, duplicate_config)
        seen["pairs"] = result.details["image_pairs"]
        return result

    def split(coco, passed_config, **kwargs):
        seen["split"] += 1
        assert passed_config is config
        assert kwargs["duplicate_pairs"] is seen["pairs"]
        result = original_split(coco, passed_config, **kwargs)
        seen["result"] = result
        return result

    def leakage(assignments, pairs, config, **kwargs):
        assert assignments is seen["result"].assignments
        assert pairs is seen["pairs"]
        return original_leakage(assignments, pairs, config, **kwargs)

    def report(result, **kwargs):
        assert result is seen["result"]
        return original_report(result, **kwargs)

    def no_reload():
        pytest.fail("Injected splits must not reload YAML")

    monkeypatch.setattr(gate, "load_splits_config", no_reload)
    monkeypatch.setattr(module, "load_splits_config", no_reload)
    monkeypatch.setattr(gate, "analyze_duplicates", analyze)
    monkeypatch.setattr(gate, "split_dataset", split)
    monkeypatch.setattr(gate, "analyze_cross_split_leakage", leakage)
    monkeypatch.setattr(module, "build_splits_report", report)
    module.cut_release(
        "v0.2.0",
        dataset_dir=dataset_dir,
        reports_dir=tmp_path / "reports",
        policy=policy,
        splits_config=config,
    )
    assert seen["analyze"] == seen["split"] == 1
    assert sorted(seen["result"].target_sizes.values()) == [3, 3, 6]


def test_release_custom_similarity_changes_duplicate_components(tmp_path):
    from presentation import release as module
    from splits.models import SplitsConfig

    dataset_dir = write_coco_dataset(tmp_path / "dataset", cats=3, dogs=3)
    policy = load_quality_policy(_policy_path(tmp_path))
    policy.duplicate_similarity_threshold.threshold = 0.0
    # Similarity >= 0 joins all images transitively; no three non-empty splits
    # are possible. The repository default 0.94 would not join all six.
    config = SplitsConfig(train=0.5, val=0.25, test=0.25, seed=9)
    reports_dir = tmp_path / "reports"
    with pytest.raises(ValueError, match="at least three independent image groups"):
        module.cut_release(
            "v0.3.0",
            dataset_dir=dataset_dir,
            reports_dir=reports_dir,
            policy=policy,
            splits_config=config,
        )
    assert not (tmp_path / "reports" / "versions.json").exists()
