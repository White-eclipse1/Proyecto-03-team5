import pytest
from manifests.models import ManifestRecord, P3Manifest
from manifests.validation import validate_no_leakage


def test_rejects_crop_id_leakage_across_splits():
    records = [
        ManifestRecord(
            crop_id="img1-ann1",
            source_image_id=1,
            duplicate_group="group-1",
            class_name="dog",
            split="train",
        ),
        ManifestRecord(
            crop_id="img1-ann1",
            source_image_id=2,
            duplicate_group="group-2",
            class_name="dog",
            split="validation",
        ),
    ]

    with pytest.raises(ValueError, match="crop_id"):
        validate_no_leakage(records)


def test_rejects_source_image_leakage_across_splits():
    records = [
        ManifestRecord(
            crop_id="img1-ann1",
            source_image_id=1,
            duplicate_group="group-1",
            class_name="dog",
            split="train",
        ),
        ManifestRecord(
            crop_id="img1-ann2",
            source_image_id=1,
            duplicate_group="group-1",
            class_name="cat",
            split="test",
        ),
    ]

    with pytest.raises(ValueError, match="source_image_id"):
        validate_no_leakage(records)


def test_rejects_duplicate_group_leakage_across_splits():
    records = [
        ManifestRecord(
            crop_id="img1-ann1",
            source_image_id=1,
            duplicate_group="group-10",
            class_name="dog",
            split="train",
        ),
        ManifestRecord(
            crop_id="img2-ann2",
            source_image_id=2,
            duplicate_group="group-10",
            class_name="dog",
            split="validation",
        ),
    ]

    with pytest.raises(ValueError, match="duplicate_group"):
        validate_no_leakage(records)


def test_manifest_extends_training_manifest_contract():
    manifest = P3Manifest(
        schema_version="1.0",
        manifest_version="p3-v1",
        dataset_version="v0.1.1",
        seed=42,
        manifest_hash="sha256:" + ("a" * 64),
        total_images=600,
        splits={
            "train": {"image_count": 420, "ratio": 0.70},
            "validation": {"image_count": 120, "ratio": 0.20},
            "test": {"image_count": 60, "ratio": 0.10},
        },
        records=[],
        counts={},
    )

    assert manifest.manifest_version == "p3-v1"
    assert manifest.dataset_version == "v0.1.1"
    assert manifest.splits.train.image_count == 420
