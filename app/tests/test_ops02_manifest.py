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
        source_release="v0.1.1",
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
    assert manifest.source_release == "v0.1.1"
    assert manifest.splits.train.image_count == 420

    record = ManifestRecord(
        crop_id="img1-ann1",
        source_image_id=1,
        duplicate_group="group-1",
        class_name="dog",
        split="train",
    )
    assert record.model_dump(by_alias=True)["class"] == "dog"
    assert "class_name" not in record.model_dump(by_alias=True)


def test_same_seed_produces_same_assignments_and_hash():
    from manifests.generate import generate_manifest

    crops = [
        {
            "crop_id": f"img{image_id}-ann{image_id}",
            "source_image_id": image_id,
            "class_name": "dog" if image_id % 2 else "cat",
        }
        for image_id in range(1, 101)
    ]

    first = generate_manifest(
        crops=crops,
        dataset_version="v0.1.1",
        seed=42,
        duplicate_groups=[],
        manifest_version="p3-v1",
    )
    second = generate_manifest(
        crops=crops,
        dataset_version="v0.1.1",
        seed=42,
        duplicate_groups=[],
        manifest_version="p3-v1",
    )

    assert [record.model_dump() for record in first.records] == [
        record.model_dump() for record in second.records
    ]
    assert first.manifest_hash == second.manifest_hash


def test_source_images_and_duplicate_groups_never_cross_splits():
    from manifests.generate import generate_manifest

    crops = [
        {
            "crop_id": f"img{image_id}-ann{image_id}",
            "source_image_id": image_id,
            "class_name": "dog" if image_id % 2 else "cat",
        }
        for image_id in range(1, 101)
    ]
    crops.append(
        {
            "crop_id": "img1-ann101",
            "source_image_id": 1,
            "class_name": "cat",
        }
    )

    manifest = generate_manifest(
        crops=crops,
        dataset_version="v0.1.1",
        seed=42,
        duplicate_groups=[[2, 3]],
        manifest_version="p3-v1",
    )

    by_source = {}
    by_group = {}

    for record in manifest.records:
        by_source.setdefault(record.source_image_id, set()).add(record.split)
        by_group.setdefault(record.duplicate_group, set()).add(record.split)

    assert all(len(splits) == 1 for splits in by_source.values())
    assert all(len(splits) == 1 for splits in by_group.values())


def test_generator_creates_70_20_10_and_keeps_both_classes_in_validation_and_test():
    from manifests.generate import generate_manifest

    crops = []
    for image_id in range(1, 101):
        class_name = "dog" if image_id % 2 else "cat"
        crops.append(
            {
                "crop_id": f"img{image_id}-ann{image_id}",
                "source_image_id": image_id,
                "class_name": class_name,
            }
        )

    manifest = generate_manifest(
        crops=crops,
        dataset_version="v0.1.1",
        seed=42,
        duplicate_groups=[],
        manifest_version="p3-v1",
    )

    assert manifest.splits.train.image_count == 70
    assert manifest.splits.validation.image_count == 20
    assert manifest.splits.test.image_count == 10

    classes_by_split = {"train": set(), "validation": set(), "test": set()}
    for record in manifest.records:
        classes_by_split[record.split].add(record.class_name)

    assert classes_by_split["validation"] == {"dog", "cat"}
    assert classes_by_split["test"] == {"dog", "cat"}


def test_p3_manifest_is_accepted_by_app_training_contract():
    from presentation.ml_contracts import TrainingManifest

    manifest = P3Manifest(
        schema_version="1.0",
        manifest_version="p3-v1",
        dataset_version="v0.1.1",
        seed=42,
        manifest_hash="sha256:" + ("a" * 64),
        total_images=10,
        splits={
            "train": {"image_count": 7, "ratio": 0.7},
            "validation": {"image_count": 2, "ratio": 0.2},
            "test": {"image_count": 1, "ratio": 0.1},
        },
        records=[
            ManifestRecord(
                crop_id="img1-ann1",
                source_image_id=1,
                duplicate_group="group-1",
                class_name="dog",
                split="train",
            )
        ],
        counts={
            "train": {"dog": 1, "cat": 0},
            "validation": {"dog": 0, "cat": 0},
            "test": {"dog": 0, "cat": 0},
        },
    )

    parsed = TrainingManifest.model_validate(manifest.model_dump())

    assert parsed.manifest_hash == manifest.manifest_hash


def test_rejects_crop_split_outside_five_percentage_point_tolerance():
    from manifests.generate import _validate_crop_split_tolerance

    records = [
        ManifestRecord(
            crop_id=f"img{i}-ann{i}",
            source_image_id=i,
            duplicate_group=f"group-{i}",
            class_name="dog" if i % 2 else "cat",
            split="train" if i <= 80 else ("validation" if i <= 90 else "test"),
        )
        for i in range(1, 101)
    ]

    with pytest.raises(ValueError, match="crop ratio"):
        _validate_crop_split_tolerance(records)


def test_declared_duplicates_share_group_and_split():
    """OPS-10 (rúbrica 7.1): imágenes declaradas duplicadas quedan en el mismo grupo y split.

    El test anterior solo comprueba que cada `duplicate_group` del manifiesto no cruce
    splits; si el generador dejara de unir los duplicados, cada imagen tendría su propio
    grupo y seguiría pasando. Con 30 pares, quedar juntos por azar es imposible.
    """
    from manifests.generate import generate_manifest

    crops = [
        {
            "crop_id": f"img{image_id}-ann{image_id}",
            "source_image_id": image_id,
            "class_name": "dog" if image_id % 2 else "cat",
        }
        for image_id in range(1, 201)
    ]
    pairs = [[image_id, image_id + 100] for image_id in range(1, 61, 2)]

    manifest = generate_manifest(
        crops=crops,
        dataset_version="v0.1.1",
        seed=42,
        duplicate_groups=pairs,
        manifest_version="p3-v1",
    )

    record = {r.source_image_id: r for r in manifest.records}
    for first, second in pairs:
        assert record[first].duplicate_group == record[second].duplicate_group
        assert record[first].split == record[second].split
