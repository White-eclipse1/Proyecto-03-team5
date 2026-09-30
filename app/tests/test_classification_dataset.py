"""ML-02: Dataset/DataLoader de clasificación dog/cat sobre el manifiesto P3 (OPS-02)."""

import json

import pytest
import torch
from PIL import Image, ImageDraw

from classification.dataset import CropClassificationDataset, build_dataloader, load_split
from classification.manifest import CropManifest, load_manifest
from classification.transforms import (
    eval_transform,
    preprocess_image,
    train_transform,
    transform_for,
)
from crops.extract import extract_crops
from presentation.ml_contracts import TrainingParams

DOG, CAT = 3, 4
IMAGE_SIZE = 32
EVAL_SPLITS = ("validation", "test", "inference")


# --- Fixtures: crops reales de ML-01 + manifiesto con el formato de OPS-02 ----


def _crops(tmp_path, count=6):
    """Genera `count` crops alternando dog/cat con la extracción real de ML-01."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    images, annotations = [], []
    for image_id in range(1, count + 1):
        image = Image.new("RGB", (64, 48), color=(20 * image_id, 90, 160))
        ImageDraw.Draw(image).rectangle([8, 8, 40, 30], fill=(255, 255 - 30 * image_id, 0))
        image.save(images_dir / f"{image_id}.png")
        images.append({"id": image_id, "file_name": f"{image_id}.png", "width": 64, "height": 48})
        category = DOG if image_id % 2 else CAT
        annotations.append(
            {
                "id": 100 + image_id,
                "image_id": image_id,
                "category_id": category,
                "bbox": [4, 4, 48, 36],
                "area": 1728.0,
                "iscrowd": 0,
            }
        )
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": DOG, "name": "dog"}, {"id": CAT, "name": "cat"}],
    }
    return extract_crops(
        coco,
        images_dir=images_dir,
        output_dir=tmp_path / "crops",
        dataset_version="v0.1.1",
        provenance={
            "release_version": "v0.1.1",
            "images_dvc_hash": "0" * 32 + ".dir",
            "annotations_dvc_hash": "1" * 32 + ".dir",
            "quality_report": "reports/releases/v0.1.1/quality.json",
        },
    )


def _manifest_payload(report, splits=None):
    splits = splits or ["train", "train", "validation", "validation", "test", "test"]
    return {
        "schema_version": "1.0",
        "manifest_version": "p3-v1.0.0",
        "dataset_version": report.dataset_version,
        "seed": 42,
        "records": [
            {
                "crop_id": crop.crop_id,
                "source_image_id": crop.image_id,
                "duplicate_group": f"g{crop.image_id}",
                "class": crop.class_name,
                "split": split,
            }
            for crop, split in zip(report.crops, splits, strict=True)
        ],
    }


def _dataset(tmp_path, split, report=None, payload=None):
    report = report or _crops(tmp_path)
    manifest = CropManifest.model_validate(payload or _manifest_payload(report))
    return CropClassificationDataset(
        manifest, report, crops_dir=tmp_path / "crops", split=split, image_size=IMAGE_SIZE
    )


def _image():
    image = Image.new("RGB", (80, 60), color=(10, 120, 200))
    ImageDraw.Draw(image).ellipse([10, 5, 60, 50], fill=(250, 200, 0))
    return image


def _transform_names(transform):
    return [type(step).__name__ for step in transform.transforms]


# --- Sin augmentation aleatoria fuera de train (requisito TDD del issue) -------


@pytest.mark.parametrize("split", EVAL_SPLITS)
def test_eval_splits_have_no_random_transforms(split):
    names = _transform_names(transform_for(split, IMAGE_SIZE))

    assert not [name for name in names if name.startswith("Random") or name == "ColorJitter"]


@pytest.mark.parametrize("split", EVAL_SPLITS)
def test_eval_preprocessing_is_identical_under_any_seed(split):
    transform = transform_for(split, IMAGE_SIZE)
    outputs = []
    for seed in range(5):
        torch.manual_seed(seed)
        outputs.append(transform(_image()))

    for output in outputs[1:]:
        assert torch.equal(output, outputs[0])


def test_train_transform_applies_random_augmentation():
    names = _transform_names(transform_for("train", IMAGE_SIZE))
    outputs = []
    for seed in range(5):
        torch.manual_seed(seed)
        outputs.append(train_transform(IMAGE_SIZE)(_image()))

    assert any(name.startswith("Random") for name in names)
    assert any(not torch.equal(output, outputs[0]) for output in outputs[1:])


def test_validation_test_and_inference_share_preprocessing():
    expected = eval_transform(IMAGE_SIZE)(_image())

    for split in EVAL_SPLITS:
        assert torch.equal(transform_for(split, IMAGE_SIZE)(_image()), expected)
    assert torch.equal(preprocess_image(_image(), IMAGE_SIZE), expected)


def test_train_and_eval_end_with_the_same_tensor_format():
    torch.manual_seed(0)
    train = train_transform(IMAGE_SIZE)(_image())
    evaluation = eval_transform(IMAGE_SIZE)(_image())

    assert train.shape == evaluation.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert train.dtype == evaluation.dtype == torch.float32


def test_inference_accepts_non_rgb_images():
    grayscale = _image().convert("L")

    assert preprocess_image(grayscale, IMAGE_SIZE).shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="split"):
        transform_for("holdout", IMAGE_SIZE)


@pytest.mark.parametrize("image_size", [0, 31, 48])
def test_invalid_image_size_is_rejected(image_size):
    with pytest.raises(ValueError, match="image_size"):
        eval_transform(image_size)


# --- Agent Test: una muestra de validation procesada repetidamente -------------


def test_validation_sample_is_deterministic_across_repeated_reads(tmp_path):
    dataset = _dataset(tmp_path, "validation")
    reads = []
    for seed in range(5):
        torch.manual_seed(seed)
        reads.append(dataset[0]["image"])

    for image in reads[1:]:
        assert torch.equal(image, reads[0])


def test_train_sample_is_augmented_across_reads(tmp_path):
    dataset = _dataset(tmp_path, "train")
    reads = []
    for seed in range(5):
        torch.manual_seed(seed)
        reads.append(dataset[0]["image"])

    assert any(not torch.equal(image, reads[0]) for image in reads[1:])


# --- Manifiesto P3 -------------------------------------------------------------


def test_manifest_reads_ops02_records(tmp_path):
    report = _crops(tmp_path)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest_payload(report)), encoding="utf-8")

    manifest = load_manifest(path)

    assert manifest.dataset_version == "v0.1.1"
    record = manifest.records[0]
    assert record.crop_id == report.crops[0].crop_id
    assert record.source_image_id == report.crops[0].image_id
    assert record.class_name == report.crops[0].class_name
    assert record.split == "train"


def test_manifest_tolerates_extra_fields_from_ops02(tmp_path):
    payload = _manifest_payload(_crops(tmp_path))
    payload["manifest_hash"] = "sha256:" + "a" * 64
    payload["records"][0]["annotation_id"] = 101

    assert CropManifest.model_validate(payload).records[0].crop_id


@pytest.mark.parametrize(
    ("field", "value"),
    [("class", "person"), ("split", "holdout"), ("crop_id", "")],
)
def test_manifest_rejects_invalid_records(tmp_path, field, value):
    payload = _manifest_payload(_crops(tmp_path))
    payload["records"][0][field] = value

    with pytest.raises(ValueError):
        CropManifest.model_validate(payload)


def test_manifest_rejects_repeated_crop_id(tmp_path):
    payload = _manifest_payload(_crops(tmp_path))
    payload["records"][1]["crop_id"] = payload["records"][0]["crop_id"]

    with pytest.raises(ValueError, match="crop_id"):
        CropManifest.model_validate(payload)


# --- Dataset -------------------------------------------------------------------


def test_dataset_maps_dog_to_0_and_cat_to_1(tmp_path):
    dataset = _dataset(tmp_path, "train")

    labels = {dataset[i]["class_name"]: dataset[i]["label"] for i in range(len(dataset))}
    assert labels == {"dog": 0, "cat": 1}
    assert dataset.class_to_index == {"dog": 0, "cat": 1}


def test_each_sample_exposes_crop_id_source_image_and_class(tmp_path):
    report = _crops(tmp_path)
    dataset = _dataset(tmp_path, "validation", report=report)
    expected = {crop.crop_id: crop for crop in report.crops}

    for index in range(len(dataset)):
        item = dataset[index]
        info = dataset.sample(index)
        crop = expected[item["crop_id"]]
        assert item["source_image_id"] == info.source_image_id == crop.image_id
        assert item["class_name"] == info.class_name == crop.class_name
        assert info.crop_id == item["crop_id"]
        assert item["image"].shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_dataset_only_contains_its_split_from_the_manifest(tmp_path):
    report = _crops(tmp_path)
    for split in ("train", "validation", "test"):
        dataset = _dataset(tmp_path, split, report=report)
        assert len(dataset) == 2
        assert {dataset.sample(i).split for i in range(len(dataset))} == {split}


def test_dataset_ignores_files_not_listed_in_the_manifest(tmp_path):
    report = _crops(tmp_path)
    Image.new("RGB", (8, 8)).save(tmp_path / "crops" / "dog" / "intruso.png")
    payload = _manifest_payload(report)

    dataset = _dataset(tmp_path, "train", report=report, payload=payload)

    crop_ids = {dataset.sample(i).crop_id for i in range(len(dataset))}
    assert crop_ids == {r["crop_id"] for r in payload["records"] if r["split"] == "train"}


def test_dataset_rejects_crop_missing_from_the_crop_report(tmp_path):
    report = _crops(tmp_path)
    payload = _manifest_payload(report)
    payload["records"][0]["crop_id"] = "img999-ann999"

    with pytest.raises(ValueError, match="img999-ann999"):
        _dataset(tmp_path, "train", report=report, payload=payload)


@pytest.mark.parametrize(("field", "value"), [("class", "cat"), ("source_image_id", 999)])
def test_dataset_rejects_manifest_disagreeing_with_the_crop(tmp_path, field, value):
    report = _crops(tmp_path)
    payload = _manifest_payload(report)
    payload["records"][0][field] = value  # el crop 0 es dog de la imagen 1

    with pytest.raises(ValueError, match=field):
        _dataset(tmp_path, "train", report=report, payload=payload)


def test_dataset_rejects_manifest_from_another_release(tmp_path):
    report = _crops(tmp_path)
    payload = _manifest_payload(report)
    payload["dataset_version"] = "v0.1.0"

    with pytest.raises(ValueError, match=r"v0\.1\.0"):
        _dataset(tmp_path, "train", report=report, payload=payload)


def test_dataset_fails_fast_when_a_crop_file_is_missing(tmp_path):
    report = _crops(tmp_path)
    (tmp_path / "crops" / report.crops[0].crop_path).unlink()

    with pytest.raises(FileNotFoundError, match=report.crops[0].crop_id):
        _dataset(tmp_path, "train", report=report)


def test_load_split_takes_image_size_from_training_params(tmp_path):
    report = _crops(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest_payload(report)), encoding="utf-8")
    report_path = tmp_path / "crops.json"
    report_path.write_text(report.model_dump_json(), encoding="utf-8")
    params = TrainingParams(
        optimizer="adam",
        batch_size=2,
        max_epochs=3,
        learning_rate=0.001,
        image_size=64,
        hidden_layers=[16],
        dropout=0.1,
        seed=7,
        patience=2,
        min_delta=0.0,
    )

    dataset = load_split(
        manifest_path, report_path, crops_dir=tmp_path / "crops", split="test", params=params
    )

    assert dataset[0]["image"].shape == (3, 64, 64)


# --- DataLoader ----------------------------------------------------------------


def test_dataloader_batches_with_metadata(tmp_path):
    loader = build_dataloader(_dataset(tmp_path, "validation"), batch_size=2, seed=0)

    batch = next(iter(loader))

    assert batch["image"].shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["label"].dtype == torch.int64
    assert len(batch["crop_id"]) == 2
    assert len(batch["source_image_id"]) == 2


def _order(loader):
    return [crop_id for batch in loader for crop_id in batch["crop_id"]]


def test_train_loader_shuffles_reproducibly_with_the_seed(tmp_path):
    report = _crops(tmp_path, count=12)
    payload = _manifest_payload(report, splits=["train"] * 12)
    dataset = _dataset(tmp_path, "train", report=report, payload=payload)

    first = _order(build_dataloader(dataset, batch_size=4, seed=1))
    again = _order(build_dataloader(dataset, batch_size=4, seed=1))
    other = _order(build_dataloader(dataset, batch_size=4, seed=2))

    assert first == again
    assert first != other
    assert sorted(first) == sorted(dataset.sample(i).crop_id for i in range(len(dataset)))


def test_eval_loaders_keep_manifest_order(tmp_path):
    report = _crops(tmp_path)
    for split in ("validation", "test"):
        dataset = _dataset(tmp_path, split, report=report)
        order = _order(build_dataloader(dataset, batch_size=1, seed=5))
        assert order == [dataset.sample(i).crop_id for i in range(len(dataset))]
