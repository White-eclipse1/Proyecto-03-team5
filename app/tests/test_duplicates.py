from io import BytesIO
from math import nextafter
from pathlib import Path
from random import Random

import imagehash
import pytest
import yaml
from PIL import Image, ImageDraw
from pydantic import ValidationError

from analyzers.duplicates import DuplicateConfig, analyze_duplicates
from policies.duplicates import load_duplicate_config
from policies.models import load_quality_policy
from presentation.contracts import QualityCheck


def encode(image, image_format="PNG", **options):
    output = BytesIO()
    image.save(output, format=image_format, **options)
    return output.getvalue()


def pattern(seed):
    random = Random(seed)
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    for _ in range(20):
        x, y = random.randrange(200), random.randrange(200)
        color = tuple(random.randrange(256) for _ in range(3))
        draw.rectangle((x, y, x + 40, y + 50), fill=color)
    return image


def phash(content):
    with Image.open(BytesIO(content)) as image:
        return imagehash.phash(image, hash_size=8)


def test_exact_copy_and_result_contract():
    content = encode(pattern(1))
    config = load_duplicate_config()
    result = analyze_duplicates({20: content, 10: content}, config)
    assert result.model_dump() == {
        "check_name": "duplicate_images",
        "passed": False,
        "metric_value": 1.0,
        "details": {
            "image_pairs": [
                {"image_id_a": 10, "image_id_b": 20, "hamming_distance": 0, "similarity": 1.0}
            ],
            "hash_bits": 64,
            "similarity_threshold": config.threshold,
            "total_images": 2,
        },
    }
    QualityCheck.model_validate({**result.model_dump(), "action": "warn"})


def test_recompressed_jpeg_with_another_filename(tmp_path):
    original = tmp_path / "original.jpg"
    recompressed = tmp_path / "recompressed-copy.jpg"
    pattern(1).save(original, quality=95)
    with Image.open(original) as image:
        image.save(recompressed, quality=65)
    assert original.read_bytes() != recompressed.read_bytes()
    result = analyze_duplicates(
        {1: original.read_bytes(), 2: recompressed.read_bytes()}, load_duplicate_config()
    )
    assert result.metric_value == 1


def test_reasonable_resize_is_detected():
    image = pattern(1)
    resized = image.resize((128, 128), Image.Resampling.LANCZOS)
    result = analyze_duplicates({1: encode(image), 2: encode(resized)}, load_duplicate_config())
    assert result.metric_value == 1


def test_visually_different_images_are_not_detected():
    result = analyze_duplicates(
        {1: encode(pattern(1)), 2: encode(pattern(2))}, load_duplicate_config()
    )
    assert result.passed
    assert result.metric_value == 0
    assert result.details["image_pairs"] == []


def test_hamming_similarity_and_inclusive_configurable_threshold():
    images = {1: encode(pattern(1)), 2: encode(pattern(2))}
    first, second = phash(images[1]), phash(images[2])
    expected_distance = sum(a != b for a, b in zip(first.hash.flat, second.hash.flat, strict=True))
    assert 0 < expected_distance < 64
    expected_similarity = 1 - expected_distance / 64
    included = analyze_duplicates(images, DuplicateConfig(threshold=expected_similarity))
    pair = included.details["image_pairs"][0]
    assert pair["hamming_distance"] == expected_distance
    assert pair["similarity"] == expected_similarity
    excluded = analyze_duplicates(
        images, DuplicateConfig(threshold=nextafter(expected_similarity, 1.0))
    )
    assert excluded.details["image_pairs"] == []


def test_unique_pairs_no_self_comparisons_determinism_and_no_mutation():
    images = {3: encode(pattern(1)), 1: encode(pattern(1)), 2: encode(pattern(1))}
    original_items = list(images.items())
    result = analyze_duplicates(images, load_duplicate_config())
    assert list(images.items()) == original_items
    pairs = result.details["image_pairs"]
    assert [(pair["image_id_a"], pair["image_id_b"]) for pair in pairs] == [(1, 2), (1, 3), (2, 3)]
    reordered = dict(reversed(original_items))
    assert analyze_duplicates(reordered, load_duplicate_config()) == result


@pytest.mark.parametrize("count", [0, 1])
def test_no_possible_pairs(count):
    images = {i: encode(pattern(i)) for i in range(count)}
    result = analyze_duplicates(images, load_duplicate_config())
    assert result.passed
    assert result.metric_value == 0
    assert result.details["image_pairs"] == []


def test_configuration_from_yaml_and_changed_yaml(tmp_path):
    policy = load_quality_policy()
    assert load_duplicate_config().threshold == policy.duplicate_similarity_threshold.threshold
    document = policy.model_dump()
    document["duplicate_similarity_threshold"]["threshold"] = 0.0
    path = tmp_path / "quality.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    images = {1: encode(pattern(1)), 2: encode(pattern(2))}
    assert analyze_duplicates(images, load_duplicate_config()).metric_value == 0
    assert analyze_duplicates(images, load_duplicate_config(path)).metric_value == 1


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), "0.94"])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValidationError):
        DuplicateConfig(threshold=threshold)


@pytest.mark.parametrize("images", [{-1: b"bad"}, {True: b"bad"}, {1: "path.jpg"}, {1: b"bad"}])
def test_invalid_image_input_is_not_silently_skipped(images):
    with pytest.raises(ValueError):
        analyze_duplicates(images, load_duplicate_config())


def test_real_dataset_jpeg_recompression_in_memory():
    source = Path(__file__).resolve().parents[2] / "data/raw/images/cat.0.jpg"
    if not source.is_file():
        pytest.skip("Real dataset image is not materialized in this checkout")
    original = source.read_bytes()
    with Image.open(BytesIO(original)) as image:
        recompressed = encode(image.convert("RGB"), "JPEG", quality=65)
    assert recompressed != original
    result = analyze_duplicates({1: original, 2: recompressed}, load_duplicate_config())
    assert result.metric_value == 1
    pair = result.details["image_pairs"][0]
    assert pair["similarity"] >= load_duplicate_config().threshold
    assert source.read_bytes() == original
