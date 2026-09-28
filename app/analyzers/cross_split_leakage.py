"""P2-53: measure existing pHash pairs across existing assignments, without I/O."""

from collections.abc import Mapping, Sequence
from math import isclose

from pydantic import BaseModel, ConfigDict, Field

from analyzers.base import AnalyzerResult

SPLITS = ("train", "val", "test")


class LeakageConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    threshold: float = Field(ge=0)
    similarity_threshold: float = Field(ge=0, le=1)


class _Pair(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    image_id_a: int = Field(ge=0)
    image_id_b: int = Field(ge=0)
    hamming_distance: int = Field(ge=0, le=64)
    similarity: float = Field(ge=0, le=1)


def _owners(assignments, image_ids):
    if not isinstance(assignments, Mapping) or set(assignments) != set(SPLITS):
        raise ValueError("Expected exactly train, val and test assignments")
    if any(type(i) is not int or i < 0 for i in image_ids):
        raise ValueError("Expected non-negative integer image IDs")
    expected = set(image_ids)
    if len(expected) != len(image_ids):
        raise ValueError("Duplicate image IDs in input")
    owners = {}
    for split in SPLITS:
        ids = assignments[split]
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)) or not ids:
            raise ValueError("Each split must contain a non-empty sequence of IDs")
        for image_id in ids:
            if type(image_id) is not int or image_id not in expected:
                raise ValueError(f"Unknown image ID in assignment: {image_id}")
            if image_id in owners:
                raise ValueError(f"Image ID assigned more than once: {image_id}")
            owners[image_id] = split
    if set(owners) != expected:
        raise ValueError("Assignments do not cover all image IDs")
    return owners


def _unique_pairs(duplicate_pairs, owners, config):
    unique = {}
    for raw in duplicate_pairs:
        pair = _Pair.model_validate(raw)
        a, b = sorted((pair.image_id_a, pair.image_id_b))
        if a == b or a not in owners or b not in owners:
            raise ValueError("Pair must reference two distinct assigned image IDs")
        if not isclose(pair.similarity, 1 - pair.hamming_distance / 64, abs_tol=1e-12, rel_tol=0):
            raise ValueError("Pair similarity is inconsistent with its 64-bit Hamming distance")
        if pair.similarity < config.similarity_threshold:
            raise ValueError("Pair is below the configured pHash detection threshold")
        metadata = (pair.hamming_distance, pair.similarity)
        if (a, b) in unique and unique[a, b] != metadata:
            raise ValueError("Conflicting metadata for a duplicate pair")
        unique[a, b] = metadata
    return unique


def analyze_cross_split_leakage(
    assignments: Mapping[str, Sequence[int]],
    duplicate_pairs: Sequence[Mapping],
    config: LeakageConfig,
    *,
    image_ids: Sequence[int],
) -> AnalyzerResult:
    """Count unique direct pHash edges crossing splits, not transitive components.

    The caller supplies the complete detector output from these same images.
    COCO IDs provide the expected universe, including images without any pairs.
    Invalid coverage raises; valid contaminated assignments produce a failed check.
    """
    owners = _owners(assignments, image_ids)
    unique = _unique_pairs(duplicate_pairs, owners, config)
    contaminated = [
        {
            "image_id_a": a,
            "image_id_b": b,
            "split_a": owners[a],
            "split_b": owners[b],
            "hamming_distance": distance,
            "similarity": similarity,
        }
        for (a, b), (distance, similarity) in sorted(unique.items())
        if owners[a] != owners[b]
    ]
    return AnalyzerResult(
        check_name="cross_split_leakage",
        metric_value=float(len(contaminated)),
        passed=len(contaminated) <= config.threshold,
        details={
            "image_pairs": contaminated,
            "total_pairs_evaluated": len(unique),
            "total_images": len(owners),
            "images_per_split": {name: len(assignments[name]) for name in SPLITS},
            "similarity_threshold": config.similarity_threshold,
            "hash_bits": 64,
        },
    )
