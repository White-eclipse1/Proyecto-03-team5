import json
from collections import defaultdict
from hashlib import sha256
from random import Random

from manifests.models import (
    ManifestRecord,
    ManifestSplit,
    ManifestSplits,
    P3Manifest,
)
from manifests.validation import validate_no_leakage

SPLIT_TARGETS = {
    "train": 0.70,
    "validation": 0.20,
    "test": 0.10,
}


def _build_duplicate_group_map(
    source_image_ids: list[int],
    duplicate_groups: list[list[int]],
) -> dict[int, str]:
    parent = {image_id: image_id for image_id in source_image_ids}

    def find(image_id: int) -> int:
        while parent[image_id] != image_id:
            parent[image_id] = parent[parent[image_id]]
            image_id = parent[image_id]
        return image_id

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    known = set(source_image_ids)

    for group in duplicate_groups:
        if not group:
            raise ValueError("duplicate_groups cannot contain empty groups")
        if any(image_id not in known for image_id in group):
            raise ValueError("duplicate_group references unknown source_image_id")
        first = group[0]
        for image_id in group[1:]:
            union(first, image_id)

    members: dict[int, list[int]] = defaultdict(list)
    for image_id in source_image_ids:
        members[find(image_id)].append(image_id)

    result = {}
    for group_members in members.values():
        group_members = sorted(group_members)
        group_id = "group-" + "-".join(str(image_id) for image_id in group_members)
        for image_id in group_members:
            result[image_id] = group_id

    return result


def _allocate_groups(
    groups: dict[str, list[int]],
    *,
    seed: int,
) -> dict[int, str]:
    rng = Random(seed)

    items = list(groups.items())
    items.sort(key=lambda item: item[0])
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)

    total_images = sum(len(images) for _, images in items)
    targets = {split: SPLIT_TARGETS[split] * total_images for split in SPLIT_TARGETS}

    counts = dict.fromkeys(SPLIT_TARGETS, 0)
    assignments: dict[int, str] = {}

    for _, image_ids in items:
        best_split = min(
            SPLIT_TARGETS,
            key=lambda split: (
                abs((counts[split] + len(image_ids)) - targets[split]),
                counts[split] / max(targets[split], 1),
                split,
            ),
        )
        for image_id in image_ids:
            assignments[image_id] = best_split
        counts[best_split] += len(image_ids)

    return assignments


def _rebalance_if_needed(
    assignments: dict[int, str],
    group_map: dict[int, str],
    *,
    seed: int,
) -> dict[int, str]:
    groups: dict[str, list[int]] = defaultdict(list)
    for image_id, group_id in group_map.items():
        groups[group_id].append(image_id)

    total = len(assignments)
    desired = {
        "train": round(total * 0.70),
        "validation": round(total * 0.20),
    }
    desired["test"] = total - desired["train"] - desired["validation"]

    counts = {
        split: sum(1 for value in assignments.values() if value == split) for split in SPLIT_TARGETS
    }

    rng = Random(seed)

    while counts != desired:
        oversized = [split for split in SPLIT_TARGETS if counts[split] > desired[split]]
        undersized = [split for split in SPLIT_TARGETS if counts[split] < desired[split]]

        if not oversized or not undersized:
            break

        moved = False

        source_groups = []
        for group_id, image_ids in groups.items():
            owner = assignments[image_ids[0]]
            if owner in oversized:
                source_groups.append((group_id, image_ids, owner))

        rng.shuffle(source_groups)
        source_groups.sort(key=lambda item: len(item[1]))

        for _, image_ids, source in source_groups:
            for destination in undersized:
                size = len(image_ids)
                if (
                    counts[source] - size >= desired[source]
                    and counts[destination] + size <= desired[destination]
                ):
                    for image_id in image_ids:
                        assignments[image_id] = destination
                    counts[source] -= size
                    counts[destination] += size
                    moved = True
                    break
            if moved:
                break

        if not moved:
            break

    return assignments


def _ensure_class_presence(
    assignments: dict[int, str],
    crops_by_image: dict[int, list[dict]],
    group_map: dict[int, str],
) -> None:
    classes_by_group: dict[str, set[str]] = defaultdict(set)
    images_by_group: dict[str, list[int]] = defaultdict(list)

    for image_id, group_id in group_map.items():
        images_by_group[group_id].append(image_id)
        for crop in crops_by_image[image_id]:
            classes_by_group[group_id].add(crop["class_name"])

    group_owner = {
        group_id: assignments[image_ids[0]] for group_id, image_ids in images_by_group.items()
    }

    for target_split in ("validation", "test"):
        present = set()
        for group_id, owner in group_owner.items():
            if owner == target_split:
                present.update(classes_by_group[group_id])

        for needed_class in ("dog", "cat"):
            if needed_class in present:
                continue

            candidate = next(
                (
                    group_id
                    for group_id, owner in group_owner.items()
                    if owner == "train" and needed_class in classes_by_group[group_id]
                ),
                None,
            )

            if candidate is None:
                continue

            image_ids = images_by_group[candidate]
            for image_id in image_ids:
                assignments[image_id] = target_split

            group_owner[candidate] = target_split
            present.update(classes_by_group[candidate])


def _manifest_hash_payload(
    *,
    dataset_version: str,
    source_release: str,
    manifest_version: str,
    seed: int,
    records: list[ManifestRecord],
) -> str:
    payload = {
        "dataset_version": dataset_version,
        "source_release": source_release,
        "manifest_version": manifest_version,
        "seed": seed,
        "records": [
            record.model_dump() for record in sorted(records, key=lambda record: record.crop_id)
        ],
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return "sha256:" + sha256(encoded).hexdigest()


def generate_manifest(
    *,
    crops: list[dict],
    dataset_version: str,
    source_release: str | None = None,
    seed: int,
    duplicate_groups: list[list[int]],
    manifest_version: str,
) -> P3Manifest:
    if not crops:
        raise ValueError("crops cannot be empty")

    crop_ids = [crop["crop_id"] for crop in crops]
    if len(crop_ids) != len(set(crop_ids)):
        raise ValueError("crop_id must be unique")

    source_image_ids = sorted({crop["source_image_id"] for crop in crops})

    crops_by_image: dict[int, list[dict]] = defaultdict(list)
    for crop in crops:
        crops_by_image[crop["source_image_id"]].append(crop)

    group_map = _build_duplicate_group_map(
        source_image_ids,
        duplicate_groups,
    )

    groups: dict[str, list[int]] = defaultdict(list)
    for image_id, group_id in group_map.items():
        groups[group_id].append(image_id)

    assignments = _allocate_groups(groups, seed=seed)
    assignments = _rebalance_if_needed(
        assignments,
        group_map,
        seed=seed,
    )
    _ensure_class_presence(
        assignments,
        crops_by_image,
        group_map,
    )

    records = [
        ManifestRecord(
            crop_id=crop["crop_id"],
            source_image_id=crop["source_image_id"],
            duplicate_group=group_map[crop["source_image_id"]],
            class_name=crop["class_name"],
            split=assignments[crop["source_image_id"]],
        )
        for crop in sorted(crops, key=lambda item: item["crop_id"])
    ]

    validate_no_leakage(records)

    image_counts = {
        split: len({record.source_image_id for record in records if record.split == split})
        for split in SPLIT_TARGETS
    }

    total_images = len(source_image_ids)

    counts = {
        split: {
            class_name: sum(
                1 for record in records if record.split == split and record.class_name == class_name
            )
            for class_name in ("dog", "cat")
        }
        for split in SPLIT_TARGETS
    }

    source_release = source_release or dataset_version

    manifest_hash = _manifest_hash_payload(
        dataset_version=dataset_version,
        source_release=source_release,
        manifest_version=manifest_version,
        seed=seed,
        records=records,
    )

    return P3Manifest(
        schema_version="1.0",
        manifest_version=manifest_version,
        dataset_version=dataset_version,
        source_release=source_release,
        seed=seed,
        manifest_hash=manifest_hash,
        total_images=total_images,
        splits=ManifestSplits(
            train=ManifestSplit(
                image_count=image_counts["train"],
                ratio=image_counts["train"] / total_images,
            ),
            validation=ManifestSplit(
                image_count=image_counts["validation"],
                ratio=image_counts["validation"] / total_images,
            ),
            test=ManifestSplit(
                image_count=image_counts["test"],
                ratio=image_counts["test"] / total_images,
            ),
        ),
        records=records,
        counts=counts,
    )
