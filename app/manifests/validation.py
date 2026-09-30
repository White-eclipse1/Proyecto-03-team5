from collections import defaultdict

from manifests.models import ManifestRecord


def _owners_by(records: list[ManifestRecord], field_name: str) -> dict[object, set[str]]:
    owners: dict[object, set[str]] = defaultdict(set)
    for record in records:
        owners[getattr(record, field_name)].add(record.split)
    return owners


def _reject_cross_split(records: list[ManifestRecord], field_name: str) -> None:
    owners = _owners_by(records, field_name)
    leaked = sorted(str(value) for value, splits in owners.items() if len(splits) > 1)
    if leaked:
        raise ValueError(
            f"{field_name} leakage across splits: {', '.join(leaked)}"
        )


def validate_no_leakage(records: list[ManifestRecord]) -> None:
    _reject_cross_split(records, "crop_id")
    _reject_cross_split(records, "source_image_id")
    _reject_cross_split(records, "duplicate_group")
