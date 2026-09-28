"""Adaptación del resultado en memoria al resumen público v1.0; sin I/O."""

from presentation.contracts import DatasetSplits, SplitsReport, SplitSummary
from splits.stratified import SplitResult, verify_assignment


def build_splits_report(result: SplitResult, *, dataset_version: str) -> SplitsReport:
    image_ids = [image_id for group in result.groups for image_id in group]
    verify_assignment(result.assignments, image_ids=image_ids, duplicate_groups=result.groups)
    total = len(image_ids)
    summaries = {
        public: SplitSummary(
            image_count=len(result.assignments[internal]),
            ratio=len(result.assignments[internal]) / total,
        )
        for internal, public in (("train", "train"), ("val", "validation"), ("test", "test"))
    }
    class_distribution = {
        public: {
            str(category_id): count for category_id, count in result.class_counts[internal].items()
        }
        for internal, public in (("train", "train"), ("val", "validation"), ("test", "test"))
    }
    leakage = {
        "status": "passed",
        "checked_groups": len(result.groups),
        "duplicate_groups": sum(len(group) > 1 for group in result.groups),
        "cross_split_groups": 0,
        "coverage": total,
    }
    return SplitsReport(
        schema_version="1.0",
        dataset_version=dataset_version,
        total_images=total,
        splits=DatasetSplits(**summaries),
        class_distribution=class_distribution,
        leakage=leakage,
    )
