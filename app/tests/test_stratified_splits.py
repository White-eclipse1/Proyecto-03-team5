from collections import Counter
from copy import deepcopy
from dataclasses import FrozenInstanceError
from io import BytesIO
from random import getstate, seed

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from analyzers.duplicates import DuplicateConfig, analyze_duplicates
from presentation.contracts import SplitsReport
from presentation.splits import build_splits_report
from splits.models import SplitsConfig, load_splits_config
from splits.stratified import SplitResult, split_dataset, verify_assignment


def dataset(labels):
    annotations = []
    for image_id, categories in enumerate(labels):
        for category in categories:
            annotations.append(
                {
                    "id": len(annotations),
                    "image_id": image_id,
                    "category_id": category,
                    "bbox": [0, 0, 10, 10],
                    "area": 100,
                    "iscrowd": 0,
                }
            )
    return {
        "images": [
            {"id": i, "file_name": f"{i}.png", "width": 64, "height": 64}
            for i in range(len(labels))
        ],
        "categories": [{"id": c, "name": f"class-{c}"} for c in (1, 2, 3)],
        "annotations": annotations,
    }


def config(**overrides):
    return SplitsConfig(train=0.6, val=0.2, test=0.2, **overrides)


def run(coco, cfg=None, pairs=(), contents=None):
    # The pure splitter hashes bytes without decoding; visual detection is separate.
    contents = (
        contents
        if contents is not None
        else {image["id"]: f"unique-{image['id']}".encode() for image in coco["images"]}
    )
    return split_dataset(coco, cfg or config(), image_contents=contents, duplicate_pairs=pairs)


def owner(result, image_id):
    return next(name for name, ids in result.assignments.items() if image_id in ids)


def test_coverage_exclusivity_and_manually_verifiable_stratification():
    coco = dataset([[1]] * 10 + [[2]] * 10)
    result = run(coco)
    assert {name: len(ids) for name, ids in result.assignments.items()} == {
        "train": 12,
        "val": 4,
        "test": 4,
    }
    assert result.class_counts == {
        "train": {1: 6, 2: 6, 3: 0},
        "val": {1: 2, 2: 2, 3: 0},
        "test": {1: 2, 2: 2, 3: 0},
    }
    ids = [i for split in result.assignments.values() for i in split]
    assert sorted(ids) == list(range(20))
    verify_assignment(result.assignments, image_ids=list(range(20)), duplicate_groups=result.groups)


def test_multilabel_and_repeated_boxes_count_images_not_annotations():
    coco = dataset([[1, 1, 2]] * 10)
    result = run(coco)
    for name, ids in result.assignments.items():
        assert result.class_counts[name][1] == len(ids)
        assert result.class_counts[name][2] == len(ids)
        assert result.target_class_counts[name][1] == result.target_class_counts[name][2]
    assert [len(result.assignments[name]) for name in ("train", "val", "test")] == [6, 2, 2]


def test_same_seed_reordering_and_global_rng_independence():
    coco = dataset([[1], [2], [1, 2], []] * 5)
    pairs = [{"image_id_a": 0, "image_id_b": 3}, {"image_id_a": 3, "image_id_b": 5}]
    baseline = run(coco, pairs=pairs)
    state = getstate()
    try:
        seed(919)
        before_rng = getstate()
        assert run(coco, pairs=pairs) == baseline
        assert getstate() == before_rng
    finally:
        from random import setstate

        setstate(state)
    reordered = {key: list(reversed(value)) for key, value in coco.items()}
    assert run(reordered, pairs=list(reversed(pairs))) == baseline


def test_seed_breaks_ties_without_affecting_requested_counts():
    coco = dataset([[1]] * 15)
    first = run(coco, config(seed=1))
    second = run(coco, config(seed=2))
    assert first.assignments != second.assignments
    assert first.class_counts == second.class_counts


@pytest.mark.parametrize(
    "ratios, expected",
    [
        ((0.5, 0.3, 0.2), (10, 6, 4)),
        ((0.2, 0.4, 0.4), (4, 8, 8)),
    ],
)
def test_custom_ratios_control_assignment(ratios, expected):
    cfg = SplitsConfig(train=ratios[0], val=ratios[1], test=ratios[2], seed=7)
    result = run(dataset([[1]] * 20), cfg)
    assert tuple(len(result.assignments[name]) for name in ("train", "val", "test")) == expected


def test_rounding_and_real_yaml_configuration():
    cfg = load_splits_config()
    result = run(dataset([[1]] * 7), cfg)
    assert result.target_sizes == {"train": 5, "val": 1, "test": 1}
    for name in result.assignments:
        assert abs(result.class_counts[name][1] - result.target_class_counts[name][1]) <= 1
    assert sum(result.target_sizes.values()) == 7


def test_same_filename_same_content_is_a_valid_alias():
    coco = dataset([[1]] * 4)
    coco["images"][1]["file_name"] = coco["images"][0]["file_name"]
    result = run(coco, contents={0: b"same", 1: b"same", 2: b"other", 3: b"third"})
    assert (0, 1) in result.groups
    assert owner(result, 0) == owner(result, 1)


def test_same_filename_different_content_is_rejected_deterministically():
    coco = dataset([[1]] * 4)
    coco["images"][1]["file_name"] = coco["images"][0]["file_name"]
    messages = []
    for document in (coco, {key: list(reversed(value)) for key, value in coco.items()}):
        with pytest.raises(ValueError, match="Conflicting content for file_name") as error:
            run(document)
        messages.append(str(error.value))
    assert messages[0] == messages[1]
    assert "image IDs 0 and 1" in messages[0]


def test_same_content_different_filenames_is_an_exact_duplicate():
    coco = dataset([[1]] * 4)
    assert coco["images"][0]["file_name"] != coco["images"][1]["file_name"]
    result = run(coco, contents={0: b"same", 1: b"same", 2: b"other", 3: b"third"})
    assert (0, 1) in result.groups
    assert owner(result, 0) == owner(result, 1)


def test_phash_pairs_form_transitive_indivisible_components():
    coco = dataset([[1]] * 6)
    pairs = [{"image_id_a": 0, "image_id_b": 1}, {"image_id_a": 1, "image_id_b": 2}]
    result = run(coco, pairs=pairs)
    assert (0, 1, 2) in result.groups
    assert len({owner(result, i) for i in (0, 1, 2)}) == 1


def test_existing_phash_detector_recompression_pairs_remain_together():
    image = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(image).rectangle((7, 12, 35, 44), fill="black")
    contents = {}
    for i, quality in enumerate((95, 60)):
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        contents[i] = buffer.getvalue()
    for i, box in ((2, (35, 3, 60, 20)), (3, (3, 38, 23, 60))):
        other = Image.new("RGB", (64, 64), "white")
        ImageDraw.Draw(other).rectangle(box, fill="black")
        buffer = BytesIO()
        other.save(buffer, format="PNG")
        contents[i] = buffer.getvalue()
    detected = analyze_duplicates(contents, DuplicateConfig(threshold=0.94))
    assert len(detected.details["image_pairs"]) == 1
    assert contents[0] != contents[1]
    result = run(dataset([[1]] * 4), pairs=detected.details["image_pairs"], contents=contents)
    assert owner(result, 0) == owner(result, 1)


@pytest.mark.parametrize(
    "assignments, message",
    [
        ({"train": (0, 1), "val": (1, 2), "test": (3,)}, "more than once"),
        ({"train": (0,), "val": (1,), "test": (2,)}, "cover all"),
        ({"train": (0, 1), "val": (2,), "test": (3, 99)}, "Unknown"),
        ({"train": (0, 2), "val": (1,), "test": (3,)}, "Leakage"),
        ({"train": (0, 1, 2), "val": (), "test": (3,)}, "non-empty"),
        ({"train": (0, 1), "validation": (2,), "test": (3,)}, "exactly"),
    ],
)
def test_verifier_rejects_contaminated_assignments(assignments, message):
    with pytest.raises(ValueError, match=message):
        verify_assignment(
            assignments, image_ids=[0, 1, 2, 3], duplicate_groups=[(0, 1), (2,), (3,)]
        )


@pytest.mark.parametrize("groups", [[(0, 1), (2,)], [(0, 1), (1, 2), (3,)], [(), (0, 1, 2, 3)]])
def test_verifier_requires_complete_valid_groups(groups):
    with pytest.raises(ValueError):
        verify_assignment(
            {"train": (0, 1), "val": (2,), "test": (3,)},
            image_ids=[0, 1, 2, 3],
            duplicate_groups=groups,
        )


def test_duplicate_ids_and_unknown_references_are_rejected():
    coco = dataset([[1]] * 4)
    coco["images"][1]["id"] = 0
    with pytest.raises(ValidationError, match="repetidos"):
        run(coco)
    coco = dataset([[1]] * 4)
    coco["annotations"][0]["image_id"] = 99
    with pytest.raises(ValidationError, match="huérfano"):
        run(coco)


@pytest.mark.parametrize(
    "pair",
    [
        {"image_id_a": 0, "image_id_b": 99},
        {"image_id_a": 0, "image_id_b": 0},
        {"image_id_a": True, "image_id_b": 1},
        {"image_id_a": 0},
    ],
)
def test_invalid_duplicate_relations_fail_explicitly(pair):
    coco = dataset([[1]] * 4)
    cfg = config()
    pairs = [pair]
    with pytest.raises(ValueError):
        run(coco, cfg, pairs=pairs)


@pytest.mark.parametrize(
    "contents", [{0: b"a"}, {0: b"a", 1: b"b", 2: b""}, {0: b"a", 1: b"b", 2: "not bytes"}]
)
def test_missing_or_invalid_contents_are_not_silently_ignored(contents):
    coco = dataset([[1]] * 3)
    cfg = config()
    with pytest.raises(ValueError):
        run(coco, cfg, contents=contents)


def test_unannotated_rare_and_empty_categories():
    result = run(dataset([[1], [2], [], [], [], []]))
    assert sum(counts[1] for counts in result.class_counts.values()) == 1
    assert sum(counts[2] for counts in result.class_counts.values()) == 1
    assert all(counts[3] == 0 for counts in result.class_counts.values())
    assert sum(map(len, result.assignments.values())) == 6
    assert all(result.assignments.values())
    empty_labels = run(dataset([[]] * 5))
    assert all(not any(counts.values()) for counts in empty_labels.class_counts.values())


def test_large_group_preserved_even_when_ratios_cannot_be_met():
    pairs = [{"image_id_a": 0, "image_id_b": i} for i in range(1, 8)]
    result = run(dataset([[1]] * 10), pairs=pairs)
    assert sorted(map(len, result.assignments.values())) == [1, 1, 8]
    assert len({owner(result, i) for i in range(8)}) == 1


@pytest.mark.parametrize("count", [1, 2])
def test_too_few_images_fail(count):
    coco = dataset([[1]] * count)
    cfg = config()
    with pytest.raises(ValueError, match="three independent"):
        run(coco, cfg)


def test_too_few_groups_fail_even_with_many_images():
    coco = dataset([[1]] * 8)
    cfg = config()
    contents = dict.fromkeys(range(8), b"same content")
    with pytest.raises(ValueError, match="three independent"):
        run(coco, cfg, contents=contents)


def test_three_groups_with_tiny_ratios_still_make_nonempty_splits():
    cfg = SplitsConfig(train=0.98, val=0.01, test=0.01)
    result = run(dataset([[1]] * 3), cfg)
    assert all(len(ids) == 1 for ids in result.assignments.values())
    assert result.target_sizes == {"train": 3, "val": 0, "test": 0}


def test_no_mutation_and_report_round_trip_v1():
    coco = dataset([[1], [2], [1, 2], [], [2], [1]])
    cfg = config()
    pairs = [{"image_id_a": 0, "image_id_b": 2}]
    contents = {i: str(i).encode() for i in range(6)}
    before = deepcopy((coco, cfg, pairs, contents))
    result = run(coco, cfg, pairs, contents)
    actual = (coco, cfg, pairs, contents)
    expected = before
    assert actual == expected
    report = build_splits_report(result, dataset_version="test-v1")
    assert report.schema_version == "1.0"
    assert report.total_images == 6
    assert set(report.splits.model_dump()) == {"train", "validation", "test"}
    for internal, public in (("train", "train"), ("val", "validation"), ("test", "test")):
        summary = getattr(report.splits, public)
        assert summary.image_count == len(result.assignments[internal])
        assert summary.ratio == len(result.assignments[internal]) / 6
    assert SplitsReport.model_validate_json(report.model_dump_json()) == report


def test_mixed_multilabel_population_stays_within_one_image_of_class_targets():
    result = run(dataset([[1]] * 10 + [[2]] * 10 + [[1, 2]] * 10))
    for name in result.assignments:
        for category in (1, 2):
            assert (
                abs(
                    result.class_counts[name][category] - result.target_class_counts[name][category]
                )
                <= 1
            )
        assert abs(len(result.assignments[name]) - result.target_sizes[name]) <= 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": True},
        {"seed": "42"},
        {"seed": 1.5},
        {"train": float("nan")},
        {"train": float("inf")},
        {"validation": 0.2},
    ],
)
def test_existing_config_rejects_invalid_seed_ratios_and_names(overrides):
    with pytest.raises(ValidationError):
        SplitsConfig.model_validate({"train": 0.6, "val": 0.2, "test": 0.2, **overrides})


def test_negative_integer_seed_is_reproducible():
    coco = dataset([[1], [2], []] * 3)
    expected = run(coco, config(seed=-7))
    actual = run(coco, config(seed=-7))
    assert actual == expected


@pytest.mark.parametrize(
    "field, nested",
    [
        ("assignments", False),
        ("target_sizes", False),
        ("class_counts", False),
        ("class_counts", True),
        ("target_class_counts", False),
        ("target_class_counts", True),
    ],
)
def test_public_result_mappings_reject_mutation(field, nested):
    result = run(dataset([[1]] * 6))
    mapping = getattr(result, field)
    mapping, key = (mapping["train"], 1) if nested else (mapping, "train")
    with pytest.raises(TypeError):
        mapping[key] = 999
    with pytest.raises(TypeError):
        del mapping[key]


def test_id_collections_and_result_attributes_are_immutable():
    result = run(dataset([[1]] * 6))
    with pytest.raises(TypeError):
        result.assignments["train"][0] = 999
    with pytest.raises(TypeError):
        result.groups[0][0] = 999
    with pytest.raises(TypeError):
        result.groups[0] = (999,)
    with pytest.raises(FrozenInstanceError):
        result.assignments = {}


def test_result_defensively_copies_all_constructor_inputs():
    assignments = {"train": [0, 1], "val": [2], "test": [3]}
    groups = [[0, 1], [2], [3]]
    sizes = {"train": 2, "val": 1, "test": 1}
    counts = {"train": {1: 2}, "val": {1: 1}, "test": {1: 1}}
    targets = {"train": {1: 2.0}, "val": {1: 1.0}, "test": {1: 1.0}}
    result = SplitResult(assignments, groups, sizes, counts, targets)
    assignments["train"].append(99)
    assignments.clear()
    groups[0].append(99)
    groups.clear()
    sizes["train"] = 99
    counts["train"][1] = 99
    counts.clear()
    targets["train"][1] = 99
    targets.clear()
    assert result.assignments == {"train": (0, 1), "val": (2,), "test": (3,)}
    assert result.groups == ((0, 1), (2,), (3,))
    assert result.target_sizes == {"train": 2, "val": 1, "test": 1}
    assert result.class_counts == {"train": {1: 2}, "val": {1: 1}, "test": {1: 1}}
    assert result.target_class_counts == {"train": {1: 2.0}, "val": {1: 1.0}, "test": {1: 1.0}}


def assert_counts_match_original_coco(coco, result):
    # Independent oracle: filter annotations by split/category, deduplicate image IDs.
    for name, ids in result.assignments.items():
        for category in coco["categories"]:
            expected = len(
                {
                    annotation["image_id"]
                    for annotation in coco["annotations"]
                    if annotation["image_id"] in ids and annotation["category_id"] == category["id"]
                }
            )
            assert result.class_counts[name][category["id"]] == expected


def test_counts_reconstructed_from_original_annotations_and_assignments():
    coco = dataset([[1, 1, 2], [2, 2], [], [1], [1, 2, 2], [2]] * 3)
    result = run(coco)
    assert_counts_match_original_coco(coco, result)
    # A duplicate annotation must not change the assignment or any reported counts.
    repeated = deepcopy(coco)
    repeated["annotations"].append({**repeated["annotations"][0], "id": 999})
    assert run(repeated) == result


def test_improvement_executes_and_preserves_all_assignment_invariants(monkeypatch):
    import splits.stratified as stratified

    subtractions = []

    class ObservedCounter(Counter):
        def subtract(self, other):
            subtractions.append(dict(other))
            return super().subtract(other)

    # One subtraction per initial group updates remaining labels; additional
    # subtractions occur only when the improvement phase moves a whole group.
    monkeypatch.setattr(stratified, "Counter", ObservedCounter)
    coco = dataset([[1]] * 10 + [[2]] * 10)
    pairs = [{"image_id_a": 0, "image_id_b": 1}]
    result = run(coco, pairs=pairs)
    assert len(subtractions) > len(result.groups), "Fixture must actually trigger improvement"
    ids = [i for members in result.assignments.values() for i in members]
    assert sorted(ids) == list(range(20))
    assert len(ids) == len(set(ids))
    assert all(result.assignments.values())
    assert owner(result, 0) == owner(result, 1)
    assert_counts_match_original_coco(coco, result)
    verify_assignment(result.assignments, image_ids=list(range(20)), duplicate_groups=result.groups)


@pytest.mark.parametrize(
    "seed_value, expected",
    [
        (1, {"train": (0, 2, 4, 6, 8, 9), "val": (1, 5), "test": (3, 7)}),
        (42, {"train": (0, 1, 3, 5, 6, 7), "val": (2, 9), "test": (4, 8)}),
        (-7, {"train": (0, 3, 4, 5, 6, 7), "val": (1, 9), "test": (2, 8)}),
    ],
)
def test_assignments_match_pre_sonar_refactor_reference(seed_value, expected):
    # Captured from commit 470b603 before extracting helpers; expectations are
    # deliberately fixed, not recomputed using the implementation under test.
    coco = dataset([[1]] * 5 + [[2]] * 5)
    actual = dict(run(coco, config(seed=seed_value)).assignments)
    assert actual == expected
