import copy

import pytest

from analyzers.cross_split_leakage import LeakageConfig, analyze_cross_split_leakage

ASSIGNMENTS = {"train": (1, 2), "val": (3,), "test": (4,)}


def pair(a, b, distance=0):
    return {
        "image_id_a": a,
        "image_id_b": b,
        "hamming_distance": distance,
        "similarity": 1 - distance / 64,
    }


def analyze(pairs, assignments=None, threshold=0):
    return analyze_cross_split_leakage(
        ASSIGNMENTS if assignments is None else assignments,
        pairs,
        LeakageConfig(threshold=threshold, similarity_threshold=0.94),
        image_ids=[1, 2, 3, 4],
    )


@pytest.mark.parametrize("pairs", [[], [pair(1, 2)]])
def test_zero_leakage(pairs):
    result = analyze(pairs)
    assert result.passed
    assert result.metric_value == 0
    assert result.details["total_pairs_evaluated"] == len(pairs)


def test_crossing_pair_is_a_failed_check_not_an_exception():
    result = analyze([pair(1, 3, 2)])
    assert not result.passed
    assert result.metric_value == 1
    assert result.details["image_pairs"] == [
        {**pair(1, 3, 2), "split_a": "train", "split_b": "val"}
    ]
    assert result.details["images_per_split"] == {"train": 2, "val": 1, "test": 1}
    assert result.details["similarity_threshold"] == 0.94


def test_unique_direct_pairs_not_transitive_components():
    result = analyze([pair(1, 3), pair(3, 4), pair(3, 1), pair(1, 2)])
    assert result.metric_value == 2  # Do not invent an edge 1-4.
    assert result.details["total_pairs_evaluated"] == 3
    assert analyze([pair(1, 3), pair(3, 4)], threshold=2).passed
    assert not analyze([pair(1, 3), pair(3, 4)], threshold=1).passed


def test_determinism_and_purity():
    assignments = copy.deepcopy(ASSIGNMENTS)
    pairs = [pair(3, 1), pair(4, 2)]
    before = copy.deepcopy((assignments, pairs))
    first = analyze(pairs, assignments)
    second = analyze(list(reversed(pairs)), dict(reversed(list(assignments.items()))))
    assert first == second
    assert (assignments, pairs) == before


@pytest.mark.parametrize(
    "assignments",
    [
        {"train": [1, 2], "val": [2, 3], "test": [4]},
        {"train": [1], "val": [3], "test": [4]},
        {"train": [1, 2, 99], "val": [3], "test": [4]},
        {"train": [1, 2], "validation": [3], "test": [4]},
        {"train": [1, 2, 3], "val": [], "test": [4]},
        {"train": "12", "val": [3], "test": [4]},
        {"train": [True, 2], "val": [3], "test": [4]},
    ],
)
def test_invalid_assignments_raise(assignments):
    with pytest.raises(ValueError):
        analyze([], assignments)


@pytest.mark.parametrize(
    "pairs",
    [
        [pair(1, 99)],
        [pair(1, 1)],
        [pair(1, 3, 10)],
        [{**pair(1, 3), "similarity": 0.96}],
        [pair(1, 3), pair(3, 1, 1)],
    ],
)
def test_invalid_pairs_raise(pairs):
    with pytest.raises(ValueError):
        analyze(pairs)
