"""P2-32: partición multietiqueta por grupos indivisibles, sin I/O."""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isclose
from random import Random
from types import MappingProxyType

from ingestion.models import CocoDataset
from splits.models import SplitsConfig

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SplitResult:
    """Resultado en memoria; los conteos por clase son de imágenes, no de cajas."""

    assignments: Mapping[str, tuple[int, ...]]
    groups: tuple[tuple[int, ...], ...]
    target_sizes: Mapping[str, int]
    class_counts: Mapping[str, Mapping[int, int]]
    target_class_counts: Mapping[str, Mapping[int, float]]

    def __post_init__(self):
        # Copy before wrapping: a proxy alone would still expose mutations of its source.
        object.__setattr__(
            self,
            "assignments",
            MappingProxyType({name: tuple(ids) for name, ids in self.assignments.items()}),
        )
        object.__setattr__(self, "groups", tuple(tuple(group) for group in self.groups))
        object.__setattr__(self, "target_sizes", MappingProxyType(dict(self.target_sizes)))
        for field in ("class_counts", "target_class_counts"):
            object.__setattr__(
                self,
                field,
                MappingProxyType(
                    {
                        name: MappingProxyType(dict(counts))
                        for name, counts in getattr(self, field).items()
                    }
                ),
            )


def verify_assignment(
    assignments: Mapping[str, Sequence[int]],
    *,
    image_ids: Sequence[int],
    duplicate_groups: Sequence[Sequence[int]],
) -> None:
    """Rechaza omisiones, IDs extra/repetidos, splits vacíos y grupos separados.

    El llamador debe suministrar los grupos completos calculados desde la entrada,
    no reconstruirlos desde una asignación potencialmente contaminada.
    """
    if set(assignments) != set(SPLIT_NAMES):
        raise ValueError("Expected exactly train, val and test assignments")
    if any(type(i) is not int or i < 0 for i in image_ids):
        raise ValueError("Expected non-negative integer image IDs")
    expected = set(image_ids)
    if len(expected) != len(image_ids):
        raise ValueError("Duplicate image IDs in input")
    owner = _assignment_owners(assignments, expected)
    _verify_groups(duplicate_groups, expected, owner)


def _assignment_owners(assignments, expected):
    owner = {}
    for name in SPLIT_NAMES:
        if not assignments[name]:
            raise ValueError("All three splits must be non-empty")
        for image_id in assignments[name]:
            if type(image_id) is not int or image_id not in expected:
                raise ValueError(f"Unknown image ID in assignment: {image_id}")
            if image_id in owner:
                raise ValueError(f"Image ID assigned more than once: {image_id}")
            owner[image_id] = name
    if set(owner) != expected:
        raise ValueError("Assignment does not cover all input images")
    return owner


def _verify_groups(duplicate_groups, expected, owner):
    grouped = set()
    for group in duplicate_groups:
        if not group:
            raise ValueError("Duplicate groups must be non-empty")
        for image_id in group:
            if type(image_id) is not int or image_id not in expected or image_id in grouped:
                raise ValueError("Duplicate groups must partition the input image IDs")
            grouped.add(image_id)
        if len({owner[image_id] for image_id in group}) != 1:
            raise ValueError("Leakage: a duplicate group crosses splits")
    if grouped != expected:
        raise ValueError("Duplicate groups do not cover all input images")


def _groups(coco, image_contents, duplicate_pairs):
    ids = sorted(image.id for image in coco.images)
    if any(type(i) is not int for i in image_contents) or set(image_contents) != set(ids):
        raise ValueError("image_contents must cover exactly the COCO image IDs")
    parent = {i: i for i in ids}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)

    _merge_content(coco, image_contents, union)
    _merge_pairs(duplicate_pairs, parent, union)
    components = {}
    for i in ids:
        components.setdefault(root(i), []).append(i)
    return tuple(sorted(tuple(group) for group in components.values()))


def _merge_content(coco, image_contents, union):
    by_hash, by_filename = {}, {}
    for image in sorted(coco.images, key=lambda image: image.id):
        content = image_contents[image.id]
        if not isinstance(content, bytes) or not content:
            raise ValueError(f"Expected non-empty encoded bytes for image {image.id}")
        digest = sha256(content).digest()
        if digest in by_hash:
            union(image.id, by_hash[digest])
        by_hash[digest] = image.id
        _merge_filename(image, digest, by_filename, union)


def _merge_filename(image, digest, by_filename, union):
    # One filename resolves to one file under the caller's dataset root.
    if image.file_name in by_filename:
        previous_id, previous_digest = by_filename[image.file_name]
        if previous_digest != digest:
            raise ValueError(
                f"Conflicting content for file_name {image.file_name!r}: "
                f"image IDs {previous_id} and {image.id}"
            )
        union(image.id, previous_id)
    else:
        by_filename[image.file_name] = (image.id, digest)


def _merge_pairs(duplicate_pairs, parent, union):
    for pair in duplicate_pairs:
        if not isinstance(pair, Mapping) or not {"image_id_a", "image_id_b"} <= pair.keys():
            raise ValueError("Each duplicate pair requires image_id_a and image_id_b")
        a, b = pair["image_id_a"], pair["image_id_b"]
        if any(type(i) is not int or i not in parent for i in (a, b)):
            raise ValueError("Duplicate pair references an unknown image ID")
        if a == b:
            raise ValueError("A duplicate pair must reference two distinct images")
        union(a, b)


def split_dataset(
    coco: CocoDataset | dict,
    config: SplitsConfig,
    *,
    image_contents: Mapping[int, bytes],
    duplicate_pairs: Sequence[Mapping],
) -> SplitResult:
    """Recibe bytes y pares de `analyze_duplicates(...).details['image_pairs']`.

    SHA-256 cubre copias exactas; la garantía visual depende de que el llamador
    proporcione todos los pares del detector sobre estas mismas imágenes.
    No decodifica píxeles ni ejecuta pHash, y nunca modifica las entradas.
    """
    # Revalidate even models constructed/mutated outside normal Pydantic validation.
    coco = CocoDataset.model_validate(coco.model_dump() if isinstance(coco, CocoDataset) else coco)
    config = SplitsConfig.model_validate(config.model_dump())
    groups = _groups(coco, image_contents, duplicate_pairs)
    if len(groups) < 3:
        raise ValueError("Need at least three independent image groups for three non-empty splits")
    ids = sorted(image.id for image in coco.images)
    category_ids = sorted(category.id for category in coco.categories)
    labels = {i: set() for i in ids}
    for annotation in coco.annotations:
        labels[annotation.image_id].add(annotation.category_id)
    totals = Counter(c for i in ids for c in labels[i])
    group_counts = {group: Counter(c for i in group for c in labels[i]) for group in groups}
    ratios = {name: getattr(config, name) for name in SPLIT_NAMES}
    # Normalize only the <=1e-6 floating-point slack permitted by SplitsConfig.
    total_ratio = sum(ratios.values())
    ratios = {name: value / total_ratio for name, value in ratios.items()}
    targets = {name: ratios[name] * len(ids) for name in SPLIT_NAMES}
    target_sizes = {name: int(targets[name]) for name in SPLIT_NAMES}
    # Dataset tie-breaking only; never used for secrets, tokens or cryptography.
    rng = Random(config.seed)

    def choose_tie(items):
        return items[rng.randrange(len(items))] if len(items) > 1 else items[0]

    remainder_order = list(SPLIT_NAMES)
    rng.shuffle(remainder_order)  # Seed affects only equal fractional remainders.
    remainder_order.sort(key=lambda name: -(targets[name] - target_sizes[name]))
    for name in remainder_order[: len(ids) - sum(target_sizes.values())]:
        target_sizes[name] += 1
    target_classes = {
        name: {c: ratios[name] * totals[c] for c in category_ids} for name in SPLIT_NAMES
    }
    assignments = {name: [] for name in SPLIT_NAMES}
    counts = {name: Counter() for name in SPLIT_NAMES}

    def cost(name, group, direction=1):
        # Increment of global squared deviation, normalized by dataset/class size.
        size_error = len(assignments[name]) - target_sizes[name]
        delta = ((size_error + direction * len(group)) ** 2 - size_error**2) / len(ids)
        for c, count in group_counts[group].items():
            error = counts[name][c] - target_classes[name][c]
            delta += ((error + direction * count) ** 2 - error**2) / totals[c]
        return delta

    _assign_initial(groups, totals, group_counts, assignments, counts, cost, choose_tie)
    _improve_assignment(groups, assignments, counts, group_counts, cost, choose_tie)

    result = SplitResult(
        assignments={name: tuple(sorted(values)) for name, values in assignments.items()},
        groups=groups,
        target_sizes=target_sizes,
        class_counts={name: {c: counts[name][c] for c in category_ids} for name in SPLIT_NAMES},
        target_class_counts=target_classes,
    )
    verify_assignment(result.assignments, image_ids=ids, duplicate_groups=groups)
    return result


def _assign_initial(groups, totals, group_counts, assignments, counts, cost, choose_tie):
    pending = list(groups)
    remaining = totals.copy()
    while pending:
        # Rarest remaining label first; largest indivisible groups break that tie.
        def priority(group):
            rarity = min((remaining[c] for c in group_counts[group]), default=float("inf"))
            return rarity, -len(group)

        best_priority = min(priority(group) for group in pending)
        group = choose_tie([g for g in pending if priority(g) == best_priority])
        empty = [name for name in SPLIT_NAMES if not assignments[name]]
        candidates = empty if len(pending) == len(empty) else list(SPLIT_NAMES)

        costs = {name: cost(name, group) for name in candidates}
        best_cost = min(costs.values())
        name = choose_tie(
            [
                name
                for name in candidates
                if isclose(costs[name], best_cost, abs_tol=1e-12, rel_tol=0)
            ]
        )
        assignments[name].extend(group)
        counts[name].update(group_counts[group])
        remaining.subtract(group_counts[group])
        pending.remove(group)


def _improving_moves(groups, assignments, cost):
    moves = []
    for group in groups:
        source = next(name for name in SPLIT_NAMES if group[0] in assignments[name])
        if len(assignments[source]) == len(group):
            continue
        for destination in SPLIT_NAMES:
            if destination != source:
                delta = cost(source, group, -1) + cost(destination, group)
                if delta < -1e-12:
                    moves.append((delta, group, source, destination))
    return moves


def _improve_assignment(groups, assignments, counts, group_counts, cost, choose_tie):
    # Whole-group moves repair avoidable rounding/greedy imbalance. Each move
    # strictly reduces the objective and preserves non-empty splits; no group splits.
    while True:
        moves = _improving_moves(groups, assignments, cost)
        if not moves:
            break
        best_delta = min(move[0] for move in moves)
        _, group, source, destination = choose_tie(
            [move for move in moves if isclose(move[0], best_delta, abs_tol=1e-12, rel_tol=0)]
        )
        for image_id in group:
            assignments[source].remove(image_id)
        assignments[destination].extend(group)
        counts[source].subtract(group_counts[group])
        counts[destination].update(group_counts[group])
