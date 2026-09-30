import json
from hashlib import sha256
from pathlib import Path

from manifests.generate import generate_manifest
from manifests.models import ManifestProvenance


def _load_duplicate_groups(quality_path: Path) -> list[list[int]]:
    quality = json.loads(quality_path.read_text(encoding="utf-8"))

    check = next(
        (
            item
            for item in quality["checks"]
            if item["check_name"] == "duplicate_similarity_threshold"
        ),
        None,
    )

    if check is None:
        raise ValueError("duplicate_similarity_threshold check not found")

    pairs = check["details"]["image_pairs"]

    parent: dict[int, int] = {}

    def find(image_id: int) -> int:
        parent.setdefault(image_id, image_id)
        while parent[image_id] != image_id:
            parent[image_id] = parent[parent[image_id]]
            image_id = parent[image_id]
        return image_id

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    for pair in pairs:
        union(pair["image_id_a"], pair["image_id_b"])

    groups: dict[int, list[int]] = {}
    for image_id in sorted(parent):
        root = find(image_id)
        groups.setdefault(root, []).append(image_id)

    return [group for group in groups.values() if len(group) > 1]


def build_release_manifest(
    *,
    crops_path: Path,
    quality_path: Path,
    output_path: Path,
    seed: int = 42,
    manifest_version: str = "p3-v1",
) -> None:
    crop_report = json.loads(crops_path.read_text(encoding="utf-8"))

    crops = [
        {
            "crop_id": crop["crop_id"],
            "source_image_id": crop["image_id"],
            "class_name": crop["class_name"],
        }
        for crop in crop_report["crops"]
    ]

    duplicate_groups = _load_duplicate_groups(quality_path)

    source_provenance = crop_report["provenance"]
    provenance = ManifestProvenance(
        release_version=source_provenance["release_version"],
        images_dvc_hash=source_provenance["images_dvc_hash"],
        annotations_dvc_hash=source_provenance["annotations_dvc_hash"],
        quality_report=source_provenance["quality_report"],
        crops_sha256="sha256:" + sha256(crops_path.read_bytes()).hexdigest(),
    )

    manifest = generate_manifest(
        crops=crops,
        dataset_version=crop_report["dataset_version"],
        source_release=source_provenance["release_version"],
        seed=seed,
        provenance=provenance,
        duplicate_groups=duplicate_groups,
        manifest_version=manifest_version,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            manifest.model_dump(by_alias=True),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]

    build_release_manifest(
        crops_path=root / "reports" / "crops.json",
        quality_path=root / "reports" / "releases" / "v0.1.1" / "quality.json",
        output_path=root / "reports" / "releases" / "v0.1.1" / "manifest.json",
    )
