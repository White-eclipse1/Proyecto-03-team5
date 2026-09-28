"""RGB pixels, PCA and t-SNE. No filesystem, network or environment access."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ingestion.models import CocoDataset
from projections.models import FeatureConfig, ProjectionsConfig


def extract_features(contents: Mapping[int, bytes], config: FeatureConfig) -> np.ndarray:
    """One row per sorted COCO image ID; RGB 16x16, EXIF transpose, Lanczos, /255."""
    rows = []
    for image_id in sorted(contents):
        try:
            with Image.open(BytesIO(contents[image_id])) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                reduced = rgb.resize((config.width, config.height), Image.Resampling.LANCZOS)
                rows.append(np.asarray(reduced, dtype=np.float64).reshape(-1) / 255.0)
        except (OSError, ValueError, TypeError, Image.DecompressionBombError) as error:
            raise ValueError(f"Cannot decode COCO image_id={image_id}") from error
    if not rows:
        raise ValueError("No images to project")
    return np.stack(rows)


def dataset_fingerprint(coco: CocoDataset, contents: Mapping[int, bytes]) -> str:
    """SHA-256 content identity, not a credential/security primitive.

    Canonical validated COCO (all fields, arrays sorted by ID) plus encoded
    image SHA-256 digests in ID order. No absolute directories or timestamps.
    """
    canonical = coco.model_dump()
    for collection in ("images", "annotations", "categories"):
        canonical[collection] = sorted(canonical[collection], key=lambda item: item["id"])
    payload = {
        "coco": canonical,
        "image_sha256": [
            {"image_id": image_id, "sha256": hashlib.sha256(contents[image_id]).hexdigest()}
            for image_id in sorted(contents)
        ],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProjectionCoordinates:
    image_ids: tuple[int, ...]
    pca: np.ndarray
    tsne: np.ndarray
    explained_variance_ratio: tuple[float, float]
    fingerprint: str


def compute_projections(
    coco: CocoDataset, contents: Mapping[int, bytes], config: ProjectionsConfig
) -> ProjectionCoordinates:
    ids = tuple(sorted(image.id for image in coco.images))
    if set(contents) != set(ids):
        raise ValueError("Image bytes must cover exactly the COCO image IDs")
    if len(ids) < 3:
        raise ValueError("At least three images are required for a two-dimensional projection")
    if config.tsne.perplexity >= len(ids):
        raise ValueError("t-SNE perplexity must be smaller than the number of images")
    features = extract_features(contents, config.features)
    if not np.isfinite(features).all():
        raise ValueError("Image features must be finite")
    if not np.any(features != features[0]):
        raise ValueError("Insufficient image variation for two principal components")
    pca_model = PCA(**config.pca.model_dump())
    pca = pca_model.fit_transform(features)
    # Two independent directions are required; reject constant/rank-one inputs.
    tolerance = np.finfo(features.dtype).eps * max(features.shape)
    if pca_model.singular_values_[1] <= tolerance * max(pca_model.singular_values_[0], 1.0):
        raise ValueError("Insufficient image variation for two principal components")
    tsne = TSNE(**config.tsne.model_dump()).fit_transform(features)
    variance = pca_model.explained_variance_ratio_
    if not all(np.isfinite(values).all() for values in (pca, tsne, variance)):
        raise ValueError("Projection produced non-finite coordinates or explained variance")
    return ProjectionCoordinates(
        ids,
        pca,
        tsne,
        (float(variance[0]), float(variance[1])),
        dataset_fingerprint(coco, contents),
    )
