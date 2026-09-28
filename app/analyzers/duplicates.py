"""P2-20: pHash sobre contenido de imágenes en memoria, sin acceso a almacenamiento."""

from collections.abc import Mapping
from io import BytesIO
from itertools import combinations

import imagehash
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from analyzers.base import AnalyzerResult


class DuplicateConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)

    threshold: float = Field(ge=0, le=1)


def analyze_duplicates(images: Mapping[int, bytes], config: DuplicateConfig) -> AnalyzerResult:
    """Recibe image_id -> bytes codificados; los nombres/rutas no intervienen.

    Usa pHash de ImageHash con hash_size=8 (64 bits). passed solo indica
    ausencia de pares similares, no aprueba un release ni evalúa splits.
    Bytes ilegibles se rechazan con ValueError, sin omitir imágenes en silencio.
    """
    if any(type(image_id) is not int or image_id < 0 for image_id in images):
        raise ValueError("Image IDs must be non-negative integers")
    hashes = {}
    for image_id in sorted(images):
        content = images[image_id]
        if not isinstance(content, bytes):
            raise ValueError(f"Image {image_id} must contain encoded bytes")
        try:
            with Image.open(BytesIO(content)) as image:
                hashes[image_id] = imagehash.phash(image, hash_size=8)
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot decode image {image_id}") from error

    pairs = []
    hash_bits = 64
    for image_id_a, image_id_b in combinations(hashes, 2):
        distance = int(hashes[image_id_a] - hashes[image_id_b])
        similarity = 1 - distance / hash_bits
        if similarity >= config.threshold:
            pairs.append(
                {
                    "image_id_a": image_id_a,
                    "image_id_b": image_id_b,
                    "hamming_distance": distance,
                    "similarity": similarity,
                }
            )

    return AnalyzerResult(
        check_name="duplicate_images",
        passed=not pairs,
        metric_value=float(len(pairs)),
        details={
            "image_pairs": pairs,
            "hash_bits": hash_bits,
            "similarity_threshold": config.threshold,
            "total_images": len(images),
        },
    )
