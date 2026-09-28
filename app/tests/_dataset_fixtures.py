"""Fixtures compartidos entre tests: imágenes JPEG sintéticas + dataset COCO en disco.

No empieza con `test_`, así pytest no lo recolecta como módulo de pruebas.
Usado por `test_gate.py`, `test_release.py` y cualquier otro test que
necesite un dataset cat/dog real en disco sin repetir esta construcción.
"""

import json
from io import BytesIO

from PIL import Image, ImageDraw


def jpeg_bytes(seed: int) -> bytes:
    """pHash mira estructura, no color plano: un cuadro sólido no basta para
    que dos imágenes se vean distintas. Se dibuja un rectángulo en una
    posición distinta por `seed` para dar estructura real."""
    image = Image.new("RGB", (64, 64), color=(20, 20, 20))
    draw = ImageDraw.Draw(image)
    x0 = (seed * 11) % 40
    y0 = (seed * 17) % 40
    draw.rectangle([x0, y0, x0 + 20, y0 + 20], fill=(220, 220, 220))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def write_coco_dataset(tmp_path, *, cats: int = 2, dogs: int = 2):
    """Escribe un `annotations/lote.json` + imágenes reales bajo `tmp_path`."""
    annotations_dir = tmp_path / "annotations"
    images_dir = tmp_path / "images"
    annotations_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)

    images, annotations = [], []
    image_id = 1
    ann_id = 1
    seed = 0
    for index in range(cats):
        file_name = f"cat.{index}.jpg"
        (images_dir / file_name).write_bytes(jpeg_bytes(seed))
        seed += 1
        images.append({"id": image_id, "file_name": file_name, "width": 64, "height": 64})
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 4,
                "bbox": [0, 0, 40, 40],
                "area": 1600.0,
                "iscrowd": 0,
            }
        )
        image_id += 1
        ann_id += 1
    for index in range(dogs):
        file_name = f"dog.{index}.jpg"
        (images_dir / file_name).write_bytes(jpeg_bytes(seed))
        seed += 1
        images.append({"id": image_id, "file_name": file_name, "width": 64, "height": 64})
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 3,
                "bbox": [0, 0, 40, 40],
                "area": 1600.0,
                "iscrowd": 0,
            }
        )
        image_id += 1
        ann_id += 1

    doc = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 3, "name": "dog"}, {"id": 4, "name": "cat"}],
    }
    (annotations_dir / "lote.json").write_text(json.dumps(doc), encoding="utf-8")
    return tmp_path
