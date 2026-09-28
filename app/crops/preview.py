"""ML-01 — hoja de verificación manual: imagen original con su bbox junto al crop.

Uso desde `app/`, después de `dvc repro crops`:

    uv run python -m crops.preview --limit 24 --out ../reports/crops-preview.png

Cada fila muestra la imagen fuente con la bbox COCO dibujada en rojo y, a su
derecha, el crop guardado. Sirve para confirmar a ojo que el crop
corresponde a la bbox; no forma parte del pipeline DVC.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from crops.models import CropReport

ROW_HEIGHT = 160
GAP = 8


def _fit(image: Image.Image, height: int) -> Image.Image:
    scale = height / image.height
    return image.resize((max(1, round(image.width * scale)), height))


def render_verification_sheet(
    report: CropReport,
    *,
    images_dir: Path,
    crops_dir: Path,
    out_path: Path,
    limit: int = 24,
) -> Path:
    rows = []
    for crop in report.crops[:limit]:
        with Image.open(images_dir / crop.source_file_name) as source:
            original = source.convert("RGB")
        x, y, width, height = crop.bbox
        ImageDraw.Draw(original).rectangle(
            [x, y, x + width, y + height], outline=(255, 0, 0), width=3
        )
        with Image.open(crops_dir / crop.crop_path) as stored:
            cropped = stored.convert("RGB")
        left, right = _fit(original, ROW_HEIGHT), _fit(cropped, ROW_HEIGHT)
        row = Image.new("RGB", (left.width + GAP + right.width, ROW_HEIGHT + 20), "white")
        row.paste(left, (0, 20))
        row.paste(right, (left.width + GAP, 20))
        ImageDraw.Draw(row).text((2, 2), f"{crop.crop_id} · {crop.class_name}", fill="black")
        rows.append(row)

    if not rows:
        raise ValueError("El reporte no tiene crops aceptados para verificar")
    sheet = Image.new(
        "RGB",
        (max(row.width for row in rows), sum(row.height + GAP for row in rows)),
        "white",
    )
    offset = 0
    for row in rows:
        sheet.paste(row, (0, offset))
        offset += row.height + GAP
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, default=repo_root / "reports" / "crops.json")
    parser.add_argument("--images", type=Path, default=repo_root / "data" / "raw" / "images")
    parser.add_argument("--crops", type=Path, default=repo_root / "data" / "crops")
    parser.add_argument("--out", type=Path, default=repo_root / "reports" / "crops-preview.png")
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()

    report = CropReport.model_validate_json(args.report.read_text(encoding="utf-8"))
    path = render_verification_sheet(
        report, images_dir=args.images, crops_dir=args.crops, out_path=args.out, limit=args.limit
    )
    print(f"Hoja de verificación: {path}")


if __name__ == "__main__":
    main()
