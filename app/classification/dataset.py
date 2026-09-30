"""ML-02 — Dataset/DataLoader de clasificación dog/cat para train, validation y test.

El Dataset no recorre ninguna carpeta: cruza los registros de una partición del
manifiesto P3 (OPS-02) con `reports/crops.json` (ML-01) por `crop_id` y abre
solo `crops_dir/<crop_path>` de esos crops. Rechaza al construirse cualquier
desacuerdo entre ambos (release, crop inexistente, clase o imagen de origen
distinta) y cualquier PNG faltante, antes de que empiece un entrenamiento.

Cada muestra es un dict con `image` (tensor `3 x image_size x image_size`),
`label` (índice de `crops.classes.CLASS_TO_INDEX`: dog=0, cat=1), `crop_id`,
`source_image_id` y `class_name`, así el DataLoader conserva la trazabilidad de
cada predicción hasta la bbox COCO original.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from classification.manifest import CropManifest, Split, load_manifest
from classification.transforms import transform_for
from crops.classes import CLASS_TO_INDEX
from crops.models import CropReport
from presentation.ml_contracts import TrainingParams


@dataclass(frozen=True)
class ClassificationSample:
    crop_id: str
    source_image_id: int
    class_name: str
    label: int
    split: Split
    path: Path


class CropClassificationDataset(Dataset):
    def __init__(
        self,
        manifest: CropManifest,
        crop_report: CropReport,
        *,
        crops_dir: Path,
        split: Split,
        image_size: int,
    ) -> None:
        if manifest.dataset_version != crop_report.dataset_version:
            raise ValueError(
                f"El manifiesto es del release {manifest.dataset_version} y los crops de "
                f"{crop_report.dataset_version}"
            )
        crops = {crop.crop_id: crop for crop in crop_report.crops}
        samples = []
        for record in manifest.records:
            if record.split != split:
                continue
            crop = crops.get(record.crop_id)
            if crop is None:
                raise ValueError(f"{record.crop_id} no está entre los crops aceptados de ML-01")
            if record.class_name != crop.class_name:
                raise ValueError(
                    f"{record.crop_id}: class={record.class_name} en el manifiesto y "
                    f"{crop.class_name} en crops.json"
                )
            if record.source_image_id != crop.image_id:
                raise ValueError(
                    f"{record.crop_id}: source_image_id={record.source_image_id} en el "
                    f"manifiesto e image_id={crop.image_id} en crops.json"
                )
            path = crops_dir / crop.crop_path
            if not path.is_file():
                raise FileNotFoundError(f"Falta el crop {record.crop_id}: {path}")
            samples.append(
                ClassificationSample(
                    crop_id=record.crop_id,
                    source_image_id=record.source_image_id,
                    class_name=record.class_name,
                    label=CLASS_TO_INDEX[record.class_name],
                    split=split,
                    path=path,
                )
            )

        self.split = split
        self.dataset_version = manifest.dataset_version
        self.class_to_index = dict(CLASS_TO_INDEX)
        self.transform = transform_for(split, image_size)
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def sample(self, index: int) -> ClassificationSample:
        """Metadatos de la muestra sin abrir la imagen."""
        return self._samples[index]

    def __getitem__(self, index: int) -> dict:
        sample = self._samples[index]
        with Image.open(sample.path) as stored:
            image = self.transform(stored.convert("RGB"))
        return {
            "image": image,
            "label": sample.label,
            "crop_id": sample.crop_id,
            "source_image_id": sample.source_image_id,
            "class_name": sample.class_name,
        }


def load_split(
    manifest_path: Path,
    crop_report_path: Path,
    *,
    crops_dir: Path,
    split: Split,
    params: TrainingParams,
) -> CropClassificationDataset:
    """Dataset de una partición con el `image_size` de la configuración de entrenamiento."""
    return CropClassificationDataset(
        load_manifest(manifest_path),
        CropReport.model_validate_json(crop_report_path.read_text(encoding="utf-8")),
        crops_dir=crops_dir,
        split=split,
        image_size=params.image_size,
    )


def build_dataloader(
    dataset: CropClassificationDataset,
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    """Baraja solo train, con un generador sembrado: misma semilla, mismo orden."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataset.split == "train",
        generator=generator,
        num_workers=num_workers,
    )
