"""ML-02 — Dataset/DataLoader de clasificación dog/cat para train, validation y test.

El Dataset no recorre ninguna carpeta: cruza los registros de una partición del
manifiesto P3 (OPS-02, `reports/releases/<version>/manifest.json`) con
`reports/crops.json` (ML-01) por `crop_id` y abre solo `crops_dir/<crop_path>`
de esos crops. Rechaza al construirse cualquier desacuerdo entre ambos
(release, fuga entre particiones, crop repetido o inexistente, clase o imagen
de origen distinta), cualquier PNG faltante y cualquier PNG cuyo sha256 no sea
el registrado para ese crop en `crops.json` (alterado o sustituido), antes de
que empiece un entrenamiento. Cada lectura vuelve a verificar el hash de los
bytes que decodifica, así un archivo cambiado después tampoco entra.
`load_split` además exige que `crops.json` sea exactamente el archivo del que
salió el manifiesto (`provenance.crops_sha256`).

Cada muestra es un dict con `image` (tensor `3 x image_size x image_size`),
`label` (índice de `crops.classes.CLASS_TO_INDEX`: dog=0, cat=1), `crop_id`,
`source_image_id` y `class_name`, así el DataLoader conserva la trazabilidad de
cada predicción hasta la bbox COCO original.
"""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import torch
from manifests.models import ManifestRecord, P3Manifest, SplitName
from manifests.validation import validate_no_leakage
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from classification.manifest import load_manifest
from classification.transforms import transform_for
from crops.classes import CLASS_TO_INDEX
from crops.models import CropRecord, CropReport
from presentation.ml_contracts import TrainingParams


@dataclass(frozen=True)
class ClassificationSample:
    crop_id: str
    source_image_id: int
    class_name: str
    label: int
    split: SplitName
    path: Path
    sha256: str


def _verified_bytes(crop_id: str, path: Path, expected: str) -> bytes:
    """Bytes del PNG, solo si su sha256 es el que `crops.json` registró al extraerlo."""
    data = path.read_bytes()
    actual = sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(
            f"{crop_id}: el sha256 de {path.name} ({actual}) no coincide con el de "
            f"crops.json ({expected}); el crop fue alterado o sustituido"
        )
    return data


def match_manifest_to_crops(
    manifest: P3Manifest, crop_report: CropReport, split: SplitName
) -> list[tuple[ManifestRecord, CropRecord]]:
    """Registros de `split` con su crop de ML-01, validados contra `crops.json`."""
    if manifest.dataset_version != crop_report.dataset_version:
        raise ValueError(
            f"El manifiesto es del release {manifest.dataset_version} y los crops de "
            f"{crop_report.dataset_version}"
        )
    crop_ids = [record.crop_id for record in manifest.records]
    if len(crop_ids) != len(set(crop_ids)):
        raise ValueError("crop_id repetido en el manifiesto")
    validate_no_leakage(manifest.records)

    crops = {crop.crop_id: crop for crop in crop_report.crops}
    matched = []
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
        matched.append((record, crop))
    return matched


class CropClassificationDataset(Dataset):
    def __init__(
        self,
        manifest: P3Manifest,
        crop_report: CropReport,
        *,
        crops_dir: Path,
        split: SplitName,
        image_size: int,
    ) -> None:
        samples = []
        for record, crop in match_manifest_to_crops(manifest, crop_report, split):
            path = crops_dir / crop.crop_path
            if not path.is_file():
                raise FileNotFoundError(f"Falta el crop {record.crop_id}: {path}")
            _verified_bytes(record.crop_id, path, crop.sha256)
            samples.append(
                ClassificationSample(
                    crop_id=record.crop_id,
                    source_image_id=record.source_image_id,
                    class_name=record.class_name,
                    label=CLASS_TO_INDEX[record.class_name],
                    split=split,
                    path=path,
                    sha256=crop.sha256,
                )
            )

        self.split = split
        self.dataset_version = manifest.dataset_version
        self.manifest_hash = manifest.manifest_hash
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
        data = _verified_bytes(sample.crop_id, sample.path, sample.sha256)
        with Image.open(BytesIO(data)) as stored:
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
    split: SplitName,
    params: TrainingParams,
) -> CropClassificationDataset:
    """Dataset de una partición con el `image_size` de la configuración de entrenamiento."""
    manifest = load_manifest(manifest_path)
    report_bytes = crop_report_path.read_bytes()
    digest = "sha256:" + sha256(report_bytes).hexdigest()
    expected = manifest.provenance.crops_sha256 if manifest.provenance else None
    if digest != expected:
        raise ValueError(
            f"crops_sha256 del manifiesto ({expected}) no coincide con {crop_report_path.name} "
            f"({digest}): el manifiesto no salió de estos crops"
        )
    return CropClassificationDataset(
        manifest,
        CropReport.model_validate_json(report_bytes),
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
