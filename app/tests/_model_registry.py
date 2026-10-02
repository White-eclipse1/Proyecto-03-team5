"""Registro de OPS-06 en una carpeta temporal, para las pruebas de APP-06 y APP-07.

Mismo formato que `reports/models/registry.json` (`classification.registry`) y que
`reports/models/s3_publications.json` (OPS-07), con checkpoints reales de ML-03.
"""

import copy
import hashlib
import json
from pathlib import Path

import torch

from classification.model import ModelConfig, build_model, save_checkpoint
from classification.registry import ModelPackage, ModelRegistry

ROOT = Path(__file__).resolve().parents[2]
MODEL = "dog-cat-resnet18"
RELEASE = "v0.1.1"
IMAGE_SIZE = 64
BUCKET = "mlops-p2-dvc-cache-280764207006"


def make_checkpoint(path: Path, *, bias: tuple[float, float] | None, seed: int = 0) -> Path:
    """ResNet18 sin pesos ImageNet. Con `bias`, la cabeza ignora los pixeles y siempre
    favorece la misma clase; sin `bias`, son pesos aleatorios (depende de la imagen)."""
    torch.manual_seed(seed)
    model = build_model(
        ModelConfig(image_size=IMAGE_SIZE, hidden_layers=[], dropout=0.0, pretrained=False),
        download_weights=False,
    )
    if bias is not None:
        with torch.no_grad():
            model.head[-1].weight.zero_()
            model.head[-1].bias.copy_(torch.tensor(bias))
    return save_checkpoint(model, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Registry:
    """Registro de OPS-06 en una carpeta temporal: `reports/models/registry.json` y paquetes."""

    def __init__(self, root: Path):
        self.root = root
        self.path = root / "reports" / "models" / "registry.json"
        self.publications_path = root / "reports" / "models" / "s3_publications.json"
        self.packages: list[dict] = []
        self.publications: list[dict] = []
        self.template = json.loads(
            (ROOT / "reports" / "models" / "registry.json").read_text(encoding="utf-8")
        )["models"][0]

    def register(self, version: str, run_id: str, checkpoint: Path | None, **changes) -> dict:
        """Una versión con su paquete; sin `checkpoint`, solo metadata (sin paquete)."""
        entry = json.loads(json.dumps(self.template))
        entry.update(
            model_version=version,
            run_id=run_id,
            checkpoint=f"runs:/{run_id}/checkpoints/best.pt",
            dataset_version=RELEASE,
        )
        entry["architecture"].update(
            image_size=IMAGE_SIZE, hidden_layers=[], dropout=0.0, pretrained=False
        )
        entry["preprocessing"]["image_size"] = IMAGE_SIZE
        entry["model_card"].update(mlflow_run_id=run_id, dataset_release=RELEASE)
        if checkpoint is None:
            entry.update(package_path=None, files=None)
        else:
            package = f"data/models/{MODEL}/{version}"
            folder = self.root / package
            (folder / "checkpoint").mkdir(parents=True, exist_ok=True)
            (folder / "checkpoint" / "best.pt").write_bytes(checkpoint.read_bytes())
            (folder / "model-card.md").write_text(f"# {MODEL} {version}\n", encoding="utf-8")
            (folder / "dependencies.json").write_text("{}\n", encoding="utf-8")
            files = {
                name: sha256(folder / name)
                for name in ("checkpoint/best.pt", "model-card.md", "dependencies.json")
            }
            entry.update(
                package_path=package,
                files=files,
                checkpoint_sha256=files["checkpoint/best.pt"],
            )
        for key, value in changes.items():
            entry[key] = value
        ModelPackage.model_validate(entry)
        self.packages.append(entry)
        self.save()
        return entry

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {"schema_version": "1.0", "models": self.packages}
        ModelRegistry.model_validate(document)
        self.path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    def checkpoint(self, version: str) -> Path:
        return self.root / "data" / "models" / MODEL / version / "checkpoint" / "best.pt"

    def publish(self, version: str, **changes) -> dict:
        """Registro de OPS-07 con cada archivo del paquete (y package.json) verificado."""
        entry = next(p for p in self.packages if p["model_version"] == version)
        folder = self.root / entry["package_path"]
        files = {}
        for name in (*entry["files"], "package.json"):
            path = folder / name
            if not path.exists():
                path.write_text("{}\n", encoding="utf-8")
            digest = sha256(path)
            files[name] = {
                "key": f"models/{MODEL}/{version}/{name}",
                "version_id": None,
                "size": path.stat().st_size,
                "sha256": digest,
                "s3_checksum_sha256": digest,
                "download_sha256": digest,
                "etag": "0" * 32,
            }
        record = {
            "model_version": version,
            "model_name": entry["model_name"],
            "run_id": entry["run_id"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "bucket": BUCKET,
            "region": "us-east-1",
            "published_at": "2026-10-02T20:15:49Z",
            "files": files,
        }
        record.update(copy.deepcopy(changes))
        self.publications.append(record)
        self.save_publications()
        return record

    def save_publications(self) -> None:
        self.publications_path.parent.mkdir(parents=True, exist_ok=True)
        document = {"schema_version": "1.0", "publications": self.publications}
        self.publications_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
