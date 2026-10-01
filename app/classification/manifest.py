"""ML-02 — lectura del manifiesto P3 70/20/10 que genera OPS-02.

Usa el contrato de OPS-02 (`manifests.models.P3Manifest`): cada registro es un
crop de ML-01 con `crop_id`, `source_image_id`, `duplicate_group`, `class` y
`split`. Al cargar el archivo se exige que `manifest_hash` coincida con su
contenido (misma fórmula con la que OPS-02 lo generó) y que no haya fuga de
`crop_id`, `source_image_id` ni `duplicate_group` entre particiones. Que cada
crop exista y coincida con `reports/crops.json` lo comprueba
`classification.dataset`.
"""

from pathlib import Path

# Fórmula oficial del hash del manifiesto; recalcularla aquí la duplicaría.
from manifests.generate import _manifest_hash_payload
from manifests.models import P3Manifest
from manifests.validation import validate_no_leakage


def manifest_hash(manifest: P3Manifest) -> str:
    return _manifest_hash_payload(
        dataset_version=manifest.dataset_version,
        source_release=manifest.source_release,
        manifest_version=manifest.manifest_version,
        seed=manifest.seed,
        records=manifest.records,
        provenance=manifest.provenance,
    )


def load_manifest(path: Path) -> P3Manifest:
    manifest = P3Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    actual = manifest_hash(manifest)
    if actual != manifest.manifest_hash:
        raise ValueError(
            f"manifest_hash no coincide con el contenido de {path.name}: declarado "
            f"{manifest.manifest_hash}, calculado {actual}"
        )
    validate_no_leakage(manifest.records)
    return manifest
