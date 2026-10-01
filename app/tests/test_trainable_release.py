"""APP-03: el release aprobado de P2 se puede entrenar desde el portal.

La API (`training/releases.py`) y la pantalla Training exigen
`reports/releases/<v>/provenance.json` y `manifest.json`. Para el release que
OPS-01 aprobó (`releases/p2_releases.json`), la procedencia commiteada debe
tener exactamente los hashes DVC verificados, y la regla de bloqueo no debe
impedir entrenarlo.
"""

import json
from pathlib import Path

import pytest

from presentation.ml_contracts import training_blocked_reason
from training.releases import load_release, published_releases

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"
P2_CATALOG = json.loads((ROOT / "app" / "releases" / "p2_releases.json").read_text("utf-8"))
APPROVED = [release["release_version"] for release in P2_CATALOG["releases"]]


def approved(version: str) -> dict:
    return next(r for r in P2_CATALOG["releases"] if r["release_version"] == version)


@pytest.mark.parametrize("version", APPROVED)
def test_approved_release_is_published_and_trainable(version):
    assert version in published_releases(REPORTS_DIR)
    release = load_release(REPORTS_DIR, version)

    assert release.quality_status == approved(version)["quality_status"]
    assert (
        training_blocked_reason(release.quality_status, release.provenance, release.manifest)
        is None
    )


@pytest.mark.parametrize("version", APPROVED)
def test_committed_provenance_has_the_approved_dvc_hashes(version):
    provenance = load_release(REPORTS_DIR, version).provenance
    assert provenance is not None, f"falta reports/releases/{version}/provenance.json"
    hashes = {output.path: output.md5 for output in provenance.dvc_outputs}

    assert provenance.dataset_version == version
    assert hashes == {
        "images": approved(version)["images_dvc_hash"],
        "annotations": approved(version)["annotations_dvc_hash"],
    }
