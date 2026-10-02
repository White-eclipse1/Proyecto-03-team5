"""OPS-05: contrato del smoke training corto."""

from __future__ import annotations

from pathlib import Path

import pytest

import training.smoke as smoke


def test_smoke_script_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "training" / "smoke.py").is_file()


def test_smoke_uses_short_cpu_compatible_training():
    root = Path(__file__).resolve().parents[1]
    text = (root / "training" / "smoke.py").read_text(encoding="utf-8")

    assert '"max_epochs": 1' in text
    assert '"image_size": 32' in text
    assert '"batch_size": 128' in text
    assert "cuda" not in text.lower()


def test_smoke_verifies_mlflow_checkpoint_contract():
    root = Path(__file__).resolve().parents[1]
    text = (root / "training" / "smoke.py").read_text(encoding="utf-8")

    assert 'f"runs:/{run_id}/checkpoints/best.pt"' in text
    assert "client.get_run(run_id)" in text
    assert "client.download_artifacts(" in text
    assert '"checkpoints/best.pt"' in text


def test_smoke_rejects_missing_crops_before_enqueue(monkeypatch, tmp_path):
    crops_dir = tmp_path / "data" / "crops"
    crops_report = tmp_path / "reports" / "crops.json"

    monkeypatch.setattr(smoke, "CROPS_DIR", crops_dir)
    monkeypatch.setattr(smoke, "CROPS_REPORT", crops_report)

    manifest = {
        "records": [
            {
                "crop_id": "img1-ann1",
                "class": "dog",
            }
        ]
    }

    with pytest.raises(SystemExit, match=r"Falta reports/crops\.json"):
        smoke._require_crops(manifest)

    crops_report.parent.mkdir(parents=True)
    crops_report.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="Faltan crops requeridos"):
        smoke._require_crops(manifest)


def test_smoke_requires_real_finished_mlflow_run(monkeypatch, tmp_path):
    class FakeInfo:
        status = "FAILED"

    class FakeRun:
        info = FakeInfo()

    class FakeClient:
        def get_run(self, run_id):
            return FakeRun()

    monkeypatch.setattr(smoke, "tracking_client", lambda: FakeClient())

    with pytest.raises(SystemExit, match="su estado es"):
        smoke._verify_mlflow_artifact("fake-run")


def test_smoke_requires_real_nonempty_checkpoint(monkeypatch, tmp_path):
    class FakeInfo:
        status = "FINISHED"

    class FakeRun:
        info = FakeInfo()

    class FakeClient:
        def get_run(self, run_id):
            return FakeRun()

        def download_artifacts(self, run_id, artifact_path, dst_path):
            checkpoint = Path(dst_path) / "best.pt"
            checkpoint.write_bytes(b"")
            return str(checkpoint)

    monkeypatch.setattr(smoke, "tracking_client", lambda: FakeClient())

    with pytest.raises(SystemExit, match=r"best\.pt valido"):
        smoke._verify_mlflow_artifact("fake-run")
