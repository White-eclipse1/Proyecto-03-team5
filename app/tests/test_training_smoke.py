"""OPS-05: contrato del smoke training corto."""

from __future__ import annotations

from pathlib import Path


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
    assert "run_id" in text
    assert 'state == "succeeded"' in text
    assert 'state == "failed"' in text
