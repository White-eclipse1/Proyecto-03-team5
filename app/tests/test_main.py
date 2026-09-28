import logging

import pytest

from presentation import main as main_module
from presentation.contracts import QualityCheck, QualityReport


class _StopLoop(Exception):
    """Marca que main() llegó al `while True: sleep(...)` final."""


def _fake_report(status):
    return QualityReport(
        schema_version="1.0",
        dataset_version="test",
        status=status,
        checks=[
            QualityCheck(
                check_name="min_images_per_class",
                passed=status == "passed",
                metric_value=1.0,
                action="fail",
            )
        ],
    )


def _patch_common(monkeypatch):
    monkeypatch.setattr(main_module, "_wait_for_dependencies", lambda: None)
    monkeypatch.setattr(
        main_module.time, "sleep", lambda _seconds: (_ for _ in ()).throw(_StopLoop())
    )


@pytest.mark.parametrize("status", ["passed", "warning", "failed"])
def test_main_logs_status_and_keeps_running(monkeypatch, caplog, status):
    """El contenedor sigue vivo pase lo que pase con el status, pero el log
    deja rastro inequívoco de un status=failed (antes quedaba en silencio)."""
    _patch_common(monkeypatch)
    monkeypatch.setattr(main_module.gate, "run", lambda: _fake_report(status))

    with caplog.at_level(logging.INFO), pytest.raises(_StopLoop):
        main_module.main()

    if status == "failed":
        assert any(
            record.levelno == logging.ERROR and "BLOQUEADA" in record.message
            for record in caplog.records
        )
    elif status == "warning":
        assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_main_does_not_swallow_gate_exceptions(monkeypatch):
    """Antes, una excepción real al correr el gate se tragaba y el contenedor
    seguía 'vivo' sin haber corrido nunca la compuerta. Ahora debe propagar
    para que `restart: on-failure` de docker-compose lo haga visible."""
    _patch_common(monkeypatch)

    def _broken_run():
        raise RuntimeError("dataset_dir no existe")

    monkeypatch.setattr(main_module.gate, "run", _broken_run)

    with pytest.raises(RuntimeError, match="dataset_dir no existe"):
        main_module.main()
