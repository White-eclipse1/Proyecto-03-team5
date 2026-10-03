"""Configuración compartida de pytest.

OPS-10: muchos tests entrenan runs cortos y guardan MLflow y checkpoints en `tmp_path`.
Una corrida completa dejaba ~17 GB de temporales, y pytest conserva las tres últimas: el
disco se llenaba y la suite fallaba con `No space left on device`. El `tmp_path` de un
test que pasa se vacía en cuanto termina; el de uno que falla se conserva para depurar.

Se vacía (no se borra el directorio): si desapareciera, pytest reusaría su nombre para el
siguiente test parametrizado y MLflow, que guarda en caché el store de esa ruta, fallaría.
"""

import shutil

import pytest


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.when == "call":
        item._ops10_passed = report.passed
    elif report.when == "teardown" and getattr(item, "_ops10_passed", False) and report.passed:
        path = item.funcargs.get("tmp_path") if hasattr(item, "funcargs") else None
        if path is not None and path.is_dir():
            for child in path.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
    return report
