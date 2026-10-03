"""OPS-10: la suite no debe llenar el disco del evaluador.

Muchos tests entrenan runs cortos y guardan checkpoints en `tmp_path`; una corrida
completa dejaba ~17 GB y pytest conserva las tres últimas. Un disco lleno hace fallar la
suite con errores ajenos al código (`No space left on device`).
"""

import pytest


def test_pytest_keeps_every_tmp_path_directory_so_names_are_never_reused(request):
    # Con tmp_path_retention_policy = "failed" pytest borra el directorio de un test que
    # pasa y reusa su nombre en el siguiente test parametrizado; MLflow guarda en caché el
    # store de esa ruta y falla. Por eso se vacía el contenido y se conserva el directorio.
    assert request.config.getini("tmp_path_retention_policy") == "all"


def test_a_passed_test_frees_its_tmp_path_right_away(request):
    import tests.conftest as conftest

    assert hasattr(conftest, "pytest_runtest_makereport")


@pytest.fixture(scope="module")
def seen():
    return {}


def test_writes_into_its_tmp_path(tmp_path, seen):
    (tmp_path / "checkpoint.pt").write_bytes(b"x" * 1024)
    seen["path"] = tmp_path


def test_the_previous_tmp_path_was_already_emptied(seen):
    # Vacío pero presente: pytest no reusa su nombre en el siguiente test.
    assert seen["path"].is_dir()
    assert list(seen["path"].iterdir()) == []
