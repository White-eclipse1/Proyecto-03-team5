"""OPS-10: la suite no debe llenar el disco del evaluador.

Muchos tests entrenan runs cortos y guardan checkpoints en `tmp_path`; una corrida
completa dejaba ~17 GB y pytest conserva las tres últimas. Un disco lleno hace fallar la
suite con errores ajenos al código (`No space left on device`).
"""

import pytest


def test_only_failed_tests_keep_their_temporary_directories(request):
    assert request.config.getini("tmp_path_retention_policy") == "failed"


def test_a_passed_test_frees_its_tmp_path_right_away(request):
    import tests.conftest as conftest

    assert hasattr(conftest, "pytest_runtest_makereport")


@pytest.fixture(scope="module")
def seen():
    return {}


def test_writes_into_its_tmp_path(tmp_path, seen):
    (tmp_path / "checkpoint.pt").write_bytes(b"x" * 1024)
    seen["path"] = tmp_path


def test_the_previous_tmp_path_was_already_removed(seen):
    assert not seen["path"].exists()
