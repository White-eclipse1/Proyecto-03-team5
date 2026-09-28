import asyncio
import json
import re
from pathlib import Path

import pytest

from presentation.mcp_server import build_server
from tests._mcp_fixtures import mcp_settings, write_annotations, write_releases

EXPECTED_TOOLS = {
    "get_dataset_summary",
    "get_quality_report",
    "get_check_result",
    "get_splits_report",
    "get_versions_report",
}


def call(server, name, arguments=None):
    return asyncio.run(server.call_tool(name, arguments or {}))


def result_json(call_result):
    return json.loads(call_result.content[0].text)


def test_registers_exactly_the_five_expected_tools(tmp_path, monkeypatch):
    settings = mcp_settings(monkeypatch, tmp_path / "dataset", tmp_path / "reports")
    server = build_server(settings)

    tools = asyncio.run(server.list_tools())

    assert {tool.name for tool in tools} == EXPECTED_TOOLS
    for tool in tools:
        assert tool.description, f"{tool.name} no tiene descripción"


def test_every_tool_is_marked_read_only(tmp_path, monkeypatch):
    settings = mcp_settings(monkeypatch, tmp_path / "dataset", tmp_path / "reports")
    server = build_server(settings)

    tools = asyncio.run(server.list_tools())

    for tool in tools:
        assert tool.annotations is not None, f"{tool.name} no tiene ToolAnnotations"
        assert tool.annotations.read_only_hint is True, (
            f"{tool.name} no está marcada read_only_hint"
        )
        assert tool.annotations.destructive_hint is False


def test_no_write_operations_in_source():
    """P2-34 criterio explícito: sin INSERT/UPDATE/DELETE/put_object en el handler."""
    source = Path(__file__).resolve().parent.parent / "presentation" / "mcp_server.py"
    text = source.read_text(encoding="utf-8")
    assert not re.search(r"\bINSERT\b|\bUPDATE\b|\bDELETE\b|put_object", text, re.IGNORECASE)
    # Tampoco importa nada de storage/ (el único tier autorizado a tocar red/DB).
    assert "storage.db" not in text
    assert "storage.object_store" not in text


def test_get_dataset_summary_counts_real_data(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    write_annotations(dataset_dir / "annotations")
    settings = mcp_settings(monkeypatch, dataset_dir, tmp_path / "reports")
    server = build_server(settings)

    result = result_json(call(server, "get_dataset_summary"))

    assert result["total_images"] == 3
    assert result["total_annotations"] == 3
    assert result["images_per_category"] == {"dog": 2, "cat": 1}


def test_get_quality_report_available_when_file_exists(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    quality = {"schema_version": "1.0", "dataset_version": "x", "status": "passed", "checks": []}
    (reports_dir / "quality.json").write_text(json.dumps(quality), encoding="utf-8")
    settings = mcp_settings(monkeypatch, tmp_path / "dataset", reports_dir)
    server = build_server(settings)

    result = result_json(call(server, "get_quality_report"))

    assert result["available"] is True
    assert result["data"]["status"] == "passed"


@pytest.mark.parametrize(
    "tool_name", ["get_quality_report", "get_splits_report", "get_versions_report"]
)
def test_report_tools_report_unavailable_instead_of_crashing(tmp_path, monkeypatch, tool_name):
    settings = mcp_settings(monkeypatch, tmp_path / "dataset", tmp_path / "reports")
    server = build_server(settings)

    result = result_json(call(server, tool_name))

    assert result["available"] is False
    assert "reason" in result


def test_get_check_result_finds_an_existing_check(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    quality = {
        "schema_version": "1.0",
        "dataset_version": "x",
        "status": "warning",
        "checks": [
            {
                "check_name": "max_imbalance_ratio",
                "passed": False,
                "metric_value": 2.0,
                "details": {},
                "action": "warn",
            }
        ],
    }
    (reports_dir / "quality.json").write_text(json.dumps(quality), encoding="utf-8")
    settings = mcp_settings(monkeypatch, tmp_path / "dataset", reports_dir)
    server = build_server(settings)

    result = result_json(call(server, "get_check_result", {"check_name": "max_imbalance_ratio"}))

    assert result["available"] is True
    assert result["data"]["metric_value"] == 2.0


def test_get_check_result_reports_unavailable_for_unknown_name(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    quality = {"schema_version": "1.0", "dataset_version": "x", "status": "passed", "checks": []}
    (reports_dir / "quality.json").write_text(json.dumps(quality), encoding="utf-8")
    settings = mcp_settings(monkeypatch, tmp_path / "dataset", reports_dir)
    server = build_server(settings)

    result = result_json(call(server, "get_check_result", {"check_name": "no_existe"}))

    assert result["available"] is False


def test_get_splits_report_defaults_to_the_latest_release(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    write_releases(reports_dir, {"v0.1.0": "failed", "v0.2.0": "warning"})
    server = build_server(mcp_settings(monkeypatch, tmp_path / "dataset", reports_dir))

    result = result_json(call(server, "get_splits_report"))

    assert result["available"] is True
    assert result["data"]["dataset_version"] == "v0.2.0"


def test_get_splits_report_can_target_a_specific_release(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    write_releases(reports_dir, {"v0.1.0": "failed", "v0.2.0": "warning"})
    server = build_server(mcp_settings(monkeypatch, tmp_path / "dataset", reports_dir))

    result = result_json(call(server, "get_splits_report", {"dataset_version": "v0.1.0"}))

    assert result["available"] is True
    assert result["data"]["dataset_version"] == "v0.1.0"


def test_get_quality_report_with_version_returns_the_frozen_release(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    write_releases(reports_dir, {"v0.1.0": "failed"})
    working_copy = {
        "schema_version": "1.0",
        "dataset_version": "local-dev",
        "status": "passed",
        "checks": [],
    }
    (reports_dir / "quality.json").write_text(json.dumps(working_copy), encoding="utf-8")
    server = build_server(mcp_settings(monkeypatch, tmp_path / "dataset", reports_dir))

    current = result_json(call(server, "get_quality_report"))
    frozen = result_json(call(server, "get_quality_report", {"dataset_version": "v0.1.0"}))

    assert current["data"]["dataset_version"] == "local-dev"
    assert frozen["data"]["dataset_version"] == "v0.1.0"
    assert frozen["data"]["status"] == "failed"


@pytest.mark.parametrize("tool_name", ["get_quality_report", "get_splits_report"])
def test_release_lookup_reports_unavailable_for_an_unknown_version(
    tmp_path, monkeypatch, tool_name
):
    reports_dir = tmp_path / "reports"
    write_releases(reports_dir, {"v0.1.0": "failed"})
    server = build_server(mcp_settings(monkeypatch, tmp_path / "dataset", reports_dir))

    result = result_json(call(server, tool_name, {"dataset_version": "v9.9.9"}))

    assert result["available"] is False
    assert "v9.9.9" in result["reason"]


def test_release_lookup_reports_unavailable_when_the_catalog_has_no_releases(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "versions.json").write_text(
        json.dumps({"schema_version": "1.0", "releases": []}), encoding="utf-8"
    )
    server = build_server(mcp_settings(monkeypatch, tmp_path / "dataset", reports_dir))

    result = result_json(call(server, "get_splits_report"))

    assert result["available"] is False
    assert "release" in result["reason"]


def test_get_dataset_summary_reports_which_dataset_version_it_describes(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    write_annotations(dataset_dir / "annotations")
    monkeypatch.setenv("DATASET_VERSION", "local-test")
    server = build_server(mcp_settings(monkeypatch, dataset_dir, tmp_path / "reports"))

    result = result_json(call(server, "get_dataset_summary"))

    assert result["dataset_version"] == "local-test"


def test_get_check_result_reports_the_dataset_version_of_its_quality_report(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True)
    check = {
        "check_name": "spatial_bias",
        "passed": True,
        "metric_value": 0.1,
        "details": {},
        "action": "warn",
    }
    quality = {"schema_version": "1.0", "dataset_version": "v0.3.0", "status": "passed"}
    (reports_dir / "quality.json").write_text(
        json.dumps({**quality, "checks": [check]}), encoding="utf-8"
    )
    server = build_server(mcp_settings(monkeypatch, tmp_path / "dataset", reports_dir))

    result = result_json(call(server, "get_check_result", {"check_name": "spatial_bias"}))

    assert result["dataset_version"] == "v0.3.0"
    assert result["data"]["metric_value"] == 0.1


def test_latest_release_is_the_highest_semantic_version_not_the_last_in_the_catalog(
    tmp_path, monkeypatch
):
    reports_dir = tmp_path / "reports"
    # Catálogo desordenado a propósito: el contrato no le da orden. Como texto,
    # "v0.2.0" > "v0.10.0"; como versión semántica es al revés.
    write_releases(reports_dir, {"v0.2.0": "failed", "v0.10.0": "warning", "v0.1.0": "failed"})
    server = build_server(mcp_settings(monkeypatch, tmp_path / "dataset", reports_dir))

    result = result_json(call(server, "get_splits_report"))

    assert result["data"]["dataset_version"] == "v0.10.0"


def test_without_semantic_versions_the_latest_is_not_guessed(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    write_releases(reports_dir, {"demo-v1": "failed", "demo-v2": "failed"})
    server = build_server(mcp_settings(monkeypatch, tmp_path / "dataset", reports_dir))

    implicit = result_json(call(server, "get_splits_report"))
    explicit = result_json(call(server, "get_splits_report", {"dataset_version": "demo-v2"}))

    assert implicit["available"] is False
    assert "demo-v1" in implicit["reason"]
    assert "demo-v2" in implicit["reason"]
    assert explicit["data"]["dataset_version"] == "demo-v2"


def test_get_dataset_summary_says_why_when_the_dataset_is_not_available_locally(
    tmp_path, monkeypatch
):
    """Sin `data/raw/annotations` (DVC sin `pull`) no es un error opaco: es "no disponible"."""
    server = build_server(mcp_settings(monkeypatch, tmp_path / "dataset", tmp_path / "reports"))

    result = call(server, "get_dataset_summary")

    assert result.is_error is False
    assert result_json(result)["available"] is False
    assert "dvc pull" in result_json(result)["reason"]


def test_get_dataset_summary_marks_a_real_summary_as_available(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    write_annotations(dataset_dir / "annotations")
    server = build_server(mcp_settings(monkeypatch, dataset_dir, tmp_path / "reports"))

    assert result_json(call(server, "get_dataset_summary"))["available"] is True
