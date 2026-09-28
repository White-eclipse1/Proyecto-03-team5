import asyncio
from pathlib import Path

import pytest

from copilot.agent import MAX_TOOL_ROUNDS, SYSTEM_PROMPT, CopilotError, answer_with_server
from copilot.contracts import ChatMessage, ChatRequest
from presentation.mcp_server import build_server
from tests._copilot_fakes import ScriptedLlm, last_tool_result, text_message, tool_use_message
from tests._mcp_fixtures import mcp_settings, write_annotations, write_releases


def ask(question: str) -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role="user", content=question)])


def run(monkeypatch, tmp_path: Path, llm, request: ChatRequest):
    """Corre el agente contra un servidor MCP real sobre `tmp_path` (sin red)."""
    server = build_server(mcp_settings(monkeypatch, tmp_path / "dataset", tmp_path / "reports"))
    return asyncio.run(answer_with_server(server, request, create_message=llm, model="m"))


def report_dataset_size(messages) -> object:
    summary = last_tool_result(messages)
    return text_message(f"El dataset tiene {summary['total_images']} imágenes.")


def test_answer_uses_a_real_tool_call_and_reports_it(monkeypatch, tmp_path):
    write_annotations(tmp_path / "dataset" / "annotations", dog_count=2)
    llm = ScriptedLlm(tool_use_message("get_dataset_summary"), report_dataset_size)

    result = run(monkeypatch, tmp_path, llm, ask("¿Cuántas imágenes hay?"))

    assert result.answer == "El dataset tiene 3 imágenes."
    assert [call.name for call in result.tool_calls] == ["get_dataset_summary"]
    assert result.tool_calls[0].result["total_images"] == 3
    assert result.tool_calls[0].is_error is False


def test_changing_the_source_data_changes_the_answer(monkeypatch, tmp_path):
    """Criterio P2-52: misma pregunta, dato fuente distinto, respuesta distinta."""
    annotations_dir = tmp_path / "dataset" / "annotations"
    answers = []
    for dog_count in (2, 5):
        write_annotations(annotations_dir, dog_count=dog_count)
        llm = ScriptedLlm(tool_use_message("get_dataset_summary"), report_dataset_size)
        answers.append(run(monkeypatch, tmp_path, llm, ask("¿Cuántas imágenes hay?")).answer)

    assert answers == ["El dataset tiene 3 imágenes.", "El dataset tiene 6 imágenes."]


def test_dataset_versions_are_collected_from_tool_results_without_duplicates(monkeypatch, tmp_path):
    write_annotations(tmp_path / "dataset" / "annotations")
    write_releases(tmp_path / "reports", {"v0.1.0": "failed"})
    llm = ScriptedLlm(
        tool_use_message("get_splits_report"),
        tool_use_message("get_quality_report", {"dataset_version": "v0.1.0"}),
        tool_use_message("get_dataset_summary"),
        text_message("listo"),
    )

    result = run(monkeypatch, tmp_path, llm, ask("resume todo"))

    # v0.1.0 aparece en dos tools pero se cita una sola vez; local-dev es la copia de trabajo.
    assert result.dataset_versions == ["v0.1.0", "local-dev"]


def test_unavailable_data_is_passed_to_the_model_untouched(monkeypatch, tmp_path):
    """Sin reportes, la tool dice `available: false`; el agente no inventa ni completa nada."""
    seen = []

    def echo_availability(messages):
        seen.append(last_tool_result(messages))
        return text_message("No puedo saberlo: todavía no hay un reporte de calidad.")

    llm = ScriptedLlm(tool_use_message("get_quality_report"), echo_availability)

    result = run(monkeypatch, tmp_path, llm, ask("¿Cuál es el status de calidad?"))

    assert seen[0]["available"] is False
    assert result.dataset_versions == []
    assert "No puedo saberlo" in result.answer


def test_a_failing_tool_is_reported_as_an_error_result_not_an_exception(monkeypatch, tmp_path):
    llm = ScriptedLlm(tool_use_message("tool_que_no_existe"), text_message("no pude consultarlo"))

    result = run(monkeypatch, tmp_path, llm, ask("hola"))

    assert result.tool_calls[0].is_error is True
    assert llm.calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_every_llm_call_gets_the_grounding_prompt_and_the_five_mcp_tools(monkeypatch, tmp_path):
    llm = ScriptedLlm(text_message("hola"))

    run(monkeypatch, tmp_path, llm, ask("hola"))

    call = llm.calls[0]
    assert call["system"] == SYSTEM_PROMPT
    assert call["model"] == "m"
    assert {tool["name"] for tool in call["tools"]} == {
        "get_dataset_summary",
        "get_quality_report",
        "get_check_result",
        "get_splits_report",
        "get_versions_report",
    }
    assert all(tool["description"] and tool["input_schema"] for tool in call["tools"])


def test_grounding_prompt_forbids_invented_figures_and_demands_citations():
    assert "no puedo saberlo" in SYSTEM_PROMPT.lower()
    assert "dataset_version" in SYSTEM_PROMPT
    assert "herramienta" in SYSTEM_PROMPT.lower()


def test_the_conversation_history_reaches_the_model(monkeypatch, tmp_path):
    request = ChatRequest(
        messages=[
            ChatMessage(role="user", content="hola"),
            ChatMessage(role="assistant", content="¿en qué te ayudo?"),
            ChatMessage(role="user", content="¿cuántas imágenes?"),
        ]
    )
    llm = ScriptedLlm(text_message("ok"))

    run(monkeypatch, tmp_path, llm, request)

    assert [m["role"] for m in llm.calls[0]["messages"]] == ["user", "assistant", "user"]


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [("refusal", "no pudo responder"), ("max_tokens", "se cortó")],
)
def test_unusable_model_stops_become_a_user_safe_error(
    monkeypatch, tmp_path, stop_reason, expected
):
    llm = ScriptedLlm(text_message("parcial", stop_reason=stop_reason))
    request = ask("hola")

    with pytest.raises(CopilotError, match=expected):
        run(monkeypatch, tmp_path, llm, request)


def test_an_empty_model_answer_becomes_a_user_safe_error(monkeypatch, tmp_path):
    llm = ScriptedLlm(text_message("   "))
    request = ask("hola")

    with pytest.raises(CopilotError, match="no devolvió"):
        run(monkeypatch, tmp_path, llm, request)


def test_a_model_that_never_stops_calling_tools_is_cut_off(monkeypatch, tmp_path):
    write_annotations(tmp_path / "dataset" / "annotations")
    llm = ScriptedLlm(*[tool_use_message("get_dataset_summary")] * MAX_TOOL_ROUNDS)
    request = ask("hola")

    with pytest.raises(CopilotError, match="demasiadas consultas"):
        run(monkeypatch, tmp_path, llm, request)

    assert len(llm.calls) == MAX_TOOL_ROUNDS
