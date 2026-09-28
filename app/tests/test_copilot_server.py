import anthropic
import httpx2
import pytest
from pydantic import SecretStr
from starlette.testclient import TestClient

from copilot.agent import CopilotError
from copilot.contracts import ChatMessage, ChatRequest
from copilot.server import create_app
from tests._copilot_fakes import ScriptedLlm, text_message, tool_use_message
from tests._mcp_fixtures import mcp_settings, write_annotations

SECRET = "sk-ant-secret-do-not-leak"
QUESTION = {"messages": [{"role": "user", "content": "¿Cuántas imágenes hay?"}]}


def client_for(monkeypatch, tmp_path, llm, *, with_key=True) -> TestClient:
    settings = mcp_settings(monkeypatch, tmp_path / "dataset", tmp_path / "reports")
    if with_key:
        settings = settings.model_copy(update={"anthropic_api_key": SecretStr(SECRET)})
    return TestClient(create_app(settings, create_message=llm))


def failing_llm(error: Exception):
    async def create_message(**_):
        raise error

    return create_message


def provider_error(kind: type[anthropic.APIStatusError], status_code: int):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status_code, request=request)
    return kind("provider said no", response=response, body=None)


def test_chat_returns_the_answer_with_its_tool_trace_and_dataset_version(monkeypatch, tmp_path):
    write_annotations(tmp_path / "dataset" / "annotations")
    llm = ScriptedLlm(tool_use_message("get_dataset_summary"), text_message("Hay 3 imágenes."))

    response = client_for(monkeypatch, tmp_path, llm).post("/chat", json=QUESTION)

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Hay 3 imágenes."
    assert body["dataset_versions"] == ["local-dev"]
    assert [call["name"] for call in body["tool_calls"]] == ["get_dataset_summary"]
    assert body["tool_calls"][0]["result"]["total_images"] == 3


def test_chat_uses_the_configured_model(monkeypatch, tmp_path):
    llm = ScriptedLlm(text_message("hola"))
    settings = mcp_settings(monkeypatch, tmp_path / "dataset", tmp_path / "reports")
    settings = settings.model_copy(update={"copilot_model": "claude-sonnet-5"})

    TestClient(create_app(settings, create_message=llm)).post("/chat", json=QUESTION)

    assert llm.calls[0]["model"] == "claude-sonnet-5"


def test_without_an_api_key_chat_says_so_and_health_reports_it(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path, None, with_key=False)

    chat = client.post("/chat", json=QUESTION)
    health = client.get("/health")

    assert chat.status_code == 503
    assert "ANTHROPIC_API_KEY" in chat.json()["error"]
    assert health.json() == {"status": "ok", "configured": False}


@pytest.mark.parametrize(
    "body",
    [
        {"messages": []},
        {"messages": [{"role": "assistant", "content": "hola"}]},
        {"messages": [{"role": "user", "content": ""}]},
        {"mensajes": "no"},
    ],
)
def test_invalid_requests_are_rejected_before_reaching_the_model(monkeypatch, tmp_path, body):
    llm = ScriptedLlm()

    response = client_for(monkeypatch, tmp_path, llm).post("/chat", json=body)

    assert response.status_code == 422
    assert llm.calls == []


def test_malformed_json_is_rejected_not_a_server_error(monkeypatch, tmp_path):
    response = client_for(monkeypatch, tmp_path, ScriptedLlm()).post(
        "/chat", content=b"{no es json"
    )

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x")), 502),
        (provider_error(anthropic.InternalServerError, 500), 502),
        (provider_error(anthropic.RateLimitError, 429), 502),
        (provider_error(anthropic.AuthenticationError, 401), 503),
        (provider_error(anthropic.PermissionDeniedError, 403), 503),
    ],
)
def test_provider_failures_become_a_clean_json_error_without_traceback(
    monkeypatch, tmp_path, error, status_code
):
    client = client_for(monkeypatch, tmp_path, failing_llm(error))

    response = client.post("/chat", json=QUESTION)

    assert response.status_code == status_code
    assert list(response.json()) == ["error"]
    assert "Traceback" not in response.text
    assert "provider said no" not in response.text
    assert SECRET not in response.text


def test_agent_failures_are_reported_with_their_own_safe_message(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path, failing_llm(CopilotError("demasiadas consultas")))

    response = client.post("/chat", json=QUESTION)

    assert response.status_code == 502
    assert response.json() == {"error": "demasiadas consultas"}


def test_the_api_key_never_appears_in_any_response(monkeypatch, tmp_path):
    llm = ScriptedLlm(text_message("ok"))
    client = client_for(monkeypatch, tmp_path, llm)

    responses = [client.get("/health"), client.post("/chat", json=QUESTION)]

    assert all(SECRET not in response.text for response in responses)


@pytest.mark.parametrize("roles", [["assistant"], ["user", "assistant"], ["assistant", "user"]])
def test_chat_request_must_start_and_end_with_the_user(roles):
    messages = [ChatMessage(role=role, content="x") for role in roles]

    with pytest.raises(ValueError, match="empezar y terminar"):
        ChatRequest(messages=messages)


def test_chat_request_accepts_a_multi_turn_conversation_ending_with_the_user():
    roles = ["user", "assistant", "user"]

    request = ChatRequest(messages=[ChatMessage(role=role, content="x") for role in roles])

    assert len(request.messages) == 3


def test_with_a_key_and_no_injected_client_the_real_provider_client_is_built(monkeypatch, tmp_path):
    settings = mcp_settings(monkeypatch, tmp_path / "dataset", tmp_path / "reports")
    settings = settings.model_copy(update={"anthropic_api_key": SecretStr(SECRET)})

    health = TestClient(create_app(settings)).get("/health")

    assert health.json() == {"status": "ok", "configured": True}
