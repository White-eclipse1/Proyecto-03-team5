"""Tier 6 — agente del Copilot (P2-52): un LLM que solo habla con datos de las herramientas.

El modelo no recibe el dataset ni los reportes: recibe las herramientas MCP de
solo lectura de `presentation/mcp_server.py` (P2-34) y las llama en un bucle
manual. El bucle es manual, y no el tool runner del SDK, para poder registrar
cada llamada real (`ToolCallTrace`): la UI muestra ese rastro y la versión del
dataset consultada sin depender de que el modelo se acuerde de citarlas.

`create_message` se inyecta (en producción es `AsyncAnthropic().messages.create`)
para probar el bucle con un LLM guionado, sin red ni API key.
"""

import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from anthropic.types import Message, ToolUseBlock
from mcp import Client, MCPError
from mcp.server.mcpserver import MCPServer
from mcp.types import CallToolResult, Tool

from copilot.contracts import ChatRequest, ChatResponse, ToolCallTrace

logger = logging.getLogger("dataset-copilot")

CreateMessage = Callable[..., Awaitable[Message]]

MAX_TOKENS = 16000
MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = """\
Eres el Copilot del pipeline de calidad del dataset (Fase 2). Respondes preguntas \
sobre el dataset, su reporte de calidad, sus splits y sus versiones publicadas.

Reglas:
1. Toda cifra, conteo, porcentaje, umbral, estado o versión que menciones debe salir \
del resultado de una herramienta que llamaste para responder. Nunca la calcules de \
memoria ni la supongas.
2. Si ninguna herramienta puede responder la pregunta, si una herramienta devuelve \
`available: false`, o si los datos no alcanzan, responde "No puedo saberlo con los \
datos disponibles" y explica qué falta. No inventes ni estimes una cifra.
3. Cita la versión del dataset consultada (el campo `dataset_version` del resultado) \
y nombra las herramientas que usaste. `local-dev` es la copia de trabajo, no un \
release; `v0.1.0` y similares son releases congelados.
4. Solo lees datos: no puedes modificar el dataset ni los reportes. Si te lo piden, \
explica que no puedes.
5. Responde en español, breve y directo.
"""


class CopilotError(Exception):
    """Falla del agente con un mensaje seguro para mostrar al usuario (sin trazas)."""


def _as_anthropic_tools(mcp_tools: list[Tool]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.input_schema,
        }
        for tool in mcp_tools
    ]


def _parse_result(result: CallToolResult) -> Any:
    text = "\n".join(block.text for block in result.content if block.type == "text")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def _call_tool(mcp_client: Client, block: ToolUseBlock) -> ToolCallTrace:
    arguments = dict(block.input)
    try:
        result = await mcp_client.call_tool(block.name, arguments)
    except MCPError as error:
        return ToolCallTrace(name=block.name, arguments=arguments, result=str(error), is_error=True)
    return ToolCallTrace(
        name=block.name,
        arguments=arguments,
        result=_parse_result(result),
        is_error=result.is_error,
    )


def _tool_result_block(block: ToolUseBlock, trace: ToolCallTrace) -> dict[str, Any]:
    tool_result: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": block.id,
        "content": json.dumps(trace.result, ensure_ascii=False),
    }
    if trace.is_error:
        tool_result["is_error"] = True
    return tool_result


def _dataset_version_of(trace: ToolCallTrace) -> str | None:
    """`dataset_version` del resultado: arriba del todo, o dentro de `data` (reportes)."""
    if trace.is_error or not isinstance(trace.result, dict):
        return None
    data = trace.result.get("data")
    nested = data.get("dataset_version") if isinstance(data, dict) else None
    version = trace.result.get("dataset_version", nested)
    return version if isinstance(version, str) else None


def _to_response(response: Message, calls: list[ToolCallTrace]) -> ChatResponse:
    if response.stop_reason == "refusal":
        raise CopilotError("El modelo no pudo responder esta consulta.")
    if response.stop_reason == "max_tokens":
        raise CopilotError("La respuesta del modelo se cortó antes de terminar. Intenta de nuevo.")
    text = "\n".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise CopilotError("El modelo no devolvió una respuesta.")
    versions = dict.fromkeys(v for trace in calls if (v := _dataset_version_of(trace)) is not None)
    return ChatResponse(answer=text, tool_calls=calls, dataset_versions=list(versions))


async def answer(
    request: ChatRequest,
    *,
    create_message: CreateMessage,
    mcp_client: Client,
    model: str,
) -> ChatResponse:
    tools = _as_anthropic_tools((await mcp_client.list_tools()).tools)
    messages: list[dict[str, Any]] = [message.model_dump() for message in request.messages]
    calls: list[ToolCallTrace] = []

    for round_number in range(1, MAX_TOOL_ROUNDS + 1):
        started = time.monotonic()
        response = await create_message(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )
        logger.info(
            "Ronda %d: stop_reason=%s tokens_in=%d tokens_out=%d (%.1fs)",
            round_number,
            response.stop_reason,
            response.usage.input_tokens,
            response.usage.output_tokens,
            time.monotonic() - started,
        )
        tool_uses = [block for block in response.content if block.type == "tool_use"]
        if response.stop_reason != "tool_use" or not tool_uses:
            return _to_response(response, calls)

        # El turno del asistente se devuelve tal cual (incluye bloques `thinking`).
        messages.append({"role": "assistant", "content": response.content})
        logger.info("Ronda %d: herramientas pedidas %s", round_number, [b.name for b in tool_uses])
        traces = [await _call_tool(mcp_client, block) for block in tool_uses]
        calls.extend(traces)
        pairs = zip(tool_uses, traces, strict=True)
        results = [_tool_result_block(block, trace) for block, trace in pairs]
        messages.append({"role": "user", "content": results})

    raise CopilotError(
        "El Copilot necesitó demasiadas consultas. Intenta una pregunta más específica."
    )


async def answer_with_server(
    server: MCPServer,
    request: ChatRequest,
    *,
    create_message: CreateMessage,
    model: str,
) -> ChatResponse:
    """Abre un cliente MCP en proceso contra `server` y corre `answer`.

    El cliente en memoria usa task groups de anyio: una excepción que se
    levanta dentro de `async with Client(...)` sale envuelta en uno o más
    `ExceptionGroup` anidados. Aquí se desenvuelven, para que quien llama pueda
    hacer `except CopilotError` / `except anthropic.APIError` sin conocer ese detalle.
    """
    try:
        async with Client(server) as mcp_client:
            return await answer(
                request, create_message=create_message, mcp_client=mcp_client, model=model
            )
    except BaseExceptionGroup as group:
        error: BaseException = group
        while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
            error = error.exceptions[0]
        if error is group:  # varias fallas a la vez: no hay una sola causa que exponer
            raise
        raise error from None
