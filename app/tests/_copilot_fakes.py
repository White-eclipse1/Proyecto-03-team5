"""Doble de prueba del LLM del Copilot (P2-52): devuelve tipos reales del SDK.

Los tests del agente nunca llaman al proveedor: `ScriptedLlm` reemplaza a
`AsyncAnthropic().messages.create` con respuestas guionadas, así CI no necesita
una API key ni gasta tokens. Prefijo `_` para que pytest no lo recolecte.
"""

import json
from collections.abc import Callable
from typing import Any

from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

Step = Message | Callable[[list[dict[str, Any]]], Message]


def _message(content: list[Any], stop_reason: str) -> Message:
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-opus-5",
        content=content,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def text_message(text: str, stop_reason: str = "end_turn") -> Message:
    return _message([TextBlock(type="text", text=text)], stop_reason)


def tool_use_message(name: str, arguments: dict[str, Any] | None = None) -> Message:
    block = ToolUseBlock(type="tool_use", id=f"toolu_{name}", name=name, input=arguments or {})
    return _message([block], "tool_use")


def last_tool_result(messages: list[dict[str, Any]]) -> Any:
    """JSON ya parseado del último `tool_result` que el agente le devolvió al modelo."""
    return json.loads(messages[-1]["content"][0]["content"])


class ScriptedLlm:
    """`create_message` falso: el paso N responde a la llamada N (falla si hay de más)."""

    def __init__(self, *steps: Step) -> None:
        self._steps = steps
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Message:
        # El agente sigue mutando `messages`; se guarda una copia de lo que vio esta llamada.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        step = self._steps[len(self.calls) - 1]
        return step(kwargs["messages"]) if callable(step) else step
