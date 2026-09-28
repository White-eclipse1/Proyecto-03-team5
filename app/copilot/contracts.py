"""Contratos JSON del endpoint del Copilot (P2-52); solo validación, sin I/O."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class CopilotModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ChatMessage(CopilotModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(CopilotModel):
    """Conversación completa: la API es sin estado, el cliente reenvía el historial."""

    messages: list[ChatMessage] = Field(min_length=1, max_length=40)

    @model_validator(mode="after")
    def starts_and_ends_with_the_user(self) -> Self:
        if self.messages[0].role != "user" or self.messages[-1].role != "user":
            raise ValueError("la conversación debe empezar y terminar con un mensaje del usuario")
        return self


class ToolCallTrace(CopilotModel):
    """Una llamada real a una herramienta MCP: qué se pidió y qué devolvió."""

    name: str
    arguments: dict[str, JsonValue]
    result: JsonValue
    is_error: bool


class ChatResponse(CopilotModel):
    answer: str
    tool_calls: list[ToolCallTrace]
    dataset_versions: list[str]
