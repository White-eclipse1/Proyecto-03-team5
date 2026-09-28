"""Tier 6 — endpoint HTTP del Copilot (P2-52).

`POST /chat` recibe la conversación completa (la API no guarda estado), la pasa
al agente y devuelve la respuesta más el rastro de herramientas y las versiones
del dataset consultadas. `GET /health` dice si hay API key configurada.

Ningún error del proveedor de IA llega al usuario como traza: se registra en el
log del servicio y la respuesta es un JSON `{"error": "..."}` con un mensaje
fijo. La API key solo se lee en `storage/settings.py`.

Desde `app/` (necesita las mismas variables que `presentation.mcp_server`):

    uv run python -m copilot.server
"""

import logging

import anthropic
import uvicorn
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from copilot.agent import CopilotError, CreateMessage, answer_with_server
from copilot.contracts import ChatRequest
from presentation.mcp_server import build_server
from storage.settings import Settings

logger = logging.getLogger("dataset-copilot")

PROVIDER_TIMEOUT_SECONDS = 45.0


def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def _provider_create_message(settings: Settings) -> CreateMessage | None:
    if settings.anthropic_api_key is None:
        return None
    client = anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key.get_secret_value(),
        timeout=PROVIDER_TIMEOUT_SECONDS,
        max_retries=1,
    )
    return client.messages.create


def _failure_response(error: CopilotError | anthropic.APIError) -> JSONResponse:
    """Traduce una falla esperada a un mensaje fijo; el detalle va solo al log."""
    if isinstance(error, CopilotError):
        return _error(502, str(error))
    if isinstance(error, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        logger.error("El proveedor rechazó las credenciales del Copilot: %s", type(error).__name__)
        return _error(503, "El Copilot no pudo autenticarse con el proveedor de IA. Revisa la key.")
    logger.error("Falla del proveedor de IA: %s", type(error).__name__, exc_info=True)
    return _error(502, "El proveedor de IA no respondió. Intenta de nuevo en unos minutos.")


def create_app(
    settings: Settings | None = None, *, create_message: CreateMessage | None = None
) -> Starlette:
    settings = settings if settings is not None else Settings()
    create_message = create_message or _provider_create_message(settings)

    def health(_: Request) -> JSONResponse:
        # Sin `async`: no espera nada. Starlette corre las funciones normales en un hilo aparte.
        return JSONResponse({"status": "ok", "configured": create_message is not None})

    async def chat(request: Request) -> JSONResponse:
        if create_message is None:
            return _error(503, "El Copilot no está configurado: falta ANTHROPIC_API_KEY.")
        try:
            payload = ChatRequest.model_validate_json(await request.body())
        except ValidationError:
            return _error(
                422, "Solicitud inválida: envía una conversación que empiece y termine contigo."
            )
        try:
            result = await answer_with_server(
                build_server(settings),
                payload,
                create_message=create_message,
                model=settings.copilot_model,
            )
        except (CopilotError, anthropic.APIError) as error:
            return _failure_response(error)
        return JSONResponse(result.model_dump(mode="json"))

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/chat", chat, methods=["POST"]),
        ]
    )


def main() -> None:
    settings = Settings()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(settings), host=settings.copilot_host, port=settings.copilot_port)


if __name__ == "__main__":
    main()
