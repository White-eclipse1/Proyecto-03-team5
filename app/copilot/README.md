# Copilot (P2-52) — cliente LLM sobre las herramientas MCP

El Copilot responde preguntas sobre el dataset, su reporte de calidad, los
splits y las versiones. **El modelo no ve el dataset: solo puede consultar las
5 herramientas de solo lectura de `presentation/mcp_server.py` (P2-34)**, y las
cifras que muestra salen de lo que esas herramientas devuelven.

```
Copilot.tsx ──POST /copilot-api/chat──▶ nginx ──▶ copilot.server (Starlette)
                                                      │
                                       agent.answer_with_server
                                        │                    │
                                  AsyncAnthropic        mcp.Client (en proceso)
                                  messages.create        └▶ presentation.mcp_server
                                                            (solo lectura)
```

| Archivo | Qué hace |
|---|---|
| `contracts.py` | Contratos JSON del endpoint (`ChatRequest`, `ChatResponse`, `ToolCallTrace`). El frontend los espeja con zod en `frontend/src/pipeline/copilotSchemas.ts`. |
| `agent.py` | Bucle modelo ↔ herramientas y el prompt de sistema. Registra cada llamada real a una herramienta. |
| `server.py` | `POST /chat` y `GET /health`. Traduce fallas a mensajes fijos, sin trazas. |

## Cómo se cumplen los criterios de aceptación

- **Cifras que salen de tool calls, no del modelo.** El prompt de sistema
  (`SYSTEM_PROMPT`) obliga a que toda cifra venga de una herramienta y el
  bucle solo le da al modelo esas herramientas. Además, cada respuesta trae
  `tool_calls` con el resultado crudo de cada herramienta, y la UI lo muestra
  en un `<details>` para poder auditar la cifra. Una respuesta sin ninguna
  llamada se marca en pantalla como "Sin consultas a herramientas".
- **Cambiar el dato fuente cambia la respuesta.** Las herramientas leen
  `DATASET_DIR`/`REPORTS_DIR` en cada llamada, sin caché. Cubierto por
  `test_changing_the_source_data_changes_the_answer`.
- **"No puedo saberlo" en vez de una cifra inventada.** Cuando un reporte no
  existe, la herramienta devuelve `{"available": false, "reason": ...}` y el
  prompt indica responder "No puedo saberlo con los datos disponibles". Eso
  último lo decide el modelo: los tests automáticos verifican el prompt y que
  el agente pase `available: false` sin tocarlo. Con la API real (`claude-opus-5`,
  2026-09-18) se probó a mano con "¿qué exactitud va a tener un modelo entrenado
  con este dataset?" y respondió que no puede saberlo, sin inventar cifras; es
  una prueba de una pregunta, no una garantía.
- **Cita la versión del dataset y las herramientas.** La respuesta trae
  `dataset_versions`, calculado por el servidor a partir del campo
  `dataset_version` de los resultados de las herramientas (no depende de que el
  modelo se acuerde de citarla). `local-dev` es la copia de trabajo; `v0.1.0` y
  similares son releases congelados de `versions.json`.
- **API key desde `.env`, con manejo de errores.** Solo `storage/settings.py`
  lee `ANTHROPIC_API_KEY` (`SecretStr`, no aparece en `repr`). Sin key el
  servicio arranca igual y `/chat` responde 503 diciendo qué falta. Si el
  proveedor falla, la respuesta es `{"error": "..."}` con un mensaje fijo; el
  detalle va solo al log del servicio.

## Configuración

| Variable | Default | Para qué |
|---|---|---|
| `ANTHROPIC_API_KEY` | (vacío) | Key del proveedor. Sin ella, el chat responde 503. |
| `COPILOT_MODEL` | `claude-opus-5` | Modelo. `claude-sonnet-5` cuesta menos por token. |
| `COPILOT_HOST` / `COPILOT_PORT` | `127.0.0.1` / `8000` | Dónde escucha. Compose abre el host a la red interna. |

Con Docker (`docker compose up --build`), pon la key en el `.env` de la raíz
(ver `.env.example`); el servicio `copilot` la recibe y nginx reenvía
`/copilot-api/` desde el frontend (`http://localhost:8080`). No se publica
ningún puerto del servicio.

Sin Docker, en dos terminales:

```bash
# app/  — mismas variables que presentation.mcp_server, más la key
ANTHROPIC_API_KEY=sk-ant-... DATABASE_URL=... MINIO_ENDPOINT=... MINIO_PORT=... \
MINIO_ACCESS_KEY=... MINIO_SECRET_KEY=... MINIO_BUCKET=... \
DATASET_DIR=../data/raw REPORTS_DIR=../reports \
uv run python -m copilot.server

# frontend/  — el proxy de Vite reenvía /copilot-api a localhost:8000
npm run dev
```

## Decisiones de diseño

- **Bucle manual, no el tool runner del SDK.** El tool runner esconde cada
  llamada; aquí hace falta el rastro para mostrarlo y para calcular
  `dataset_versions`. Además `create_message` se inyecta, así los tests usan
  un LLM guionado (`tests/_copilot_fakes.py`) que devuelve tipos reales del SDK
  y CI no necesita key ni gasta tokens.
- **Cliente MCP en proceso** (`mcp.Client(build_server(settings))`), no un
  subproceso por stdio: mismas herramientas y mismo protocolo, sin lanzar un
  Python por pregunta. `Client` acepta también `StdioServerParameters` o una
  URL si algún día el servidor MCP vive aparte.
- **API sin estado.** El frontend reenvía el historial en cada pregunta; el
  servicio no guarda conversaciones. Tope de 40 mensajes de 4000 caracteres.
- **Sin `fallbacks` de rechazo del proveedor.** Un `stop_reason: "refusal"` se
  devuelve como error limpio ("El modelo no pudo responder esta consulta"). La
  opción beta de reintento server-side no se activó: no se pudo verificar su
  forma sin una key real.
- **Timeouts:** 45 s por llamada al proveedor con 1 reintento; nginx espera hasta
  180 s por respuesta (varias rondas de herramientas). Máximo 8 rondas.

## Observabilidad

Cada ronda registra en el log del servicio `copilot` el `stop_reason`, los
tokens de entrada y salida y la duración, y qué herramientas pidió el modelo
(solo nombres, nunca la key ni el contenido de las preguntas). Sirve para
depurar respuestas lentas y para estimar costo: una pregunta típica de dos
rondas usa del orden de 7 mil tokens de entrada y 600 de salida.

## Pruebas

```bash
uv run pytest tests/test_copilot_agent.py tests/test_copilot_server.py   # app/
npx vitest run tests/copilot.test.tsx                                    # frontend/
```

Ninguna prueba automática llama al proveedor.
