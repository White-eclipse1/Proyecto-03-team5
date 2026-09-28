import { useCallback, useRef, useState } from "react";
import { chatErrorSchema, chatResponseSchema, type ToolCallTrace } from "./copilotSchemas";

/** nginx (docker-compose) y el proxy de Vite reenvían esto al servicio `copilot`. */
const CHAT_URL = "/copilot-api/chat";

/** Tope del contrato `ChatRequest` en el servicio (app/copilot/contracts.py). */
const MAX_HISTORY = 40;

export interface UserTurn {
  id: string;
  role: "user";
  content: string;
}

export interface AssistantTurn {
  id: string;
  role: "assistant";
  content: string;
  /** Llamadas reales a herramientas que produjeron la respuesta (puede ser ninguna). */
  toolCalls: Array<ToolCallTrace & { id: string }>;
  datasetVersions: string[];
}

export type Turn = UserTurn | AssistantTurn;

let turnCounter = 0;
const nextTurnId = () => `turn-${++turnCounter}`;

/** Últimos `MAX_HISTORY` turnos; el contrato exige que empiece con un mensaje del usuario. */
function recentHistory(turns: Turn[]): Turn[] {
  const recent = turns.slice(-MAX_HISTORY);
  const firstUser = recent.findIndex((turn) => turn.role === "user");
  return firstUser === -1 ? [] : recent.slice(firstUser);
}

async function requestAnswer(turns: Turn[]): Promise<AssistantTurn> {
  let response: Response;
  try {
    response = await fetch(CHAT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: recentHistory(turns).map(({ role, content }) => ({ role, content })),
      }),
    });
  } catch {
    throw new Error("No se pudo contactar al Copilot. Revisa que el servicio esté encendido.");
  }

  const body: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const error = chatErrorSchema.safeParse(body);
    if (error.success) throw new Error(error.data.error);
    throw new Error(
      response.status === 504
        ? "El Copilot tardó demasiado en responder."
        : `El Copilot respondió con estado ${response.status}.`
    );
  }

  const parsed = chatResponseSchema.safeParse(body);
  if (!parsed.success) throw new Error("La respuesta del Copilot no tiene el formato esperado.");

  const id = nextTurnId();
  return {
    id,
    role: "assistant",
    content: parsed.data.answer,
    toolCalls: parsed.data.tool_calls.map((call, position) => ({
      ...call,
      id: `${id}-tool-${position}`,
    })),
    datasetVersions: parsed.data.dataset_versions,
  };
}

/**
 * Conversación con el Copilot (P2-52). El servicio no guarda estado: cada
 * pregunta reenvía el historial. Si falla, el mensaje del usuario se conserva
 * y `retry` lo reintenta sin duplicarlo.
 */
export function useCopilotChat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inFlight = useRef(false);

  const ask = useCallback(async (history: Turn[]) => {
    if (inFlight.current) return;
    inFlight.current = true;
    setIsSending(true);
    setError(null);
    try {
      const answer = await requestAnswer(history);
      setTurns([...history, answer]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Error desconocido.");
    } finally {
      inFlight.current = false;
      setIsSending(false);
    }
  }, []);

  const send = useCallback(
    (text: string) => {
      const content = text.trim();
      if (!content || inFlight.current) return;
      const history: Turn[] = [...turns, { id: nextTurnId(), role: "user", content }];
      setTurns(history);
      void ask(history);
    },
    [turns, ask]
  );

  const retry = useCallback(() => {
    void ask(turns);
  }, [turns, ask]);

  return { turns, isSending, error, send, retry };
}
