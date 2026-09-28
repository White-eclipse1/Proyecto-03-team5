import type { AssistantTurn, Turn } from "../useCopilotChat";

export const SUGGESTIONS = [
  "¿Cuántas imágenes hay por categoría?",
  "¿Cuál es el estado del último reporte de calidad?",
  "¿Cómo quedaron los splits del último release?",
];

function formatArguments(args: Record<string, unknown>): string {
  return Object.entries(args)
    .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
    .join(", ");
}

/**
 * Qué herramientas produjeron la respuesta y de qué versión del dataset. Sale
 * del rastro que registra el servicio, no de lo que el modelo diga de sí mismo.
 */
function ToolTrace({ turn }: Readonly<{ turn: AssistantTurn }>) {
  if (turn.toolCalls.length === 0) {
    return (
      <p className="border-t border-border pt-3 text-xs text-status-pending">
        Sin consultas a herramientas: esta respuesta no usó datos del dataset.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-2 border-t border-border pt-3 text-xs text-ink-muted">
      <p>
        Versión del dataset consultada:{" "}
        <strong className="text-ink">
          {turn.datasetVersions.length > 0 ? turn.datasetVersions.join(", ") : "no informada"}
        </strong>
      </p>
      <ul className="flex flex-col gap-1">
        {turn.toolCalls.map((call) => (
          <li key={call.id}>
            <details>
              <summary className="cursor-pointer font-mono text-ink">
                {call.name}({formatArguments(call.arguments)}){call.is_error && " — falló"}
              </summary>
              <pre className="mt-1 max-h-64 overflow-auto rounded-lg bg-sidebar p-2 text-[11px] text-ink">
                {JSON.stringify(call.result, null, 2)}
              </pre>
            </details>
          </li>
        ))}
      </ul>
    </div>
  );
}

function TurnBubble({ turn }: Readonly<{ turn: Turn }>) {
  if (turn.role === "user") {
    return (
      <div className="ml-auto max-w-[85%] whitespace-pre-wrap rounded-2xl bg-accent-lilac-soft px-4 py-3 text-sm text-ink">
        {turn.content}
      </div>
    );
  }

  return (
    <div className="flex max-w-[95%] flex-col gap-3 rounded-2xl border border-border bg-surface px-4 py-3 shadow-card">
      <p className="whitespace-pre-wrap text-sm text-ink">{turn.content}</p>
      <ToolTrace turn={turn} />
    </div>
  );
}

export function Conversation({
  turns,
  isSending,
}: Readonly<{ turns: Turn[]; isSending: boolean }>) {
  return (
    <div
      role="log"
      aria-live="polite"
      aria-label="Conversación con el Copilot"
      className="flex flex-col gap-4"
    >
      {turns.map((turn) => (
        <TurnBubble key={turn.id} turn={turn} />
      ))}
      {isSending && (
        <p className="text-sm text-ink-muted">El Copilot está consultando el dataset…</p>
      )}
    </div>
  );
}

export function Suggestions({
  onPick,
  disabled,
}: Readonly<{ onPick: (question: string) => void; disabled: boolean }>) {
  return (
    <div className="rounded-2xl border border-dashed border-border-strong bg-surface px-6 py-8 text-center">
      <p className="text-sm text-ink-muted">
        Pregunta por el dataset, su reporte de calidad, los splits o las versiones. Las cifras salen
        de herramientas de solo lectura, no del modelo.
      </p>
      <div className="mt-4 flex flex-wrap justify-center gap-2">
        {SUGGESTIONS.map((question) => (
          <button
            key={question}
            type="button"
            disabled={disabled}
            onClick={() => onPick(question)}
            className="rounded-full border border-border bg-canvas px-3 py-1.5 text-xs text-ink transition-colors hover:bg-sidebar disabled:opacity-50"
          >
            {question}
          </button>
        ))}
      </div>
    </div>
  );
}
