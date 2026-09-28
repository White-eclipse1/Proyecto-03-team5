import { type FormEvent, useState } from "react";
import { Button } from "@/components/ui/Button";
import { Conversation, Suggestions } from "../components/CopilotConversation";
import { PageHeader } from "../components/PageHeader";
import { useCopilotChat } from "../useCopilotChat";

/** Mismo tope que `ChatMessage.content` en app/copilot/contracts.py. */
const MAX_QUESTION_LENGTH = 4000;

function ErrorBanner({ message, onRetry }: Readonly<{ message: string; onRetry: () => void }>) {
  return (
    <div
      role="alert"
      className="flex items-center justify-between gap-3 rounded-xl border border-status-pending bg-status-pending-soft px-4 py-3 text-sm text-ink"
    >
      <span>{message}</span>
      <Button type="button" size="sm" onClick={onRetry}>
        Reintentar
      </Button>
    </div>
  );
}

/**
 * Chat del Copilot (P2-52). Cada respuesta muestra las herramientas MCP que
 * consultó y la versión del dataset; el modelo no puede mostrar cifras que no
 * vengan de una de ellas (ver `app/copilot/agent.py`).
 */
export function CopilotPage() {
  const { turns, isSending, error, send, retry } = useCopilotChat();
  const [draft, setDraft] = useState("");

  function submit(event: FormEvent) {
    event.preventDefault();
    send(draft);
    setDraft("");
  }

  return (
    <main className="flex-1 px-6 py-6 lg:px-10 lg:py-8">
      <div className="mx-auto flex max-w-3xl flex-col gap-6">
        <PageHeader title="Copilot" subtitle="Asistente de consulta sobre el dataset" />

        {turns.length === 0 && <Suggestions onPick={send} disabled={isSending} />}
        <Conversation turns={turns} isSending={isSending} />
        {error && <ErrorBanner message={error} onRetry={retry} />}

        <form onSubmit={submit} className="flex gap-2">
          <label htmlFor="copilot-question" className="sr-only">
            Pregunta para el Copilot
          </label>
          <input
            id="copilot-question"
            value={draft}
            maxLength={MAX_QUESTION_LENGTH}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Pregunta sobre el dataset…"
            className="flex-1 rounded-lg border border-border bg-surface px-3 py-2 text-sm text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-lilac/40"
          />
          <Button type="submit" variant="primary" isLoading={isSending} disabled={!draft.trim()}>
            Enviar
          </Button>
        </form>
      </div>
    </main>
  );
}
