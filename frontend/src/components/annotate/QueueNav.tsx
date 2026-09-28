interface QueueNavProps {
  queue: number[];
  currentIndex: number;
  onGoTo: (index: number) => void;
  onFinishQueue: () => void;
}

export function QueueNav({ queue, currentIndex, onGoTo, onFinishQueue }: QueueNavProps) {
  if (queue.length === 0) return null;

  const isLast = currentIndex === queue.length - 1;
  const isFirst = currentIndex === 0;

  return (
    <div className="flex items-center justify-between gap-3 rounded-xl border border-border bg-surface px-4 py-2.5 shadow-card">
      <p className="text-xs text-ink-muted">
        Imagen {currentIndex + 1} de {queue.length} en la cola
      </p>
      <div className="flex items-center gap-2">
        <button
          type="button"
          disabled={isFirst}
          onClick={() => onGoTo(currentIndex - 1)}
          className="rounded-lg border border-border px-3 py-1.5 text-sm font-medium text-ink-muted transition-colors hover:bg-sidebar disabled:cursor-not-allowed disabled:text-ink-faint/60"
        >
          ← Anterior
        </button>
        {isLast ? (
          <button
            type="button"
            onClick={onFinishQueue}
            className="rounded-lg bg-accent-lilac px-3 py-1.5 text-sm font-medium text-white shadow-sm transition-colors hover:bg-accent-lilac/90"
          >
            Terminar cola
          </button>
        ) : (
          <button
            type="button"
            onClick={() => onGoTo(currentIndex + 1)}
            className="rounded-lg bg-accent-lilac px-3 py-1.5 text-sm font-medium text-white shadow-sm transition-colors hover:bg-accent-lilac/90"
          >
            Guardar y siguiente →
          </button>
        )}
      </div>
    </div>
  );
}
