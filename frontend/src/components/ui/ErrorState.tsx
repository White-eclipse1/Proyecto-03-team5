interface ErrorStateProps {
  /** Encabezado corto y genérico, p. ej. "No se pudo cargar el tablero." */
  title?: string;
  message: string;
  onRetry: () => void;
}

export function ErrorState({ title, message, onRetry }: ErrorStateProps): JSX.Element {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-border bg-surface px-6 py-16 text-center">
      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-status-pending-soft text-status-pending">
        !
      </div>
      {title && <p className="text-sm font-medium text-ink">{title}</p>}
      <p className="max-w-sm text-sm text-ink-muted">{message}</p>
      <button
        type="button"
        onClick={onRetry}
        className="mt-1 rounded-lg bg-ink px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-ink/90"
      >
        Reintentar
      </button>
    </div>
  );
}
