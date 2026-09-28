import type { ImageStatus } from "../../types/schemas";
import { StatusBadge } from "../search/StatusBadge";

export function Toolbar({
  filename,
  status,
  zoom,
  onBack,
  onZoomIn,
  onZoomOut,
  onZoomReset,
  canUndo,
  onUndo,
  onFinalize,
  isFinalizing,
}: {
  filename: string;
  status: ImageStatus;
  zoom: number;
  onBack: () => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
  onZoomReset: () => void;
  canUndo: boolean;
  onUndo: () => void;
  onFinalize: () => void;
  isFinalizing: boolean;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border bg-surface px-4 py-2.5 shadow-card">
      <div className="flex min-w-0 items-center gap-3">
        <button
          type="button"
          onClick={onBack}
          className="flex items-center gap-1 rounded-lg px-2 py-1.5 text-sm font-medium text-ink-muted transition-colors hover:bg-sidebar hover:text-ink"
        >
          <svg viewBox="0 0 20 20" className="h-4 w-4" fill="currentColor" aria-hidden>
            <path
              fillRule="evenodd"
              d="M12.79 5.23a.75.75 0 0 1 0 1.06L9.06 10l3.73 3.71a.75.75 0 1 1-1.06 1.06l-4.25-4.25a.75.75 0 0 1 0-1.06l4.25-4.25a.75.75 0 0 1 1.06 0Z"
              clipRule="evenodd"
            />
          </svg>
          Volver
        </button>
        <span className="h-4 w-px bg-border" aria-hidden />
        <p className="truncate text-sm font-medium text-ink">{filename}</p>
        <StatusBadge status={status} />
      </div>

      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1 rounded-lg border border-border px-1">
          <button
            type="button"
            onClick={onZoomOut}
            aria-label="Alejar"
            className="rounded-md px-2 py-1 text-sm text-ink-muted transition-colors hover:bg-sidebar"
          >
            −
          </button>
          <button
            type="button"
            onClick={onZoomReset}
            className="min-w-[3.5rem] rounded-md px-1 py-1 text-xs text-ink-faint transition-colors hover:bg-sidebar"
          >
            {Math.round(zoom * 100)}%
          </button>
          <button
            type="button"
            onClick={onZoomIn}
            aria-label="Acercar"
            className="rounded-md px-2 py-1 text-sm text-ink-muted transition-colors hover:bg-sidebar"
          >
            +
          </button>
        </div>

        <button
          type="button"
          disabled={!canUndo}
          onClick={onUndo}
          className="rounded-lg border border-border px-3 py-1.5 text-sm font-medium text-ink-muted transition-colors hover:bg-sidebar disabled:cursor-not-allowed disabled:text-ink-faint/60"
        >
          Deshacer
          <span className="ml-1 text-xs text-ink-faint">⌘Z</span>
        </button>

        <button
          type="button"
          disabled={status === "completed" || isFinalizing}
          onClick={onFinalize}
          className="rounded-lg bg-accent-lilac px-3 py-1.5 text-sm font-medium text-white shadow-sm transition-colors hover:bg-accent-lilac/90 disabled:cursor-not-allowed disabled:bg-border disabled:text-ink-faint disabled:shadow-none"
        >
          {status === "completed" ? "Finalizada" : isFinalizing ? "Finalizando…" : "Finalizar"}
        </button>
      </div>
    </div>
  );
}
