import type { PendingEntry } from "../../hooks/useUploadQueue";
import { getImageFileUrl } from "../../lib/api/images";
import { StatusBadge } from "../search/StatusBadge";

function DeleteIcon() {
  return (
    <svg viewBox="0 0 20 20" className="h-4 w-4" fill="currentColor" aria-hidden>
      <path
        fillRule="evenodd"
        d="M8.75 1a.75.75 0 0 0-.75.75V2h-3a.75.75 0 0 0 0 1.5h.3l.7 12.2A2 2 0 0 0 7.99 17.5h4.02a2 2 0 0 0 1.99-1.8l.7-12.2h.3a.75.75 0 0 0 0-1.5h-3v-.25a.75.75 0 0 0-.75-.75h-2.5Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function PendingImageList({
  entries,
  isLoading,
  error,
  selectedIds,
  onToggle,
  onDelete,
  onRetry,
}: {
  entries: PendingEntry[];
  isLoading: boolean;
  error: string | null;
  selectedIds: Set<number>;
  onToggle: (id: number) => void;
  onDelete: (id: number) => void;
  onRetry: () => void;
}) {
  if (isLoading && entries.length === 0) {
    return <p className="text-sm text-ink-faint">Cargando imágenes pendientes…</p>;
  }

  if (error) {
    return (
      <div className="flex items-center gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
        <span className="flex-1">{error}</span>
        <button
          type="button"
          onClick={onRetry}
          className="rounded-lg border border-red-200 bg-surface px-2 py-1 text-xs font-medium transition-colors hover:bg-red-100"
        >
          Reintentar
        </button>
      </div>
    );
  }

  if (entries.length === 0) {
    return <p className="text-sm text-ink-faint">No hay imágenes pendientes por anotar.</p>;
  }

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
      {entries.map((entry) => {
        const checked = selectedIds.has(entry.id);
        return (
          <div
            key={entry.id}
            className={`group relative flex flex-col overflow-hidden rounded-xl border bg-surface shadow-card transition-colors ${
              checked ? "border-accent-lilac ring-2 ring-accent-lilac/20" : "border-border"
            }`}
          >
            <label className="relative block cursor-pointer">
              <input
                type="checkbox"
                className="absolute left-2 top-2 z-10 h-4 w-4 rounded border-border accent-accent-lilac"
                checked={checked}
                onChange={() => onToggle(entry.id)}
              />
              <img
                src={entry.previewUrl ?? getImageFileUrl(entry.id)}
                alt={entry.filename}
                className="aspect-square w-full object-cover"
              />
              <span className="absolute right-2 top-2 z-10">
                <StatusBadge status={entry.status} />
              </span>
            </label>

            <div className="flex items-center gap-1 px-2 py-1.5">
              <span className="min-w-0 flex-1 truncate text-xs text-ink-muted">
                {entry.filename}
              </span>
              <button
                type="button"
                aria-label={`Eliminar ${entry.filename}`}
                onClick={() => onDelete(entry.id)}
                className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-ink-faint transition-colors hover:bg-red-50 hover:text-red-600"
              >
                <DeleteIcon />
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
