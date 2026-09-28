import type { Toast } from "../../hooks/useToasts";

const VARIANT_STYLES: Record<Toast["variant"], string> = {
  error: "border-red-200 bg-red-50 text-red-800",
  success: "border-status-done-soft bg-status-done-soft text-status-done",
  info: "border-border bg-surface text-ink",
};

export function ToastStack({
  toasts,
  onDismiss,
}: {
  toasts: Toast[];
  onDismiss: (id: number) => void;
}) {
  if (toasts.length === 0) return null;

  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-80 flex-col gap-2">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          role="status"
          className={`pointer-events-auto flex items-start justify-between gap-3 rounded-xl border px-4 py-3 text-sm shadow-sm ${VARIANT_STYLES[toast.variant]}`}
        >
          <span>{toast.message}</span>
          <button
            type="button"
            onClick={() => onDismiss(toast.id)}
            className="text-ink-faint transition-colors hover:text-ink-muted"
            aria-label="Cerrar notificación"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
