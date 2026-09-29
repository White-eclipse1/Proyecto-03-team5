import type { ReactNode } from "react";
import { ErrorState } from "@/components/ui/ErrorState";
import { Skeleton } from "@/components/ui/Skeleton";
import type { MlResourceState } from "../useMlResource";

/** Envuelve loading/error/empty/success de un `useMlResource(...)`. */
export function MlResourceBoundary<T>({
  state,
  emptyMessage,
  children,
}: Readonly<{
  state: MlResourceState<T> & { reload: () => void };
  emptyMessage: string;
  children: (data: T) => ReactNode;
}>) {
  if (state.status === "loading") {
    return (
      <div role="status" aria-live="polite" className="flex flex-col gap-4">
        <span className="sr-only">Cargando…</span>
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
      </div>
    );
  }

  if (state.status === "error") {
    const { code, message, retryable } = state.error;
    return (
      <div className="flex flex-col gap-2">
        {retryable ? (
          <ErrorState title="No se pudo cargar." message={message} onRetry={state.reload} />
        ) : (
          <div className="rounded-2xl border border-border bg-surface px-6 py-16 text-center">
            <p className="text-sm font-medium text-ink">No se pudo cargar.</p>
            <p className="mt-1 text-sm text-ink-muted">{message}</p>
          </div>
        )}
        <p className="text-center font-mono text-xs text-ink-faint">{code}</p>
      </div>
    );
  }

  if (state.status === "empty") {
    return (
      <div className="rounded-2xl border border-dashed border-border-strong bg-surface px-6 py-16 text-center">
        <p className="text-sm text-ink-muted">{emptyMessage}</p>
      </div>
    );
  }

  return <>{children(state.data)}</>;
}
