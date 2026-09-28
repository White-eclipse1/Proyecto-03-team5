import type { ReactNode } from "react";
import { ErrorState } from "@/components/ui/ErrorState";
import { Skeleton } from "@/components/ui/Skeleton";

type ReportState<T> =
  | { status: "loading"; reload: () => void }
  | { status: "error"; message: string; reload: () => void }
  | { status: "success"; data: T; reload: () => void };

/** Envuelve el loading/error/success de un `useReportFetch(...)` (ver dataSource.ts). */
export function ReportBoundary<T>({
  state,
  children,
}: Readonly<{
  state: ReportState<T>;
  children: (data: T) => ReactNode;
}>) {
  if (state.status === "loading") {
    return (
      <div className="flex flex-col gap-4">
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <ErrorState
        title="No se pudo cargar el reporte."
        message={state.message}
        onRetry={state.reload}
      />
    );
  }

  return <>{children(state.data)}</>;
}
