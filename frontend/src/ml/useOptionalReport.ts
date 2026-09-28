import { useCallback, useEffect, useState } from "react";
import type { ZodType } from "zod";

export type OptionalReportState<T> =
  | { status: "loading" }
  | { status: "missing" }
  | { status: "error"; message: string }
  | { status: "success"; data: T };

/**
 * Como `useReportFetch` (pipeline/), pero un 404 es un estado propio
 * (`missing`): el archivo es opcional por diseño, p. ej. la procedencia de un
 * release cortado antes de APP-02 o el manifiesto 70/20/10 que aún no existe.
 */
export function useOptionalReport<T>(url: string | null, schema: ZodType<T>) {
  const [state, setState] = useState<OptionalReportState<T>>({ status: "loading" });

  const load = useCallback(() => {
    let cancelled = false;
    setState({ status: "loading" });

    if (url === null) {
      setState({ status: "error", message: "Referencia de reporte inválida." });
      return () => {
        cancelled = true;
      };
    }

    fetch(url)
      .then(async (res) => {
        if (res.status === 404) {
          if (!cancelled) setState({ status: "missing" });
          return;
        }
        if (!res.ok) throw new Error(`El servidor respondió con estado ${res.status}.`);
        const parsed = schema.safeParse(await res.json());
        if (!parsed.success) throw new Error("El reporte no tiene el formato esperado.");
        if (!cancelled) setState({ status: "success", data: parsed.data });
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : "Error desconocido.";
        setState({ status: "error", message });
      });

    return () => {
      cancelled = true;
    };
  }, [url, schema]);

  useEffect(() => load(), [load]);

  return state;
}
