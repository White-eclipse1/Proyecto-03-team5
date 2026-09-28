import { useCallback, useEffect, useState } from "react";
import type { ZodType } from "zod";

type FetchState<T> =
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "success"; data: T };

/**
 * Fetch + validación Zod de un reporte estático servido por nginx en
 * `/reports/*.json` (ver docker-compose.yml: volumen compartido con el
 * servicio `app`, que es quien escribe ahí — sin backend HTTP nuevo).
 *
 * A diferencia de `useValidatedFetch` (hooks/), esto NO antepone
 * `API_BASE_URL`: estos archivos nunca pasan por el proxy `/api` del
 * backend Node, nginx los sirve directo desde su propio root. Un 404 es un
 * caso esperado (el contrato todavía no existe, p. ej. splits.json antes
 * de que exista el algoritmo de splits), no una falla de red distinta.
 */
export function useReportFetch<T>(url: string | null, schema: ZodType<T>) {
  const [state, setState] = useState<FetchState<T>>({ status: "loading" });

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
        if (!res.ok) {
          throw new Error(
            res.status === 404
              ? "Este reporte todavía no existe."
              : `El servidor respondió con estado ${res.status}.`
          );
        }
        const json: unknown = await res.json();
        const parsed = schema.safeParse(json);
        if (!parsed.success) {
          throw new Error("El reporte no tiene el formato esperado.");
        }
        if (!cancelled) {
          setState({ status: "success", data: parsed.data });
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const message =
          err instanceof Error ? err.message : "Error desconocido al cargar el reporte.";
        setState({ status: "error", message });
      });

    return () => {
      cancelled = true;
    };
  }, [url, schema]);

  useEffect(() => load(), [load]);

  return { ...state, reload: load };
}
