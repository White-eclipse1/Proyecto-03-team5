import { useCallback, useEffect, useState } from "react";
import type { ZodType } from "zod";
import { API_BASE_URL } from "@/lib/api/client";
import { type ContractError, errorResponseSchema } from "./schemas";

/**
 * Estado de UI de cualquier fuente de las pantallas de modelos (APP-01).
 * `empty` es distinto de `success`: una lista vacía válida no es un error ni
 * se rellena con datos de ejemplo. El error siempre tiene la forma de
 * `ContractError`, venga del backend (ErrorResponse) o del propio cliente.
 */
export type MlResourceState<T> =
  | { status: "loading" }
  | { status: "error"; error: ContractError }
  | { status: "empty" }
  | { status: "success"; data: T };

const NOT_CONNECTED: ContractError = {
  code: "source_not_connected",
  message: "Esta fuente de datos todavía no está conectada.",
  retryable: true,
};

const CONTRACT_MISMATCH: ContractError = {
  code: "contract_mismatch",
  message: "La respuesta no cumple el contrato esperado.",
  retryable: false,
};

class ResourceError extends Error {
  constructor(public readonly contractError: ContractError) {
    super(contractError.message);
  }
}

export async function errorFromResponse(res: Response): Promise<ContractError> {
  if (res.status === 404) return NOT_CONNECTED;
  const body: unknown = await res.json().catch(() => null);
  const parsed = errorResponseSchema.safeParse(body);
  if (parsed.success) return parsed.data.error;
  return {
    code: `http_${res.status}`,
    message: `El servidor respondió con estado ${res.status}.`,
    retryable: res.status >= 500,
  };
}

/**
 * Fetch + validación Zod contra `API_BASE_URL` (proxy `/api`). Las fuentes
 * reales las conectan APP-02…APP-07; mientras el backend no exponga la ruta,
 * el 404 se muestra como "fuente no conectada", nunca con datos inventados.
 *
 * `reload` vuelve a mostrar "Cargando" (lo usa "Reintentar"). `refresh` es para
 * sondear (APP-03): actualiza los datos sin parpadeo y, si esa consulta falla,
 * conserva lo que ya se veía en vez de reemplazarlo por un error.
 */
export function useMlResource<T>(path: string, schema: ZodType<T>, isEmpty: (data: T) => boolean) {
  const [state, setState] = useState<MlResourceState<T>>({ status: "loading" });

  const load = useCallback(
    (silent: boolean) => {
      let cancelled = false;
      if (!silent) setState({ status: "loading" });

      fetch(`${API_BASE_URL}${path}`)
        .then(async (res) => {
          if (!res.ok) throw new ResourceError(await errorFromResponse(res));
          const parsed = schema.safeParse(await res.json());
          if (!parsed.success) throw new ResourceError(CONTRACT_MISMATCH);
          if (!cancelled) {
            setState(
              isEmpty(parsed.data) ? { status: "empty" } : { status: "success", data: parsed.data }
            );
          }
        })
        .catch((err: unknown) => {
          if (cancelled || silent) return;
          const error: ContractError =
            err instanceof ResourceError
              ? err.contractError
              : {
                  code: "network_error",
                  message: "No se pudo contactar al servidor.",
                  retryable: true,
                };
          setState({ status: "error", error });
        });

      return () => {
        cancelled = true;
      };
    },
    [path, schema, isEmpty]
  );

  useEffect(() => load(false), [load]);

  const reload = useCallback(() => load(false), [load]);
  const refresh = useCallback(() => load(true), [load]);

  return { ...state, reload, refresh };
}
