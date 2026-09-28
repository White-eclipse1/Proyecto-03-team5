import type { z } from "zod";

/**
 * Nunca se importa Drizzle, esquemas SQL ni clientes de MinIO aquí.
 * Todo pasa por HTTP contra el backend.
 */

/**
 * Si la variable no está definida o viene vacía, se usa el proxy `/api`
 * declarado en vite.config.ts, que reenvía al backend. Una cadena vacía no
 * activa `??`, por eso se comprueba explícitamente.
 */
const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL as string | undefined;

export const API_BASE_URL: string =
  configuredBaseUrl !== undefined && configuredBaseUrl.trim() !== "" ? configuredBaseUrl : "/api";

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * Extrae el mensaje de un body de error ya parseado como JSON.
 *
 * El backend responde `{ "error": "..." }`, y algunos endpoints usan
 * `{ "message": "..." }`, así que se aceptan ambas claves para que el usuario
 * vea siempre el motivo real en lugar de un mensaje genérico.
 */
export function extractMessageField(body: unknown): string | null {
  if (body === null || typeof body !== "object") return null;

  const candidate = body as { error?: unknown; message?: unknown };

  if (typeof candidate.error === "string") return candidate.error;
  if (typeof candidate.message === "string") return candidate.message;

  return null;
}

async function extractErrorMessage(res: Response, fallback: string): Promise<string> {
  try {
    const body: unknown = await res.json();
    return extractMessageField(body) ?? fallback;
  } catch {
    // el body no era JSON, se usa el mensaje por defecto
    return fallback;
  }
}

function defaultMessageForStatus(status: number): string {
  if (status === 400) return "Solicitud inválida.";
  if (status === 404) return "No se encontró el recurso solicitado.";
  if (status === 413) return "El archivo excede el tamaño máximo permitido (5 MiB).";
  if (status >= 500) return "Error del servidor. Intenta de nuevo.";
  return `Error inesperado (${status}).`;
}

/**
 * Llama al backend y valida la respuesta JSON con el schema dado.
 */
export async function apiRequest<T>(
  path: string,
  schema: z.ZodType<T>,
  init?: RequestInit
): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(0, "No se pudo conectar con el servidor.");
  }

  if (!res.ok) {
    const message = await extractErrorMessage(res, defaultMessageForStatus(res.status));
    throw new ApiError(res.status, message);
  }

  const json: unknown = await res.json();
  const parsed = schema.safeParse(json);
  if (!parsed.success) {
    throw new Error(`Respuesta inválida del servidor en ${path}: ${parsed.error.message}`);
  }
  return parsed.data;
}

/**
 * Variante para endpoints que responden 204 No Content (p. ej. DELETE, PATCH
 * de status) y por lo tanto no tienen body que validar.
 */
export async function apiRequestVoid(path: string, init?: RequestInit): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(0, "No se pudo conectar con el servidor.");
  }
  if (!res.ok) {
    const message = await extractErrorMessage(res, defaultMessageForStatus(res.status));
    throw new ApiError(res.status, message);
  }
}

export function jsonBody(body: unknown): RequestInit {
  return {
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}
