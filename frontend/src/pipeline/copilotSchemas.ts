import { z } from "zod";

/**
 * Espeja `app/copilot/contracts.py` (P2-52). Igual que con los reportes: la
 * respuesta del agente nunca se toma por buena, se valida antes de mostrarla.
 */

export const toolCallTraceSchema = z.object({
  name: z.string(),
  arguments: z.record(z.string(), z.unknown()),
  result: z.unknown(),
  is_error: z.boolean(),
});
export type ToolCallTrace = z.infer<typeof toolCallTraceSchema>;

export const chatResponseSchema = z.object({
  answer: z.string().min(1),
  tool_calls: z.array(toolCallTraceSchema),
  dataset_versions: z.array(z.string()),
});
export type ChatResponse = z.infer<typeof chatResponseSchema>;

/** Forma de todos los errores del servicio: `{"error": "<mensaje seguro>"}`. */
export const chatErrorSchema = z.object({ error: z.string() });
