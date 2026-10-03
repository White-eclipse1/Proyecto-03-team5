import { eq } from 'drizzle-orm';

import { db } from '../db/client.js';
import {
  type InferenceQueueEntry,
  inferenceQueueEntries,
  type NewInferenceQueueEntry,
} from '../db/schema.js';

/**
 * Busca una inferencia ya enviada a la cola.
 *
 * APP-08 usa esta consulta para que un retry o doble click sea idempotente.
 */
export async function findInferenceQueueEntryByKey(
  idempotencyKey: string,
): Promise<InferenceQueueEntry | null> {
  const rows = await db
    .select()
    .from(inferenceQueueEntries)
    .where(eq(inferenceQueueEntries.idempotencyKey, idempotencyKey))
    .limit(1);

  return rows[0] ?? null;
}

/**
 * Persiste la trazabilidad ML asociada a una imagen real de la cola.
 */
export async function createInferenceQueueEntry(
  entry: NewInferenceQueueEntry,
): Promise<InferenceQueueEntry> {
  const result = await db.insert(inferenceQueueEntries).values(entry).$returningId();
  const created = result[0];

  if (!created) {
    throw new Error('No se pudo crear la entrada de inferencia en la cola.');
  }

  const rows = await db
    .select()
    .from(inferenceQueueEntries)
    .where(eq(inferenceQueueEntries.id, created.id))
    .limit(1);

  const row = rows[0];

  if (!row) {
    throw new Error('No se pudo recuperar la entrada de inferencia recién creada.');
  }

  return row;
}
