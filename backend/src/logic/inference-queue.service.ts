import {
  createInferenceQueueEntry,
  findImageById,
  findInferenceQueueEntryByKey,
} from '../data/index.js';

import { deleteImage, type UploadImageInput, uploadImage } from './image-upload.service.js';
import {
  buildInferenceQueueKey,
  type InferenceQueueMetadata,
  inferenceQueueMetadataSchema,
} from './inference-queue.validation.js';

export interface EnqueueInferenceInput {
  image: UploadImageInput;
  metadata: InferenceQueueMetadata;
}

export interface EnqueueInferenceResult {
  imageId: number;
  created: boolean;
  idempotencyKey: string;
}

/**
 * APP-08 — convierte el resultado de Inference en una entrada REAL de la
 * cola existente.
 *
 * El archivo se guarda usando exactamente el flujo normal:
 * MinIO -> images -> status pending.
 *
 * La tabla inference_queue_entries únicamente agrega trazabilidad ML.
 */
export async function enqueueInferenceResult(
  input: EnqueueInferenceInput,
): Promise<EnqueueInferenceResult> {
  const metadata = inferenceQueueMetadataSchema.parse(input.metadata);
  const idempotencyKey = buildInferenceQueueKey(metadata);

  // Retry o doble click después de que la primera petición terminó.
  const existing = await findInferenceQueueEntryByKey(idempotencyKey);

  if (existing) {
    const image = await findImageById(existing.imageId);

    if (image) {
      return {
        imageId: image.id,
        created: false,
        idempotencyKey,
      };
    }
  }

  // Utiliza la misma ruta real de uploads del portal:
  // objeto en MinIO + registro pending en MariaDB.
  const uploaded = await uploadImage(input.image);

  try {
    await createInferenceQueueEntry({
      imageId: uploaded.id,
      idempotencyKey,
      sourceKind: metadata.sourceKind,
      sourceRef: metadata.sourceRef,
      modelName: metadata.modelName,
      modelVersion: metadata.modelVersion,
      runId: metadata.runId,
      checkpointSha256: metadata.checkpointSha256,
      predictedClass: metadata.predictedClass,
      probabilityDog: metadata.probabilities.dog,
      probabilityCat: metadata.probabilities.cat,
    });

    return {
      imageId: uploaded.id,
      created: true,
      idempotencyKey,
    };
  } catch (error) {
    /*
     * Dos requests idénticos pueden llegar simultáneamente:
     *
     * A: comprueba inexistencia -> upload
     * B: comprueba inexistencia -> upload
     * A: INSERT gana
     * B: INSERT choca con UNIQUE(idempotency_key)
     *
     * Después del choque volvemos a consultar. Si ya existe, borramos
     * el upload duplicado y devolvemos la entrada ganadora.
     */
    const winner = await findInferenceQueueEntryByKey(idempotencyKey);

    if (winner) {
      await deleteImage(uploaded.id).catch(() => undefined);

      return {
        imageId: winner.imageId,
        created: false,
        idempotencyKey,
      };
    }

    // Si no fue un conflicto idempotente real, limpiamos el upload para
    // no dejar basura en MinIO/MariaDB y propagamos el fallo original.
    await deleteImage(uploaded.id).catch(() => undefined);
    throw error;
  }
}
