import { createHash } from 'node:crypto';

import { z } from 'zod';

const probabilitiesSchema = z
  .object({
    dog: z.number().min(0).max(1),
    cat: z.number().min(0).max(1),
  })
  .refine(({ dog, cat }) => Math.abs(dog + cat - 1) < 1e-6, {
    message: 'Las probabilidades deben sumar aproximadamente 1.',
  });

export const inferenceQueueMetadataSchema = z
  .object({
    sourceKind: z.enum(['upload', 'crop']),
    sourceRef: z.string().trim().min(1),
    modelName: z.string().trim().min(1),
    modelVersion: z
      .string()
      .trim()
      .regex(/^\d+\.\d+\.\d+$/, 'modelVersion debe usar SemVer.'),
    runId: z.string().trim().min(1),
    checkpointSha256: z.string().regex(/^[a-f0-9]{64}$/i),
    predictedClass: z.enum(['dog', 'cat']),
    probabilities: probabilitiesSchema,
  })
  // APP-09: la clase enviada es la de mayor probabilidad, como la devuelve Inference.
  .refine(
    ({ predictedClass, probabilities }) =>
      probabilities[predictedClass] === Math.max(probabilities.dog, probabilities.cat),
    {
      message: 'predictedClass debe ser la clase de mayor probabilidad.',
      path: ['predictedClass'],
    },
  );

export type InferenceQueueMetadata = z.infer<typeof inferenceQueueMetadataSchema>;

/**
 * Clave idempotente de una inferencia enviada a la cola.
 *
 * Dos clicks/retries del mismo resultado generan exactamente la misma clave.
 * Cambiar imagen/crop, modelo o resultado genera una clave diferente.
 */
export function buildInferenceQueueKey(input: unknown): string {
  const metadata = inferenceQueueMetadataSchema.parse(input);

  const canonical = JSON.stringify({
    sourceKind: metadata.sourceKind,
    sourceRef: metadata.sourceRef,
    modelName: metadata.modelName,
    modelVersion: metadata.modelVersion,
    runId: metadata.runId,
    checkpointSha256: metadata.checkpointSha256,
    predictedClass: metadata.predictedClass,
    probabilities: metadata.probabilities,
  });

  return createHash('sha256').update(canonical).digest('hex');
}
