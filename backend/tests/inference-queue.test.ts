import { describe, expect, it } from 'vitest';

import {
  buildInferenceQueueKey,
  inferenceQueueMetadataSchema,
} from '../src/logic/inference-queue.validation.js';

const base = {
  sourceKind: 'crop' as const,
  sourceRef: 'v0.1.1:img42-ann1007',
  modelName: 'dog-cat-resnet18',
  modelVersion: '1.0.0',
  runId: 'bb448230424146349a969253d30db43b',
  checkpointSha256: '84d6c88b4bd84c3e6f0229dd23f7f188ff85622bbb39e5c3988a97598fa90d52',
  predictedClass: 'cat',
  probabilities: {
    dog: 0.03,
    cat: 0.97,
  },
};

describe('APP-08 - inference → annotation queue', () => {
  it('acepta la trazabilidad completa de una inferencia', () => {
    expect(inferenceQueueMetadataSchema.parse(base)).toEqual(base);
  });

  it('conserva model version y predicción', () => {
    const parsed = inferenceQueueMetadataSchema.parse(base);

    expect(parsed.modelVersion).toBe('1.0.0');
    expect(parsed.predictedClass).toBe('cat');
    expect(parsed.probabilities).toEqual({
      dog: 0.03,
      cat: 0.97,
    });
  });

  it('rechaza una predicción sin referencia de origen', () => {
    expect(() =>
      inferenceQueueMetadataSchema.parse({
        ...base,
        sourceRef: '',
      }),
    ).toThrow();
  });

  it('rechaza probabilidades incoherentes', () => {
    expect(() =>
      inferenceQueueMetadataSchema.parse({
        ...base,
        probabilities: {
          dog: 0.9,
          cat: 0.9,
        },
      }),
    ).toThrow();
  });

  it('rechaza una clase que no es la más probable (hallazgo de APP-09)', () => {
    expect(() =>
      inferenceQueueMetadataSchema.parse({
        ...base,
        predictedClass: 'dog',
      }),
    ).toThrow();
  });

  it('genera la misma clave para un retry o doble click', () => {
    const first = buildInferenceQueueKey(base);
    const retry = buildInferenceQueueKey({ ...base });

    expect(first).toBe(retry);
    expect(first).toHaveLength(64);
  });

  it('otra model version produce otra entrada idempotente', () => {
    expect(buildInferenceQueueKey(base)).not.toBe(
      buildInferenceQueueKey({
        ...base,
        modelVersion: '1.1.0',
      }),
    );
  });

  it('otra referencia de imagen produce otra entrada', () => {
    expect(buildInferenceQueueKey(base)).not.toBe(
      buildInferenceQueueKey({
        ...base,
        sourceRef: 'v0.1.1:img43-ann1008',
      }),
    );
  });
});
