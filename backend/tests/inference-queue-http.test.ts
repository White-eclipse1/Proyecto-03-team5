import type { Server } from 'node:http';

import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { z } from 'zod';

const SIX_MIB = 6 * 1024 * 1024;

const enqueueInferenceResultMock = vi.fn(async () => ({
  imageId: 321,
  created: true,
  idempotencyKey: 'a'.repeat(64),
}));

let server: Server | undefined;
let baseUrl: string;

beforeAll(async () => {
  process.env.NODE_ENV = 'test';
  process.env.DATABASE_URL = 'mysql://root:ci@127.0.0.1:3306/image_repo';
  process.env.MINIO_ENDPOINT = '127.0.0.1';
  process.env.MINIO_ACCESS_KEY = 'ci';
  process.env.MINIO_SECRET_KEY = 'ci';
  process.env.MINIO_BUCKET = 'image-annotations';
  process.env.MAX_UPLOAD_SIZE_BYTES = String(10 * 1024 * 1024);

  vi.doMock('../src/logic/index.js', () => {
    class MockNotFoundError extends Error {}
    class MockValidationError extends Error {}

    return {
      checkHealth: vi.fn(),
      createAnnotationForImage: vi.fn(),
      createSettingsService: vi.fn(() => ({
        get: vi.fn(),
        saveQuality: vi.fn(),
        saveSplits: vi.fn(),
      })),
      deleteAnnotation: vi.fn(),
      deleteImage: vi.fn(),
      enqueueInferenceResult: enqueueInferenceResultMock,
      exportCocoDataset: vi.fn(),
      getAnnotationsForImage: vi.fn(),
      getCategories: vi.fn(),
      getDashboardSummary: vi.fn(),
      getImageFile: vi.fn(),
      idParamSchema: z.coerce.number().int().positive(),
      imageSearchSchema: z.object({}).passthrough(),
      initializeApplication: vi.fn(),
      NotFoundError: MockNotFoundError,
      searchImages: vi.fn(),
      setImageStatus: vi.fn(),
      updateAnnotation: vi.fn(),
      uploadImage: vi.fn(),
      ValidationError: MockValidationError,
    };
  });

  const { app } = await import('../src/ui/server.js');

  await new Promise<void>((resolve) => {
    server = app.listen(0, '127.0.0.1', resolve);
  });

  const address = server.address();

  if (address === null || typeof address === 'string') {
    throw new Error('No se pudo obtener el puerto HTTP del test.');
  }

  baseUrl = `http://127.0.0.1:${address.port}`;
});

afterAll(async () => {
  vi.doUnmock('../src/logic/index.js');

  if (server === undefined) return;

  await new Promise<void>((resolve, reject) => {
    server!.close((error) => {
      if (error) reject(error);
      else resolve();
    });
  });
});

describe('APP-08 - endpoint inference → annotation queue', () => {
  it('acepta por HTTP una imagen de 6 MiB y la entrega a la cola', async () => {
    const form = new FormData();

    form.append(
      'image',
      new Blob([new Uint8Array(SIX_MIB)], {
        type: 'image/jpeg',
      }),
      'six-mib.jpg',
    );

    form.append(
      'metadata',
      JSON.stringify({
        sourceKind: 'upload',
        sourceRef: `sha256:${'b'.repeat(64)}`,
        modelName: 'dog-cat-resnet18',
        modelVersion: '1.0.0',
        runId: '0a1b2c3d4e5f60718293a4b5c6d7e8f9',
        checkpointSha256: '84d6c88b4bd84c3e6f0229dd23f7f188ff85622bbb39e5c3988a97598fa90d52',
        predictedClass: 'cat',
        probabilities: {
          dog: 0.03,
          cat: 0.97,
        },
      }),
    );

    const response = await fetch(`${baseUrl}/images/from-inference`, {
      method: 'POST',
      body: form,
    });

    expect(response.status).toBe(201);
    expect(enqueueInferenceResultMock).toHaveBeenCalledTimes(1);

    const input = enqueueInferenceResultMock.mock.calls[0]?.[0];

    expect(input?.image.sizeBytes).toBe(SIX_MIB);
    expect(input?.image.filename).toBe('six-mib.jpg');
    expect(input?.image.mimeType).toBe('image/jpeg');
  });
});
