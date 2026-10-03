import { beforeAll, describe, expect, it, vi } from 'vitest';

/**
 * APP-09: metadatos inválidos en la cola de anotación son un error del cliente (400),
 * no un 500. Antes `schema.parse` lanzaba un ZodError que la capa UI no mapeaba.
 */

const uploadImage = vi.fn();
let enqueueInferenceResult: typeof import('../src/logic/inference-queue.service.js').enqueueInferenceResult;
let ValidationError: typeof import('../src/logic/errors.js').ValidationError;

beforeAll(async () => {
  process.env.DATABASE_URL = 'mysql://root:ci@127.0.0.1:3306/image_repo';
  process.env.MINIO_ENDPOINT = '127.0.0.1';
  process.env.MINIO_ACCESS_KEY = 'ci';
  process.env.MINIO_SECRET_KEY = 'ci';
  process.env.MINIO_BUCKET = 'image-annotations';
  vi.doMock('../src/data/index.js', () => ({
    createInferenceQueueEntry: vi.fn(),
    findImageById: vi.fn(),
    findInferenceQueueEntryByKey: vi.fn(),
  }));
  vi.doMock('../src/logic/image-upload.service.js', () => ({
    deleteImage: vi.fn(),
    uploadImage,
  }));
  ({ enqueueInferenceResult } = await import('../src/logic/inference-queue.service.js'));
  ({ ValidationError } = await import('../src/logic/errors.js'));
});

const image = {
  filename: 'x.png',
  mimeType: 'image/png',
  sizeBytes: 3,
  buffer: Buffer.from('png'),
};
const metadata = {
  sourceKind: 'crop',
  sourceRef: 'v0.1.1:img369-ann355',
  modelName: 'dog-cat-resnet18',
  modelVersion: '1.0.0',
  runId: 'bb448230424146349a969253d30db43b',
  checkpointSha256: '84d6c88b4bd84c3e6f0229dd23f7f188ff85622bbb39e5c3988a97598fa90d52',
  predictedClass: 'cat',
  probabilities: { dog: 0.3679, cat: 0.6321 },
};

describe('APP-09: metadatos inválidos de la cola de anotación', () => {
  it.each([
    ['una clase que no es la más probable', { predictedClass: 'dog' }],
    ['probabilidades que no suman 1', { probabilities: { dog: 0.9, cat: 0.9 } }],
    ['una versión que no es SemVer', { modelVersion: '3' }],
  ])('rechaza %s como ValidationError, sin subir nada', async (_label, changes) => {
    await expect(
      enqueueInferenceResult({ image, metadata: { ...metadata, ...changes } as never }),
    ).rejects.toBeInstanceOf(ValidationError);
    expect(uploadImage).not.toHaveBeenCalled();
  });
});
