import { createHash } from 'node:crypto';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { TraceabilityUnavailableError, ValidationError } from '../src/logic/errors.js';
import type { InferenceQueueMetadata } from '../src/logic/inference-queue.validation.js';
import { verifyInferenceTraceability } from '../src/logic/inference-traceability.js';

/**
 * APP-09 (hallazgo en APP-08): la cola de anotación guardaba la trazabilidad que mandaba
 * el navegador sin compararla con lo real. Ahora se verifica contra el registro de
 * OPS-06 (`reports/models/registry.json`) y los recortes de ML-01 (`reports/crops.json`).
 */

const RUN = 'bb448230424146349a969253d30db43b';
const CHECKPOINT = '84d6c88b4bd84c3e6f0229dd23f7f188ff85622bbb39e5c3988a97598fa90d52';
const CROP = Buffer.from('png del recorte img369-ann355');
const UPLOAD = Buffer.from('imagen subida por el usuario');
const sha256 = (data: Buffer) => createHash('sha256').update(data).digest('hex');

let dir: string;
let sources: { registryPath: string; cropsReportPath: string };

const crop: InferenceQueueMetadata = {
  sourceKind: 'crop',
  sourceRef: 'v0.1.1:img369-ann355',
  modelName: 'dog-cat-resnet18',
  modelVersion: '1.0.0',
  runId: RUN,
  checkpointSha256: CHECKPOINT,
  predictedClass: 'cat',
  probabilities: { dog: 0.3679, cat: 0.6321 },
};

const upload: InferenceQueueMetadata = {
  ...crop,
  sourceKind: 'upload',
  sourceRef: `sha256:${sha256(UPLOAD)}`,
};

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'app09-'));
  sources = {
    registryPath: join(dir, 'registry.json'),
    cropsReportPath: join(dir, 'crops.json'),
  };
  writeFileSync(
    sources.registryPath,
    JSON.stringify({
      schema_version: '1.0',
      models: [
        {
          model_name: 'dog-cat-resnet18',
          model_version: '1.0.0',
          run_id: RUN,
          checkpoint_sha256: CHECKPOINT,
        },
      ],
    }),
  );
  writeFileSync(
    sources.cropsReportPath,
    JSON.stringify({
      dataset_version: 'v0.1.1',
      crops: [{ crop_id: 'img369-ann355', sha256: sha256(CROP) }],
    }),
  );
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

describe('APP-09: trazabilidad real de lo que se envía a la cola de anotación', () => {
  it('acepta un recorte y una imagen subida con su trazabilidad real', () => {
    expect(() => verifyInferenceTraceability(crop, CROP, sources)).not.toThrow();
    expect(() => verifyInferenceTraceability(upload, UPLOAD, sources)).not.toThrow();
  });

  it.each([
    ['otra versión', { modelVersion: '9.9.9' }],
    ['otro modelo', { modelName: 'otro-modelo' }],
    ['otro run', { runId: 'f'.repeat(32) }],
    ['checkpoint falso', { checkpointSha256: '0'.repeat(64) }],
  ])('rechaza %s que no está en el registro de modelos', (_label, changes) => {
    expect(() => verifyInferenceTraceability({ ...crop, ...changes }, CROP, sources)).toThrow(
      ValidationError,
    );
  });

  it('rechaza una imagen subida que no es la del sha256 declarado', () => {
    expect(() => verifyInferenceTraceability(upload, Buffer.from('otra imagen'), sources)).toThrow(
      /sha256/,
    );
  });

  it.each([
    ['sin el prefijo sha256:', { sourceRef: sha256(UPLOAD) }],
    ['con un hash que no es hex', { sourceRef: 'sha256:xyz' }],
  ])('rechaza un upload %s', (_label, changes) => {
    expect(() => verifyInferenceTraceability({ ...upload, ...changes }, UPLOAD, sources)).toThrow(
      ValidationError,
    );
  });

  it.each([
    ['un recorte que no existe', { sourceRef: 'v0.1.1:img1-ann1' }],
    ['un recorte de otro release', { sourceRef: 'v0.1.0:img369-ann355' }],
    ['una referencia mal formada', { sourceRef: 'img369-ann355' }],
  ])('rechaza %s', (_label, changes) => {
    expect(() => verifyInferenceTraceability({ ...crop, ...changes }, CROP, sources)).toThrow(
      ValidationError,
    );
  });

  it('rechaza un recorte cuya imagen no es la registrada en crops.json', () => {
    expect(() => verifyInferenceTraceability(crop, Buffer.from('otro png'), sources)).toThrow(
      /sha256/,
    );
  });

  it('sin el registro de modelos no se puede verificar: el servicio no está disponible', () => {
    rmSync(sources.registryPath);

    expect(() => verifyInferenceTraceability(crop, CROP, sources)).toThrow(
      TraceabilityUnavailableError,
    );
  });
});
