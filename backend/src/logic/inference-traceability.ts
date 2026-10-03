import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

import { TraceabilityUnavailableError, ValidationError } from './errors.js';
import type { InferenceQueueMetadata } from './inference-queue.validation.js';

/**
 * APP-09 (hallazgo en APP-08) — la trazabilidad que llega a la cola de anotación se
 * compara con lo real antes de guardarla:
 *
 * - modelo, versión, run y checkpoint: `reports/models/registry.json` (OPS-06);
 * - recorte: el PNG debe tener el sha256 registrado en `reports/crops.json` (ML-01)
 *   para ese release y `crop_id`;
 * - imagen subida: `sourceRef` es `sha256:<hex>` de los bytes recibidos.
 *
 * Así la cola no guarda una versión inexistente, un checkpoint falso ni una imagen
 * que no es la que se clasificó.
 */

export interface TraceabilitySources {
  registryPath: string;
  cropsReportPath: string;
}

interface RegistryEntry {
  model_name?: unknown;
  model_version?: unknown;
  run_id?: unknown;
  checkpoint_sha256?: unknown;
}

interface CropEntry {
  crop_id?: unknown;
  sha256?: unknown;
}

const CROP_REF = /^(?<release>[A-Za-z0-9][A-Za-z0-9._-]*):(?<cropId>img\d+-ann\d+)$/;
const UPLOAD_REF = /^sha256:(?<sha>[0-9a-f]{64})$/;

function sha256(data: Buffer): string {
  return createHash('sha256').update(data).digest('hex');
}

function readJson(path: string, label: string): Record<string, unknown> {
  try {
    const document: unknown = JSON.parse(readFileSync(path, 'utf-8'));
    if (typeof document !== 'object' || document === null) {
      throw new Error(`${label} no es un objeto`);
    }
    return document as Record<string, unknown>;
  } catch (error) {
    throw new TraceabilityUnavailableError(
      `No se pudo leer ${label} para verificar la trazabilidad (${String(error)}).`,
    );
  }
}

function verifyModel(metadata: InferenceQueueMetadata, registryPath: string): void {
  const registry = readJson(registryPath, 'reports/models/registry.json');
  const models = Array.isArray(registry.models) ? (registry.models as RegistryEntry[]) : [];
  const entry = models.find(
    (model) =>
      model.model_name === metadata.modelName && model.model_version === metadata.modelVersion,
  );
  if (entry === undefined) {
    throw new ValidationError(
      `${metadata.modelName} ${metadata.modelVersion} no está en el registro de modelos.`,
    );
  }
  if (entry.run_id !== metadata.runId) {
    throw new ValidationError(
      `El run ${metadata.runId} no es el de ${metadata.modelName} ${metadata.modelVersion}.`,
    );
  }
  if (entry.checkpoint_sha256 !== metadata.checkpointSha256) {
    throw new ValidationError(
      `El checkpoint ${metadata.checkpointSha256} no es el registrado para ${metadata.modelVersion}.`,
    );
  }
}

function verifyCrop(metadata: InferenceQueueMetadata, image: Buffer, cropsPath: string): void {
  const match = CROP_REF.exec(metadata.sourceRef);
  if (match?.groups === undefined) {
    throw new ValidationError('sourceRef de un recorte es <release>:img<id>-ann<id>.');
  }
  const { release, cropId } = match.groups;
  const report = readJson(cropsPath, 'reports/crops.json');
  const crops = Array.isArray(report.crops) ? (report.crops as CropEntry[]) : [];
  const entry =
    report.dataset_version === release ? crops.find((c) => c.crop_id === cropId) : undefined;
  if (entry === undefined) {
    throw new ValidationError(`No existe el recorte ${cropId} en el release ${release}.`);
  }
  if (entry.sha256 !== sha256(image)) {
    throw new ValidationError(
      `La imagen enviada no es el recorte ${cropId}: su sha256 no es el registrado.`,
    );
  }
}

function verifyUpload(metadata: InferenceQueueMetadata, image: Buffer): void {
  const match = UPLOAD_REF.exec(metadata.sourceRef);
  if (match?.groups === undefined) {
    throw new ValidationError('sourceRef de una imagen subida es sha256:<64 hex>.');
  }
  if (match.groups.sha !== sha256(image)) {
    throw new ValidationError('La imagen enviada no es la clasificada: su sha256 es otro.');
  }
}

export function verifyInferenceTraceability(
  metadata: InferenceQueueMetadata,
  image: Buffer,
  sources: TraceabilitySources,
): void {
  verifyModel(metadata, sources.registryPath);
  if (metadata.sourceKind === 'crop') {
    verifyCrop(metadata, image, sources.cropsReportPath);
  } else {
    verifyUpload(metadata, image);
  }
}
