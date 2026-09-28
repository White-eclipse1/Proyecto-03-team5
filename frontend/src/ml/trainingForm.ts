import type { z } from "zod";
import {
  TRAINING_PARAM_NAMES,
  type TrainingJobRequest,
  type TrainingParamName,
  trainingJobRequestSchema,
} from "./schemas";

/** Lo que el usuario escribió, tal cual: la conversión y validación ocurren al enviar. */
export type RawTrainingParams = Record<TrainingParamName, string>;

/** Valores iniciales del formulario (editables); no representan ningún job real. */
export const DEFAULT_TRAINING_PARAMS: RawTrainingParams = {
  optimizer: "adam",
  batch_size: "32",
  max_epochs: "50",
  learning_rate: "0.001",
  image_size: "224",
  hidden_layers: "512, 256",
  dropout: "0.2",
  seed: "42",
  patience: "5",
  min_delta: "0.001",
};

export const TRAINING_PARAM_HINTS: Record<TrainingParamName, string> = {
  optimizer: "adam, adamw o sgd.",
  batch_size: "Entero de 1 a 256.",
  max_epochs: "Entero de 1 a 500.",
  learning_rate: "Mayor que 0 y hasta 1 (p. ej. 0.001).",
  image_size: "Múltiplo de 32, de 32 a 1024 px.",
  hidden_layers: "Neuronas por capa separadas por comas (máx. 5 capas); vacío = sin capas.",
  dropout: "De 0 a menos de 1.",
  seed: "Entero de 0 a 2147483647.",
  patience: "Épocas sin mejora antes de parar; entre 1 y max_epochs.",
  min_delta: "Mejora mínima que cuenta como progreso, de 0 a 1.",
};

export type TrainingFormField = TrainingParamName | "release";
export type TrainingFormErrors = Partial<Record<TrainingFormField, string>>;

/** "" → undefined (obligatorio), texto → NaN (no numérico); ambos los rechaza el schema. */
function toNumber(raw: string): number | undefined {
  const value = raw.trim();
  return value === "" ? undefined : Number(value);
}

/** Una lista con partes no numéricas se deja como texto para que falle como lista. */
function toLayers(raw: string): number[] | string {
  if (raw.trim() === "") return [];
  const layers = raw.split(",").map(toNumber);
  return layers.every((layer): layer is number => layer !== undefined && !Number.isNaN(layer))
    ? layers
    : raw;
}

export function toTrainingRequest(datasetVersion: string, raw: RawTrainingParams): unknown {
  return {
    schema_version: "1.0",
    dataset_version: datasetVersion,
    params: {
      optimizer: raw.optimizer,
      batch_size: toNumber(raw.batch_size),
      max_epochs: toNumber(raw.max_epochs),
      learning_rate: toNumber(raw.learning_rate),
      image_size: toNumber(raw.image_size),
      hidden_layers: toLayers(raw.hidden_layers),
      dropout: toNumber(raw.dropout),
      seed: toNumber(raw.seed),
      patience: toNumber(raw.patience),
      min_delta: toNumber(raw.min_delta),
    },
  };
}

function isParamName(value: unknown): value is TrainingParamName {
  return TRAINING_PARAM_NAMES.includes(value as TrainingParamName);
}

/** Primer mensaje de cada campo; `dataset_version` se muestra junto al selector de release. */
export function issuesToErrors(issues: readonly z.core.$ZodIssue[]): TrainingFormErrors {
  const errors: TrainingFormErrors = {};
  for (const issue of issues) {
    const [root, field] = issue.path;
    const key = root === "params" && isParamName(field) ? field : "release";
    errors[key] ??= key === "release" ? "Selecciona un release aprobado." : issue.message;
  }
  return errors;
}

export type TrainingValidation =
  | { ok: true; request: TrainingJobRequest }
  | { ok: false; errors: TrainingFormErrors };

/** Validación completa antes de cualquier POST: mismo contrato que Python. */
export function validateTrainingForm(
  datasetVersion: string,
  raw: RawTrainingParams
): TrainingValidation {
  const parsed = trainingJobRequestSchema.safeParse(toTrainingRequest(datasetVersion, raw));
  if (parsed.success) return { ok: true, request: parsed.data };
  return { ok: false, errors: issuesToErrors(parsed.error.issues) };
}
