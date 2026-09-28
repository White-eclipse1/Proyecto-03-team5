import { z } from "zod";

/**
 * APP-01: espeja `app/presentation/ml_contracts.py` (forma v1.0). Ambos lados
 * validan el mismo corpus de `app/presentation/examples/ml/` (ejemplos válidos e
 * `invalid_cases.json`), así que un cambio incompatible en un solo lado rompe los
 * tests del otro. Igual que en `pipeline/schemas.ts`, ningún componente confía en
 * el JSON sin validar y los tipos salen de `z.infer`.
 *
 * Los IDs son los reales de cada sistema, nunca etiquetas de UI:
 * dataset_version (release del dataset), manifest_hash (`md5:` DVC o `sha256:`),
 * experiment_id / run_id (MLflow), checkpoint (`runs:/<run_id>/<ruta>` del mismo
 * run) y model_version (versión del Model Registry de MLflow, entero en string).
 *
 * APP-02 agrega el request de entrenamiento (`trainingParamsSchema`, compartido
 * por el request y el job), la regla del Quality Gate (`trainingBlockedReason`) y
 * los archivos por release `provenance.json` y `manifest.json`. Los mensajes de
 * error de los parámetros son los que ve el usuario en el formulario.
 */

const identifierSchema = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]*$/);
const manifestHashSchema = z.string().regex(/^(md5:[0-9a-f]{32}|sha256:[0-9a-f]{64})$/);
const runIdSchema = z.string().regex(/^[0-9a-f]{32}$/);
const experimentIdSchema = z.string().regex(/^[0-9]+$/);
const checkpointSchema = z.string().regex(/^runs:\/[0-9a-f]{32}\/[^\s]+$/);
const modelVersionSchema = z.string().regex(/^[1-9][0-9]*$/);
const errorCodeSchema = z.string().regex(/^[a-z][a-z0-9_]*$/);
const timestampSchema = z.string().regex(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$/);
const labelSchema = z.string().min(1);
const countSchema = z.number().int().min(0);
const ratioSchema = z.number().min(0).max(1);
const schemaVersionSchema = z.literal("1.0");

function checkpointRunId(checkpoint: string): string {
  return checkpoint.slice("runs:/".length).split("/", 1)[0] ?? "";
}

function hasDuplicates(values: readonly string[]): boolean {
  return new Set(values).size !== values.length;
}

export const contractErrorSchema = z.strictObject({
  code: errorCodeSchema,
  message: labelSchema,
  retryable: z.boolean(),
});
export type ContractError = z.infer<typeof contractErrorSchema>;

export const errorResponseSchema = z.strictObject({
  schema_version: schemaVersionSchema,
  error: contractErrorSchema,
});

/** Número con mensaje propio: cubre vacío (undefined), texto (NaN) y tipo incorrecto. */
function numberField(name: string, integer: boolean) {
  const base = z.number({
    error: integer ? `${name} debe ser un número entero.` : `${name} debe ser un número.`,
  });
  return integer ? base.int(`${name} debe ser un número entero.`) : base;
}

export const TRAINING_PARAM_NAMES = [
  "optimizer",
  "batch_size",
  "max_epochs",
  "learning_rate",
  "image_size",
  "hidden_layers",
  "dropout",
  "seed",
  "patience",
  "min_delta",
] as const;
export type TrainingParamName = (typeof TRAINING_PARAM_NAMES)[number];

export const trainingParamsSchema = z
  .strictObject({
    optimizer: z.enum(["adam", "adamw", "sgd"], {
      error: "optimizer debe ser adam, adamw o sgd.",
    }),
    batch_size: numberField("batch_size", true)
      .min(1, "batch_size debe ser al menos 1.")
      .max(256, "batch_size no puede ser mayor que 256."),
    max_epochs: numberField("max_epochs", true)
      .min(1, "max_epochs debe ser al menos 1.")
      .max(500, "max_epochs no puede ser mayor que 500."),
    learning_rate: numberField("learning_rate", false)
      .gt(0, "learning_rate debe ser mayor que 0.")
      .max(1, "learning_rate no puede ser mayor que 1."),
    image_size: numberField("image_size", true)
      .min(32, "image_size debe ser al menos 32.")
      .max(1024, "image_size no puede ser mayor que 1024.")
      .multipleOf(32, "image_size debe ser múltiplo de 32."),
    hidden_layers: z
      .array(
        numberField("hidden_layers", true)
          .min(1, "Cada capa de hidden_layers necesita al menos 1 neurona.")
          .max(4096, "Cada capa de hidden_layers admite como máximo 4096 neuronas."),
        { error: "hidden_layers debe ser una lista de enteros separados por comas." }
      )
      .max(5, "hidden_layers admite como máximo 5 capas."),
    dropout: numberField("dropout", false)
      .min(0, "dropout no puede ser negativo.")
      .lt(1, "dropout debe ser menor que 1."),
    seed: numberField("seed", true)
      .min(0, "seed no puede ser negativo.")
      .max(2 ** 31 - 1, "seed no puede ser mayor que 2147483647."),
    patience: numberField("patience", true).min(1, "patience debe ser al menos 1."),
    min_delta: numberField("min_delta", false)
      .min(0, "min_delta no puede ser negativo.")
      .max(1, "min_delta no puede ser mayor que 1."),
  })
  .superRefine((params, context) => {
    if (params.patience > params.max_epochs) {
      context.addIssue({
        code: "custom",
        message: "patience no puede ser mayor que max_epochs.",
        path: ["patience"],
      });
    }
  });
export type TrainingParams = z.infer<typeof trainingParamsSchema>;

/** Body de `POST /api/ml/training/jobs`, validado antes de que exista cualquier job. */
export const trainingJobRequestSchema = z.strictObject({
  schema_version: schemaVersionSchema,
  dataset_version: identifierSchema,
  params: trainingParamsSchema,
});
export type TrainingJobRequest = z.infer<typeof trainingJobRequestSchema>;

/** Regla del Quality Gate, igual que `training_blocked_reason` en Python. */
export function trainingBlockedReason(qualityStatus: string): string | null {
  return qualityStatus === "failed"
    ? "El release no pasó el Quality Gate (failed); no se puede entrenar con él."
    : null;
}

export const trainingStatusSchema = z.enum([
  "queued",
  "running",
  "succeeded",
  "failed",
  "cancelled",
]);

export const trainingJobSchema = z
  .strictObject({
    job_id: identifierSchema,
    status: trainingStatusSchema,
    dataset_version: identifierSchema,
    manifest_hash: manifestHashSchema,
    experiment_id: experimentIdSchema,
    run_id: runIdSchema.nullable(),
    checkpoint: checkpointSchema.nullable(),
    params: trainingParamsSchema,
    created_at: timestampSchema,
    started_at: timestampSchema.nullable(),
    finished_at: timestampSchema.nullable(),
    error: contractErrorSchema.nullable(),
  })
  .superRefine((job, context) => {
    const issue = (message: string) => context.addIssue({ code: "custom", message });
    const terminal = ["succeeded", "failed", "cancelled"].includes(job.status);
    if (job.status === "queued" && (job.run_id !== null || job.started_at !== null)) {
      issue("Un job queued todavía no tiene run_id ni started_at");
    }
    if (
      (job.status === "running" || job.status === "succeeded") &&
      (job.run_id === null || job.started_at === null)
    ) {
      issue("Un job running o succeeded requiere run_id y started_at");
    }
    if ((job.finished_at !== null) !== terminal) {
      issue("finished_at existe exactamente cuando el job terminó");
    }
    if ((job.checkpoint !== null) !== (job.status === "succeeded")) {
      issue("checkpoint existe exactamente cuando el job terminó con éxito");
    }
    if ((job.error !== null) !== (job.status === "failed")) {
      issue("error existe exactamente cuando el job falló");
    }
    if (job.checkpoint !== null && checkpointRunId(job.checkpoint) !== job.run_id) {
      issue("El checkpoint debe pertenecer al mismo run_id");
    }
  });
export type TrainingJob = z.infer<typeof trainingJobSchema>;

export const trainingJobsResponseSchema = z
  .strictObject({ schema_version: schemaVersionSchema, jobs: z.array(trainingJobSchema) })
  .refine(
    (response) => !hasDuplicates(response.jobs.map((job) => job.job_id)),
    "job_id debe ser único"
  );
export type TrainingJobsResponse = z.infer<typeof trainingJobsResponseSchema>;

export const experimentRunSchema = z
  .strictObject({
    run_id: runIdSchema,
    experiment_id: experimentIdSchema,
    run_name: labelSchema,
    status: z.enum(["SCHEDULED", "RUNNING", "FINISHED", "FAILED", "KILLED"]),
    start_time: timestampSchema,
    end_time: timestampSchema.nullable(),
    dataset_version: identifierSchema,
    manifest_hash: manifestHashSchema,
    params: z.record(z.string(), z.string()),
    metrics: z.record(z.string(), z.number()),
  })
  .refine(
    (run) => (run.end_time === null) === (run.status === "SCHEDULED" || run.status === "RUNNING"),
    "end_time es null exactamente mientras el run está activo"
  );
export type ExperimentRun = z.infer<typeof experimentRunSchema>;

export const runsResponseSchema = z
  .strictObject({ schema_version: schemaVersionSchema, runs: z.array(experimentRunSchema) })
  .refine(
    (response) => !hasDuplicates(response.runs.map((run) => run.run_id)),
    "run_id debe ser único"
  );
export type RunsResponse = z.infer<typeof runsResponseSchema>;

export const evaluationSchema = z
  .strictObject({
    evaluation_id: identifierSchema,
    run_id: runIdSchema,
    checkpoint: checkpointSchema,
    model_name: identifierSchema.nullable(),
    model_version: modelVersionSchema.nullable(),
    dataset_version: identifierSchema,
    manifest_hash: manifestHashSchema,
    split: z.enum(["validation", "test"]),
    metrics: z.strictObject({
      map50: ratioSchema,
      map50_95: ratioSchema,
      precision: ratioSchema,
      recall: ratioSchema,
    }),
    per_class: z.array(
      z.strictObject({ category_name: labelSchema, ap50: ratioSchema, support: countSchema })
    ),
    created_at: timestampSchema,
  })
  .superRefine((evaluation, context) => {
    const issue = (message: string) => context.addIssue({ code: "custom", message });
    if ((evaluation.model_name === null) !== (evaluation.model_version === null)) {
      issue("model_name y model_version van juntos");
    }
    if (checkpointRunId(evaluation.checkpoint) !== evaluation.run_id) {
      issue("El checkpoint debe pertenecer al mismo run_id");
    }
    if (hasDuplicates(evaluation.per_class.map((entry) => entry.category_name))) {
      issue("category_name debe ser único");
    }
  });
export type Evaluation = z.infer<typeof evaluationSchema>;

export const evaluationsResponseSchema = z
  .strictObject({ schema_version: schemaVersionSchema, evaluations: z.array(evaluationSchema) })
  .refine(
    (response) =>
      !hasDuplicates(response.evaluations.map((evaluation) => evaluation.evaluation_id)),
    "evaluation_id debe ser único"
  );
export type EvaluationsResponse = z.infer<typeof evaluationsResponseSchema>;

export const registeredModelVersionSchema = z
  .strictObject({
    model_name: identifierSchema,
    model_version: modelVersionSchema,
    status: z.enum(["PENDING_REGISTRATION", "READY", "FAILED_REGISTRATION"]),
    aliases: z.array(identifierSchema),
    run_id: runIdSchema,
    checkpoint: checkpointSchema,
    dataset_version: identifierSchema,
    manifest_hash: manifestHashSchema,
    created_at: timestampSchema,
  })
  .superRefine((model, context) => {
    if (checkpointRunId(model.checkpoint) !== model.run_id) {
      context.addIssue({ code: "custom", message: "El checkpoint debe pertenecer al run_id" });
    }
    if (hasDuplicates(model.aliases)) {
      context.addIssue({ code: "custom", message: "aliases debe ser único" });
    }
  });
export type RegisteredModelVersion = z.infer<typeof registeredModelVersionSchema>;

export const modelsResponseSchema = z
  .strictObject({
    schema_version: schemaVersionSchema,
    models: z.array(registeredModelVersionSchema),
  })
  .superRefine((response, context) => {
    const versions = response.models.map(
      (model) => `${model.model_name}\u0000${model.model_version}`
    );
    if (hasDuplicates(versions)) {
      context.addIssue({ code: "custom", message: "(model_name, model_version) debe ser único" });
    }
    const aliases = response.models.flatMap((model) =>
      model.aliases.map((alias) => `${model.model_name}\u0000${alias}`)
    );
    if (hasDuplicates(aliases)) {
      context.addIssue({ code: "custom", message: "Un alias apunta a una sola versión" });
    }
  });
export type ModelsResponse = z.infer<typeof modelsResponseSchema>;

const nonNegativeSchema = z.number().min(0);

export const predictionSchema = z
  .strictObject({
    category_name: labelSchema,
    score: ratioSchema,
    /** Formato COCO: [x, y, ancho, alto] en píxeles. */
    bbox: z.array(nonNegativeSchema).length(4),
  })
  .refine(
    (prediction) => (prediction.bbox[2] ?? 0) > 0 && (prediction.bbox[3] ?? 0) > 0,
    "El ancho y alto de la bbox deben ser positivos"
  );
export type Prediction = z.infer<typeof predictionSchema>;

export const inferenceResponseSchema = z
  .strictObject({
    schema_version: schemaVersionSchema,
    request_id: identifierSchema,
    model_name: identifierSchema,
    model_version: modelVersionSchema,
    run_id: runIdSchema,
    dataset_version: identifierSchema,
    image: z.strictObject({
      width: z.number().int().positive(),
      height: z.number().int().positive(),
    }),
    predictions: z.array(predictionSchema),
    latency_ms: nonNegativeSchema,
  })
  .refine(
    (response) =>
      response.predictions.every(
        ({ bbox: [x = 0, y = 0, width = 0, height = 0] }) =>
          x + width <= response.image.width && y + height <= response.image.height
      ),
    "La bbox debe caber en la imagen"
  );
export type InferenceResponse = z.infer<typeof inferenceResponseSchema>;

export const releaseProvenanceSchema = z
  .strictObject({
    schema_version: schemaVersionSchema,
    dataset_version: identifierSchema,
    dvc_outputs: z
      .array(
        z.strictObject({
          path: labelSchema,
          md5: z.string().regex(/^[0-9a-f]{32}(.dir)?$/),
          nfiles: countSchema.nullable(),
        })
      )
      .min(1),
  })
  .refine(
    (provenance) => !hasDuplicates(provenance.dvc_outputs.map((output) => output.path)),
    "path debe ser único"
  );
export type ReleaseProvenance = z.infer<typeof releaseProvenanceSchema>;

const manifestSplitSchema = z.strictObject({ image_count: countSchema, ratio: ratioSchema });

export const trainingManifestSchema = z
  .strictObject({
    schema_version: schemaVersionSchema,
    dataset_version: identifierSchema,
    manifest_hash: manifestHashSchema,
    total_images: z.number().int().positive(),
    splits: z.strictObject({
      train: manifestSplitSchema,
      validation: manifestSplitSchema,
      test: manifestSplitSchema,
    }),
  })
  .superRefine((manifest, context) => {
    const splits = Object.values(manifest.splits);
    if (splits.reduce((total, split) => total + split.image_count, 0) !== manifest.total_images) {
      context.addIssue({ code: "custom", message: "Los conteos deben sumar total_images" });
    }
    // Python: math.isclose(abs_tol=1e-6).
    if (
      splits.some(
        (split) => Math.abs(split.ratio - split.image_count / manifest.total_images) > 1e-6
      )
    ) {
      context.addIssue({ code: "custom", message: "ratio = image_count / total_images" });
    }
  });
export type TrainingManifest = z.infer<typeof trainingManifestSchema>;

/** Mismo mapa que `CONTRACTS` en ml_contracts.py; los tests exigen que coincidan. */
export const ML_CONTRACTS = {
  training_jobs: trainingJobsResponseSchema,
  runs: runsResponseSchema,
  evaluations: evaluationsResponseSchema,
  models: modelsResponseSchema,
  inference: inferenceResponseSchema,
  error: errorResponseSchema,
  training_request: trainingJobRequestSchema,
  provenance: releaseProvenanceSchema,
  manifest: trainingManifestSchema,
} as const;
