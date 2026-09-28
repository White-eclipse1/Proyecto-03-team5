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
    params: z.record(z.string(), z.union([z.string(), z.number(), z.boolean()])),
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

/** Mismo mapa que `CONTRACTS` en ml_contracts.py; los tests exigen que coincidan. */
export const ML_CONTRACTS = {
  training_jobs: trainingJobsResponseSchema,
  runs: runsResponseSchema,
  evaluations: evaluationsResponseSchema,
  models: modelsResponseSchema,
  inference: inferenceResponseSchema,
  error: errorResponseSchema,
} as const;
