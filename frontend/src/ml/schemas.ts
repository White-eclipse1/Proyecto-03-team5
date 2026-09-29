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
 * Un recorte se identifica por sus IDs COCO image_id / annotation_id (ML-01).
 *
 * El modelo es un clasificador multiclase de recortes: la evaluación trae
 * accuracy_top1, f1_macro, métricas por clase, matriz de confusión y predicciones
 * por recorte, y todo debe cuadrar con la matriz; la inferencia devuelve una
 * clase con su distribución de probabilidades, nunca bounding boxes.
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
const cocoIdSchema = z.number().int().min(0);
const probabilitiesSchema = z.record(labelSchema, ratioSchema);

/** Igual que METRIC_TOLERANCE en Python: métricas redondeadas a 3 decimales. */
export const METRIC_TOLERANCE = 1e-3;

function checkpointRunId(checkpoint: string): string {
  return checkpoint.slice("runs:/".length).split("/", 1)[0] ?? "";
}

function hasDuplicates(values: readonly (string | number)[]): boolean {
  return new Set(values).size !== values.length;
}

const isClose = (reported: number, expected: number) =>
  Math.abs(reported - expected) <= METRIC_TOLERANCE;
const ratio = (numerator: number, denominator: number) =>
  denominator === 0 ? 0 : numerator / denominator;
const f1Score = (precision: number, recall: number) =>
  precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
const sum = (values: readonly number[]) => values.reduce((total, value) => total + value, 0);

/** Salida softmax: una probabilidad por clase, suma ~1 y argmax = predicted_class. */
function distributionIssue(
  probabilities: Record<string, number>,
  predictedClass: string,
  classNames: readonly string[] | null
): string | null {
  const names = Object.keys(probabilities);
  const values = Object.values(probabilities);
  if (
    classNames !== null &&
    (names.length !== classNames.length || !classNames.every((name) => name in probabilities))
  ) {
    return "probabilities debe tener exactamente una entrada por clase";
  }
  if (names.length < 2) return "probabilities necesita al menos dos clases";
  if (!isClose(sum(values), 1)) return "Las probabilidades deben sumar 1";
  if (probabilities[predictedClass] !== Math.max(...values)) {
    return "predicted_class debe ser la clase de mayor probabilidad";
  }
  return null;
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

/** Los 7 hiperparámetros obligatorios de un job; sin defaults. */
export const trainingParamsSchema = z.strictObject({
  optimizer: z.enum(["adam", "adamw", "sgd"]),
  batch_size: z.number().int().min(1).max(256),
  max_epochs: z.number().int().min(1).max(500),
  learning_rate: z.number().gt(0).max(1),
  image_size: z.number().int().min(32).max(1024).multipleOf(32),
  hidden_layers: z.array(z.number().int().min(1).max(4096)).max(5),
  dropout: z.number().min(0).lt(1),
});
export type TrainingParams = z.infer<typeof trainingParamsSchema>;

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
    /** Orden de per_class y de ambos ejes de la matriz (filas: real, columnas: predicha). */
    class_names: z.array(labelSchema).min(2),
    metrics: z.strictObject({ accuracy_top1: ratioSchema, f1_macro: ratioSchema }),
    per_class: z.array(
      z.strictObject({
        class_name: labelSchema,
        precision: ratioSchema,
        recall: ratioSchema,
        f1: ratioSchema,
        support: countSchema,
      })
    ),
    confusion_matrix: z.array(z.array(countSchema)),
    /** Predicción por recorte del split, para poder recalcular las métricas. */
    predictions: z.array(
      z.strictObject({
        image_id: cocoIdSchema,
        annotation_id: cocoIdSchema,
        true_class: labelSchema,
        predicted_class: labelSchema,
        probabilities: probabilitiesSchema,
      })
    ),
    created_at: timestampSchema,
  })
  .superRefine((evaluation, context) => {
    const issue = (message: string) => context.addIssue({ code: "custom", message });
    const names = evaluation.class_names;
    const matrix = evaluation.confusion_matrix;
    if ((evaluation.model_name === null) !== (evaluation.model_version === null)) {
      issue("model_name y model_version van juntos");
    }
    if (checkpointRunId(evaluation.checkpoint) !== evaluation.run_id) {
      issue("El checkpoint debe pertenecer al mismo run_id");
    }
    if (hasDuplicates(names)) {
      return issue("class_names debe ser único");
    }
    const perClassNames = evaluation.per_class.map((entry) => entry.class_name);
    if (perClassNames.join("\u0000") !== names.join("\u0000")) {
      return issue("per_class sigue class_names, una entrada por clase");
    }
    if (matrix.length !== names.length || matrix.some((row) => row.length !== names.length)) {
      return issue("confusion_matrix debe ser cuadrada con una fila por clase");
    }
    // La matriz es la fuente de verdad: todas las métricas se recalculan desde ella.
    const cell = (row: number, column: number) => matrix[row]?.[column] ?? 0;
    const total = sum(matrix.map(sum));
    if (total === 0) return issue("confusion_matrix no puede estar vacía");
    const f1Scores = evaluation.per_class.map((entry, index) => {
      const hits = cell(index, index);
      const actual = sum(matrix[index] ?? []);
      const precision = ratio(hits, sum(names.map((_, row) => cell(row, index))));
      const recall = ratio(hits, actual);
      const f1 = f1Score(precision, recall);
      if (entry.support !== actual) issue("support debe ser la suma de la fila de su clase");
      if (!isClose(entry.precision, precision)) issue(`precision de ${entry.class_name} no cuadra`);
      if (!isClose(entry.recall, recall)) issue(`recall de ${entry.class_name} no cuadra`);
      if (!isClose(entry.f1, f1)) issue(`f1 de ${entry.class_name} no cuadra`);
      return f1;
    });
    const trace = sum(names.map((_, index) => cell(index, index)));
    if (!isClose(evaluation.metrics.accuracy_top1, trace / total)) {
      issue("accuracy_top1 debe ser traza / total de la matriz");
    }
    if (!isClose(evaluation.metrics.f1_macro, sum(f1Scores) / names.length)) {
      issue("f1_macro debe ser el promedio del f1 por clase");
    }

    const predictions = evaluation.predictions;
    if (hasDuplicates(predictions.map((prediction) => prediction.annotation_id))) {
      issue("annotation_id debe ser único");
    }
    const counts = new Map<string, number>();
    for (const prediction of predictions) {
      const row = names.indexOf(prediction.true_class);
      const column = names.indexOf(prediction.predicted_class);
      if (row < 0 || column < 0) {
        return issue("true_class y predicted_class deben estar en class_names");
      }
      const problem = distributionIssue(
        prediction.probabilities,
        prediction.predicted_class,
        names
      );
      if (problem !== null) return issue(problem);
      const key = `${row},${column}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    const mismatch = names.some((_, row) =>
      names.some((_, column) => (counts.get(`${row},${column}`) ?? 0) !== cell(row, column))
    );
    if (mismatch) {
      issue("Las predicciones deben sumar exactamente la matriz de confusión");
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

/** Recorte elegido en la UI: una anotación COCO de un release del dataset. */
export const cropSelectionSchema = z.strictObject({
  dataset_version: identifierSchema,
  image_id: cocoIdSchema,
  annotation_id: cocoIdSchema,
});
export type CropSelection = z.infer<typeof cropSelectionSchema>;

/** Body de `POST /api/ml/inference`: clasificar un recorte existente. */
export const inferenceRequestSchema = z.strictObject({
  schema_version: schemaVersionSchema,
  model_name: identifierSchema,
  model_version: modelVersionSchema,
  crop: cropSelectionSchema,
});
export type InferenceRequest = z.infer<typeof inferenceRequestSchema>;

/**
 * Una clase para el recorte más la distribución completa. `dataset_version` es el
 * release con el que se entrenó el modelo; `crop.dataset_version`, el del recorte.
 */
export const inferenceResponseSchema = z
  .strictObject({
    schema_version: schemaVersionSchema,
    request_id: identifierSchema,
    model_name: identifierSchema,
    model_version: modelVersionSchema,
    run_id: runIdSchema,
    dataset_version: identifierSchema,
    crop: cropSelectionSchema,
    predicted_class: labelSchema,
    probabilities: probabilitiesSchema,
    latency_ms: z.number().min(0),
  })
  .superRefine((response, context) => {
    const problem = distributionIssue(response.probabilities, response.predicted_class, null);
    if (problem !== null) context.addIssue({ code: "custom", message: problem });
  });
export type InferenceResponse = z.infer<typeof inferenceResponseSchema>;

/** Mismo mapa que `CONTRACTS` en ml_contracts.py; los tests exigen que coincidan. */
export const ML_CONTRACTS = {
  training_jobs: trainingJobsResponseSchema,
  runs: runsResponseSchema,
  evaluations: evaluationsResponseSchema,
  models: modelsResponseSchema,
  inference_request: inferenceRequestSchema,
  inference: inferenceResponseSchema,
  error: errorResponseSchema,
} as const;
