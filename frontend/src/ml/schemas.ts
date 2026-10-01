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
 *
 * APP-02 agrega el request de entrenamiento (`trainingJobRequestSchema`, que
 * comparte `trainingParamsSchema` con el job), seed y early stopping, la regla del
 * Quality Gate (`trainingBlockedReason`) y los archivos por release
 * `provenance.json` y `manifest.json`. Los mensajes de error de los parámetros son
 * los que ve el usuario en el formulario.
 */

const identifierSchema = z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]*$/);
const manifestHashSchema = z.string().regex(/^(md5:[0-9a-f]{32}|sha256:[0-9a-f]{64})$/);
const runIdSchema = z.string().regex(/^[0-9a-f]{32}$/);
const gitCommitSchema = z.string().regex(/^[0-9a-f]{40}$/);
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

/**
 * Body de `POST /api/ml/training/jobs`, validado antes de que exista cualquier job.
 * `manifest_hash` fija el job al manifiesto 70/20/10 exacto del release.
 */
export const trainingJobRequestSchema = z.strictObject({
  schema_version: schemaVersionSchema,
  dataset_version: identifierSchema,
  manifest_hash: manifestHashSchema,
  params: trainingParamsSchema,
});
export type TrainingJobRequest = z.infer<typeof trainingJobRequestSchema>;

export const trainingStatusSchema = z.enum([
  "queued",
  "running",
  "succeeded",
  "failed",
  "cancelled",
]);

/** APP-03: última época reportada por el worker y sus métricas. */
export const trainingProgressSchema = z
  .strictObject({
    epoch: countSchema,
    max_epochs: z.number().int().min(1),
    metrics: z.record(labelSchema, z.number()),
    updated_at: timestampSchema,
  })
  .refine((progress) => progress.epoch <= progress.max_epochs, {
    message: "epoch no puede pasar de max_epochs",
  });
export type TrainingProgress = z.infer<typeof trainingProgressSchema>;

/**
 * El worker (OPS-04) asigna experiment_id y run_id al crear el run de MLflow, así
 * que un job queued todavía no tiene ninguno de los dos.
 */
export const trainingJobSchema = z
  .strictObject({
    job_id: identifierSchema,
    status: trainingStatusSchema,
    dataset_version: identifierSchema,
    manifest_hash: manifestHashSchema,
    experiment_id: experimentIdSchema.nullable(),
    run_id: runIdSchema.nullable(),
    checkpoint: checkpointSchema.nullable(),
    params: trainingParamsSchema,
    created_at: timestampSchema,
    started_at: timestampSchema.nullable(),
    finished_at: timestampSchema.nullable(),
    error: contractErrorSchema.nullable(),
    progress: trainingProgressSchema.nullable(),
  })
  .superRefine((job, context) => {
    const issue = (message: string) => context.addIssue({ code: "custom", message });
    const terminal = ["succeeded", "failed", "cancelled"].includes(job.status);
    if (job.status === "queued" && (job.run_id !== null || job.started_at !== null)) {
      issue("Un job queued todavía no tiene run_id ni started_at");
    }
    if (job.status === "queued" && job.progress !== null) {
      issue("Un job queued todavía no reporta progreso");
    }
    if (job.run_id !== null && job.experiment_id === null) {
      issue("Un job con run_id pertenece a un experimento de MLflow");
    }
    if (job.progress !== null && job.progress.max_epochs !== job.params.max_epochs) {
      issue("progress.max_epochs debe coincidir con params.max_epochs");
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

const trainingLogEntrySchema = z.strictObject({
  seq: z.number().int().min(1),
  timestamp: timestampSchema,
  level: z.enum(["info", "warning", "error"]),
  message: labelSchema,
});
export type TrainingLogEntry = z.infer<typeof trainingLogEntrySchema>;

/** APP-03: `GET /api/ml/training/jobs/{job_id}/logs`, de la línea más vieja a la más nueva. */
export const trainingLogsResponseSchema = z
  .strictObject({
    schema_version: schemaVersionSchema,
    job_id: identifierSchema,
    entries: z.array(trainingLogEntrySchema),
  })
  .refine(
    (response) =>
      response.entries
        .slice(1)
        .every((entry, index) => entry.seq > (response.entries[index]?.seq ?? 0)),
    { message: "seq debe ser estrictamente creciente" }
  );
export type TrainingLogsResponse = z.infer<typeof trainingLogsResponseSchema>;

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
    /** APP-04: commit del código que entrenó; null si el run no lo registró. */
    git_commit: gitCommitSchema.nullable(),
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

const curvePointSchema = z.strictObject({ step: countSchema, value: z.number() });
export type CurvePoint = z.infer<typeof curvePointSchema>;

/** APP-04: `GET /api/ml/runs/{run_id}/curves`, historial por época de cada métrica. */
export const runCurvesResponseSchema = z
  .strictObject({
    schema_version: schemaVersionSchema,
    run_id: runIdSchema,
    curves: z.record(labelSchema, z.array(curvePointSchema)),
  })
  .refine(
    (response) =>
      Object.values(response.curves).every((points) =>
        points.slice(1).every((point, index) => point.step > (points[index]?.step ?? -1))
      ),
    { message: "los steps de cada curva deben ser estrictamente crecientes" }
  );
export type RunCurvesResponse = z.infer<typeof runCurvesResponseSchema>;

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

const manifestProvenanceSchema = z.strictObject({
  release_version: identifierSchema,
  images_dvc_hash: z.string(),
  annotations_dvc_hash: z.string(),
  quality_report: z.string(),
  crops_sha256: manifestHashSchema,
});

const manifestSplitSchema = z.strictObject({
  image_count: countSchema,
  ratio: ratioSchema,
  crop_count: countSchema.optional(),
  crop_ratio: ratioSchema.optional(),
});
export const SPLIT_TARGETS = { train: 0.7, validation: 0.2, test: 0.1 } as const;

const manifestRecordSchema = z.strictObject({
  crop_id: identifierSchema,
  source_image_id: cocoIdSchema,
  duplicate_group: identifierSchema,
  class: labelSchema,
  split: z.enum(["train", "validation", "test"]),
});

const manifestClassCountsSchema = z.strictObject({
  dog: countSchema,
  cat: countSchema,
});

const manifestCountsSchema = z.strictObject({
  train: manifestClassCountsSchema,
  validation: manifestClassCountsSchema,
  test: manifestClassCountsSchema,
});

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
    manifest_version: identifierSchema.optional(),
    source_release: identifierSchema.optional(),
    provenance: manifestProvenanceSchema.optional(),
    seed: z
      .number()
      .int()
      .min(0)
      .max(2 ** 31 - 1)
      .optional(),
    records: z.array(manifestRecordSchema).optional(),
    counts: manifestCountsSchema.optional(),
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
    // 70/20/10 salvo redondeo entero: menos de una imagen de distancia al objetivo.
    for (const [name, target] of Object.entries(SPLIT_TARGETS)) {
      const split = manifest.splits[name as keyof typeof SPLIT_TARGETS];
      if (Math.abs(split.image_count - target * manifest.total_images) >= 1) {
        context.addIssue({
          code: "custom",
          message: `${name} debe ser el ${Math.round(target * 100)}% de total_images`,
        });
      }
    }
  });
export type TrainingManifest = z.infer<typeof trainingManifestSchema>;

/**
 * Igual que `training_blocked_reason` en Python: `null` si el release se puede
 * entrenar de forma reproducible. Un Quality Gate `failed` bloquea (warning no), y
 * también falta o invalidez de la procedencia DVC o del manifiesto 70/20/10: pasa
 * `null` en cualquiera de los dos si el archivo falta o no valida.
 */
export function trainingBlockedReason(
  qualityStatus: string,
  provenance: ReleaseProvenance | null,
  manifest: TrainingManifest | null
): string | null {
  if (qualityStatus === "failed") {
    return "El release no pasó el Quality Gate (failed); no se puede entrenar con él.";
  }
  if (provenance === null) {
    return "El release no tiene un provenance.json de DVC válido; no se puede entrenar de forma reproducible.";
  }
  if (manifest === null) {
    return "El release no tiene un manifest.json 70/20/10 válido; no se puede entrenar de forma reproducible.";
  }
  if (provenance.dataset_version !== manifest.dataset_version) {
    return "La procedencia y el manifiesto corresponden a releases distintos.";
  }
  return null;
}

/** Igual que `training_request_rejection`: lo que revisa el POST antes de crear el job. */
export function trainingRequestRejection(
  request: TrainingJobRequest,
  qualityStatus: string,
  provenance: ReleaseProvenance | null,
  manifest: TrainingManifest | null
): string | null {
  const reason = trainingBlockedReason(qualityStatus, provenance, manifest);
  if (reason !== null || manifest === null) return reason;
  if (manifest.dataset_version !== request.dataset_version) {
    return "El manifiesto no corresponde al release solicitado.";
  }
  if (manifest.manifest_hash !== request.manifest_hash) {
    return "El manifest_hash no coincide con el manifiesto del release.";
  }
  return null;
}

/** Mismo mapa que `CONTRACTS` en ml_contracts.py; los tests exigen que coincidan. */
export const ML_CONTRACTS = {
  training_jobs: trainingJobsResponseSchema,
  training_logs: trainingLogsResponseSchema,
  runs: runsResponseSchema,
  run_curves: runCurvesResponseSchema,
  evaluations: evaluationsResponseSchema,
  models: modelsResponseSchema,
  inference_request: inferenceRequestSchema,
  inference: inferenceResponseSchema,
  error: errorResponseSchema,
  training_request: trainingJobRequestSchema,
  provenance: releaseProvenanceSchema,
  manifest: trainingManifestSchema,
} as const;
