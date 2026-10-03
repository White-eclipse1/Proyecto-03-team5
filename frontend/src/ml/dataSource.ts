import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE_URL } from "@/lib/api/client";
import {
  type ContractError,
  type ExperimentRun,
  evaluationOverviewSchema,
  type InferenceRequest,
  type InferenceResponse,
  inferenceResponseSchema,
  type ModelsResponse,
  modelsResponseSchema,
  type RunCurvesResponse,
  type RunsResponse,
  runCurvesResponseSchema,
  runsResponseSchema,
  type TrainingJob,
  type TrainingJobRequest,
  type TrainingJobsResponse,
  type TrainingLogEntry,
  trainingJobSchema,
  trainingJobsResponseSchema,
  trainingLogsResponseSchema,
} from "./schemas";
import { errorFromResponse, useMlResource } from "./useMlResource";

/**
 * Rutas del backend para las pantallas de modelos, relativas a `API_BASE_URL`.
 * `trainingJobs` lo sirve `ml-api` (APP-03); APP-04…APP-07 conectan las demás fuentes
 * reales respetando los contratos de `schemas.ts`. `evaluation`, `predictionsCsv` y
 * `crops` son de APP-05.
 */
export const ML_ENDPOINTS = {
  trainingJobs: "/ml/training/jobs",
  runs: "/ml/runs",
  evaluation: "/ml/evaluation",
  predictionsCsv: "/ml/evaluation/predictions.csv",
  crops: "/ml/crops",
  models: "/ml/models",
  inference: "/ml/inference",
  inferenceUpload: "/ml/inference/upload",
} as const;

// Referencias estables: useMlResource las usa como dependencias de su efecto.
const noJobs = (response: TrainingJobsResponse) => response.jobs.length === 0;
const noRuns = (response: RunsResponse) => response.runs.length === 0;
const never = () => false;
const noModels = (response: ModelsResponse) => response.models.length === 0;
/** APP-06: solo se puede inferir con una versión cuyo paquete está en este servidor. */
const noServableModels = (response: ModelsResponse) =>
  !response.models.some((model) => model.servable);

export function useTrainingJobs() {
  return useMlResource(ML_ENDPOINTS.trainingJobs, trainingJobsResponseSchema, noJobs);
}

export function useExperimentRuns() {
  return useMlResource(ML_ENDPOINTS.runs, runsResponseSchema, noRuns);
}

/** APP-05: siempre trae un estado (sin candidato también es una respuesta, no "vacío"). */
export function useEvaluationOverview() {
  return useMlResource(ML_ENDPOINTS.evaluation, evaluationOverviewSchema, never);
}

/** APP-06: descarga de un archivo del paquete de una versión (ml-api lo verifica). */
export function modelFileUrl(version: string, name: string): string {
  const path = name.split("/").map(encodeURIComponent).join("/");
  return `${API_BASE_URL}${ML_ENDPOINTS.models}/${encodeURIComponent(version)}/files/${path}`;
}

/** URL del recorte de una predicción: `crop_id = img<image_id>-ann<annotation_id>` (ML-01). */
export function cropUrl(imageId: number, annotationId: number): string {
  return `${API_BASE_URL}${ML_ENDPOINTS.crops}/img${imageId}-ann${annotationId}`;
}

export function useRegisteredModels() {
  return useMlResource(ML_ENDPOINTS.models, modelsResponseSchema, noModels);
}

export function useReadyModels() {
  return useMlResource(ML_ENDPOINTS.models, modelsResponseSchema, noServableModels);
}

export type CreateTrainingJobResult =
  | { ok: true; job: TrainingJob }
  | { ok: false; error: ContractError };

/**
 * `POST /api/ml/training/jobs` con un request ya validado. El backend vuelve a
 * validarlo con el mismo contrato (`TrainingJobRequest` en ml_contracts.py).
 */
export async function createTrainingJob(
  request: TrainingJobRequest
): Promise<CreateTrainingJobResult> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${ML_ENDPOINTS.trainingJobs}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
  } catch {
    return {
      ok: false,
      error: {
        code: "network_error",
        message: "No se pudo contactar al servidor.",
        retryable: true,
      },
    };
  }
  if (res.status === 404) {
    return {
      ok: false,
      error: {
        code: "source_not_connected",
        message: "El servicio de entrenamiento todavía no está conectado; no se creó ningún job.",
        retryable: true,
      },
    };
  }
  if (!res.ok) return { ok: false, error: await errorFromResponse(res) };
  const parsed = trainingJobSchema.safeParse(await res.json().catch(() => null));
  if (!parsed.success) {
    return {
      ok: false,
      error: {
        code: "contract_mismatch",
        message: "El servidor respondió, pero el job no cumple el contrato esperado.",
        retryable: false,
      },
    };
  }
  return { ok: true, job: parsed.data };
}

/** APP-07: lo que se clasifica, una imagen subida o un recorte del portal. */
export type InferenceInput =
  | { kind: "upload"; modelName: string; modelVersion: string; file: File }
  | { kind: "crop"; request: InferenceRequest };

export type InferenceResult =
  | { ok: true; response: InferenceResponse }
  | { ok: false; error: ContractError };

/**
 * `POST /api/ml/inference/upload` (multipart) o `POST /api/ml/inference` (JSON). La
 * respuesta se valida con el contrato y debe ser del modelo pedido.
 */
export async function classifyImage(input: InferenceInput): Promise<InferenceResult> {
  const [modelName, modelVersion] =
    input.kind === "upload"
      ? [input.modelName, input.modelVersion]
      : [input.request.model_name, input.request.model_version];
  let res: Response;
  try {
    if (input.kind === "upload") {
      const form = new FormData();
      form.append("model_name", input.modelName);
      form.append("model_version", input.modelVersion);
      form.append("file", input.file);
      res = await fetch(`${API_BASE_URL}${ML_ENDPOINTS.inferenceUpload}`, {
        method: "POST",
        body: form,
      });
    } else {
      res = await fetch(`${API_BASE_URL}${ML_ENDPOINTS.inference}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input.request),
      });
    }
  } catch {
    return {
      ok: false,
      error: {
        code: "network_error",
        message: "No se pudo contactar al servidor.",
        retryable: true,
      },
    };
  }
  if (!res.ok) return { ok: false, error: await errorFromResponse(res) };
  const parsed = inferenceResponseSchema.safeParse(await res.json().catch(() => null));
  if (!parsed.success) {
    return {
      ok: false,
      error: {
        code: "contract_mismatch",
        message: "La respuesta no cumple el contrato esperado.",
        retryable: false,
      },
    };
  }
  const response = parsed.data;
  if (
    response.model_name !== modelName ||
    response.model_version !== modelVersion ||
    response.source !== input.kind
  ) {
    return {
      ok: false,
      error: {
        code: "model_mismatch",
        message: "La respuesta no corresponde a la model version elegida.",
        retryable: false,
      },
    };
  }
  return { ok: true, response };
}

export type EnqueueInferenceResult =
  | {
      ok: true;
      imageId: number;
      created: boolean;
      idempotencyKey: string;
    }
  | {
      ok: false;
      error: ContractError;
    };

/**
 * APP-08 — envía el resultado real de Inference a la cola existente
 * de anotación.
 *
 * Para uploads reutiliza el archivo original.
 * Para crops descarga el recorte real servido por el portal.
 */
export async function enqueueInferenceResult(
  response: InferenceResponse,
  originalFile: File | null
): Promise<EnqueueInferenceResult> {
  let file: File;

  try {
    if (response.source === "upload") {
      if (originalFile === null || response.upload === null) {
        return {
          ok: false,
          error: {
            code: "missing_inference_source",
            message: "No está disponible la imagen original de esta inferencia.",
            retryable: false,
          },
        };
      }

      file = originalFile;
    } else {
      if (response.crop === null) {
        return {
          ok: false,
          error: {
            code: "missing_inference_source",
            message: "No está disponible el recorte de esta inferencia.",
            retryable: false,
          },
        };
      }

      const cropResponse = await fetch(
        cropUrl(response.crop.image_id, response.crop.annotation_id)
      );

      if (!cropResponse.ok) {
        return {
          ok: false,
          error: {
            code: "crop_download_failed",
            message: "No se pudo recuperar el recorte para enviarlo a anotación.",
            retryable: true,
          },
        };
      }

      const blob = await cropResponse.blob();
      file = new File(
        [blob],
        `img${response.crop.image_id}-ann${response.crop.annotation_id}.png`,
        { type: blob.type || "image/png" }
      );
    }

    const sourceRef =
      response.source === "crop" && response.crop !== null
        ? `${response.crop.dataset_version}:img${response.crop.image_id}-ann${response.crop.annotation_id}`
        : `sha256:${response.upload?.sha256 ?? ""}`;

    const metadata = {
      sourceKind: response.source,
      sourceRef,
      modelName: response.model_name,
      modelVersion: response.model_version,
      runId: response.run_id,
      checkpointSha256: response.checkpoint_sha256,
      predictedClass: response.predicted_class,
      probabilities: {
        dog: response.probabilities.dog ?? 0,
        cat: response.probabilities.cat ?? 0,
      },
    };

    const form = new FormData();
    form.append("image", file);
    form.append("metadata", JSON.stringify(metadata));

    const queueResponse = await fetch(`${API_BASE_URL}/images/from-inference`, {
      method: "POST",
      body: form,
    });

    if (!queueResponse.ok) {
      return {
        ok: false,
        error: await errorFromResponse(queueResponse),
      };
    }

    const body: unknown = await queueResponse.json().catch(() => null);

    if (
      typeof body !== "object" ||
      body === null ||
      !("imageId" in body) ||
      !("created" in body) ||
      !("idempotencyKey" in body) ||
      typeof body.imageId !== "number" ||
      typeof body.created !== "boolean" ||
      typeof body.idempotencyKey !== "string"
    ) {
      return {
        ok: false,
        error: {
          code: "contract_mismatch",
          message: "La respuesta de la cola no cumple el contrato esperado.",
          retryable: false,
        },
      };
    }

    return {
      ok: true,
      imageId: body.imageId,
      created: body.created,
      idempotencyKey: body.idempotencyKey,
    };
  } catch {
    return {
      ok: false,
      error: {
        code: "network_error",
        message: "No se pudo contactar al servidor de anotación.",
        retryable: true,
      },
    };
  }
}

export type TrainingLogsState = {
  entries: TrainingLogEntry[];
  error: ContractError | null;
  loaded: boolean;
};

const NO_LOGS_YET: TrainingLogsState = { entries: [], error: null, loaded: false };

/**
 * APP-03: `GET /api/ml/training/jobs/{job_id}/logs?after=<seq>`. Pide solo las líneas
 * nuevas y, mientras `active`, vuelve a preguntar cada `pollMs`.
 */
export function useTrainingJobLogs(jobId: string, active: boolean, pollMs: number) {
  const [state, setState] = useState<TrainingLogsState>(NO_LOGS_YET);
  const lastSeq = useRef(0);
  const currentJob = useRef(jobId);

  const fetchNew = useCallback(async () => {
    const fail = (error: ContractError) => {
      if (currentJob.current === jobId) setState((prev) => ({ ...prev, error, loaded: true }));
    };
    let res: Response;
    try {
      res = await fetch(
        `${API_BASE_URL}${ML_ENDPOINTS.trainingJobs}/${encodeURIComponent(jobId)}/logs?after=${lastSeq.current}`
      );
    } catch {
      fail({
        code: "network_error",
        message: "No se pudo contactar al servidor.",
        retryable: true,
      });
      return;
    }
    if (!res.ok) {
      fail(await errorFromResponse(res));
      return;
    }
    const parsed = trainingLogsResponseSchema.safeParse(await res.json().catch(() => null));
    if (!parsed.success || parsed.data.job_id !== jobId) {
      fail({
        code: "contract_mismatch",
        message: "La respuesta no cumple el contrato esperado.",
        retryable: false,
      });
      return;
    }
    if (currentJob.current !== jobId) return;
    const fresh = parsed.data.entries.filter((entry) => entry.seq > lastSeq.current);
    lastSeq.current = fresh.at(-1)?.seq ?? lastSeq.current;
    setState((prev) => ({ entries: [...prev.entries, ...fresh], error: null, loaded: true }));
  }, [jobId]);

  useEffect(() => {
    currentJob.current = jobId;
    lastSeq.current = 0;
    setState(NO_LOGS_YET);
    void fetchNew();
  }, [jobId, fetchNew]);

  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => void fetchNew(), pollMs);
    return () => clearInterval(timer);
  }, [active, fetchNew, pollMs]);

  return state;
}

/** APP-04: UI de MLflow (solo loopback en el host, ver docker-compose.yml). */
const MLFLOW_UI_URL = (import.meta.env.VITE_MLFLOW_UI_URL ?? "http://localhost:5000").replace(
  /\/$/,
  ""
);

/** El mismo run (mismo experiment_id y run_id) en la UI de MLflow. */
export function mlflowRunUrl(run: Pick<ExperimentRun, "experiment_id" | "run_id">): string {
  return `${MLFLOW_UI_URL}/#/experiments/${run.experiment_id}/runs/${run.run_id}`;
}

export type RunCurvesState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; error: ContractError }
  | { status: "success"; data: RunCurvesResponse };

/**
 * APP-04: `GET /api/ml/runs/{run_id}/curves`. Con `runId` null no pide nada (un hueco
 * de la comparación). Vuelve a pedir cuando cambia `version`, para seguir un run activo.
 */
export function useRunCurves(runId: string | null, version = 0): RunCurvesState {
  const [state, setState] = useState<RunCurvesState>({ status: "idle" });

  // biome-ignore lint/correctness/useExhaustiveDependencies: `version` no se lee; es el disparador para volver a pedir las curvas de un run en curso.
  useEffect(() => {
    if (runId === null) {
      setState({ status: "idle" });
      return;
    }
    let cancelled = false;
    setState((prev) =>
      prev.status === "success" && prev.data.run_id === runId ? prev : { status: "loading" }
    );
    fetch(`${API_BASE_URL}${ML_ENDPOINTS.runs}/${encodeURIComponent(runId)}/curves`)
      .then(async (res) => {
        if (!res.ok) throw await errorFromResponse(res);
        const parsed = runCurvesResponseSchema.safeParse(await res.json().catch(() => null));
        if (!parsed.success || parsed.data.run_id !== runId) {
          throw {
            code: "contract_mismatch",
            message: "La respuesta no cumple el contrato esperado.",
            retryable: false,
          } satisfies ContractError;
        }
        if (!cancelled) setState({ status: "success", data: parsed.data });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        const contractError =
          typeof error === "object" && error !== null && "code" in error
            ? (error as ContractError)
            : {
                code: "network_error",
                message: "No se pudo contactar al servidor.",
                retryable: true,
              };
        setState({ status: "error", error: contractError });
      });
    return () => {
      cancelled = true;
    };
  }, [runId, version]);

  return state;
}
