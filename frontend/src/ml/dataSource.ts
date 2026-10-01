import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE_URL } from "@/lib/api/client";
import {
  type ContractError,
  type EvaluationsResponse,
  evaluationsResponseSchema,
  type ModelsResponse,
  modelsResponseSchema,
  type RunsResponse,
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
 * reales respetando los contratos de `schemas.ts`.
 */
export const ML_ENDPOINTS = {
  trainingJobs: "/ml/training/jobs",
  runs: "/ml/runs",
  evaluations: "/ml/evaluations",
  models: "/ml/models",
  inference: "/ml/inference",
} as const;

// Referencias estables: useMlResource las usa como dependencias de su efecto.
const noJobs = (response: TrainingJobsResponse) => response.jobs.length === 0;
const noRuns = (response: RunsResponse) => response.runs.length === 0;
const noEvaluations = (response: EvaluationsResponse) => response.evaluations.length === 0;
const noModels = (response: ModelsResponse) => response.models.length === 0;
const noReadyModels = (response: ModelsResponse) =>
  !response.models.some((model) => model.status === "READY");

export function useTrainingJobs() {
  return useMlResource(ML_ENDPOINTS.trainingJobs, trainingJobsResponseSchema, noJobs);
}

export function useExperimentRuns() {
  return useMlResource(ML_ENDPOINTS.runs, runsResponseSchema, noRuns);
}

export function useEvaluations() {
  return useMlResource(ML_ENDPOINTS.evaluations, evaluationsResponseSchema, noEvaluations);
}

export function useRegisteredModels() {
  return useMlResource(ML_ENDPOINTS.models, modelsResponseSchema, noModels);
}

export function useReadyModels() {
  return useMlResource(ML_ENDPOINTS.models, modelsResponseSchema, noReadyModels);
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
