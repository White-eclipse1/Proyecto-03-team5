import {
  type EvaluationsResponse,
  evaluationsResponseSchema,
  type ModelsResponse,
  modelsResponseSchema,
  type RunsResponse,
  runsResponseSchema,
  type TrainingJobsResponse,
  trainingJobsResponseSchema,
} from "./schemas";
import { useMlResource } from "./useMlResource";

/**
 * Rutas del backend para las pantallas de modelos, relativas a `API_BASE_URL`.
 * Todavía no las implementa ningún servicio: APP-02…APP-07 conectan las fuentes
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
