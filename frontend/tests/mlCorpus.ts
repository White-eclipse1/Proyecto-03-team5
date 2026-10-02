import { readFileSync } from "node:fs";
import path from "node:path";

/**
 * Corpus de contrato compartido con Python (`app/presentation/examples/ml/`).
 * Se lee en tiempo de ejecución, no con `import ... from "*.json"`: así el
 * `tsc` del build Docker del frontend (contexto `./frontend`) no necesita
 * archivos fuera de su carpeta. Solo para tests: nunca es fuente de producción.
 */
export type MlContractName =
  | "training_jobs"
  | "training_logs"
  | "runs"
  | "run_curves"
  | "evaluations"
  | "models"
  | "inference_request"
  | "inference"
  | "error"
  | "training_request"
  | "provenance"
  | "manifest";

export interface InvalidCase {
  name: string;
  contract: MlContractName;
  reason: string;
  /** Campo que ambos lados deben señalar (`params.batch_size`), si aplica. */
  field?: string;
  document: unknown;
}

const CORPUS = path.resolve(__dirname, "../../app/presentation/examples/ml");

function readJson(filename: string): unknown {
  return JSON.parse(readFileSync(path.join(CORPUS, filename), "utf-8"));
}

export function loadMlExample(contract: MlContractName): unknown {
  return readJson(`${contract}.json`);
}

export function loadInvalidCases(): InvalidCase[] {
  return readJson("invalid_cases.json") as InvalidCase[];
}
