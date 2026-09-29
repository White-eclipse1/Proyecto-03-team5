import { describe, expect, it } from "vitest";
import { evaluationsResponseSchema, ML_CONTRACTS, modelsResponseSchema } from "../src/ml/schemas";
import { loadInvalidCases, loadMlExample, type MlContractName } from "./mlCorpus";

// APP-01: mismo corpus que `app/tests/test_ml_contracts.py` (Pydantic).
const CONTRACT_NAMES: MlContractName[] = [
  "training_jobs",
  "runs",
  "evaluations",
  "models",
  "inference_request",
  "inference",
  "error",
];

describe("APP-01 contratos de modelos (espejo Zod de ml_contracts.py)", () => {
  it("declara exactamente los mismos contratos que Python", () => {
    expect(Object.keys(ML_CONTRACTS).sort()).toEqual([...CONTRACT_NAMES].sort());
  });

  it.each(CONTRACT_NAMES)("acepta el ejemplo compartido de %s sin alterar valores", (name) => {
    const document = loadMlExample(name);
    const parsed = ML_CONTRACTS[name].safeParse(document);
    expect(parsed.error).toBeUndefined();
    expect(parsed.data).toEqual(document);
  });

  it.each(CONTRACT_NAMES)("congela schema_version en 1.0 para %s", (name) => {
    for (const value of [null, "2.0", 1]) {
      const document = { ...(loadMlExample(name) as object), schema_version: value };
      expect(ML_CONTRACTS[name].safeParse(document).success).toBe(false);
    }
  });

  it.each([
    ["training_jobs", "jobs"],
    ["runs", "runs"],
    ["evaluations", "evaluations"],
    ["models", "models"],
  ] as const)("acepta %s vacío como estado empty válido", (name, key) => {
    expect(ML_CONTRACTS[name].safeParse({ schema_version: "1.0", [key]: [] }).success).toBe(true);
  });

  it.each(loadInvalidCases().map((c) => [c.name, c] as const))(
    "rechaza el caso compartido %s",
    (_name, invalid) => {
      expect(ML_CONTRACTS[invalid.contract].safeParse(invalid.document).success).toBe(false);
    }
  );

  it("mantiene dataset_version y model_version como campos distintos", () => {
    const parsed = modelsResponseSchema.parse(loadMlExample("models"));
    const model = parsed.models[0]!;
    expect(model.dataset_version).toBe("demo-v1.0.0");
    expect(model.model_version).toBe("3");
    const swapped = {
      ...parsed,
      models: [{ ...model, model_version: model.dataset_version }],
    };
    expect(modelsResponseSchema.safeParse(swapped).success).toBe(false);
  });

  it("acepta métricas redondeadas a 3 decimales y rechaza desvíos mayores", () => {
    const parsed = evaluationsResponseSchema.parse(loadMlExample("evaluations"));
    const withAccuracy = (accuracy_top1: number) => ({
      ...parsed,
      evaluations: [
        { ...parsed.evaluations[0]!, metrics: { ...parsed.evaluations[0]!.metrics, accuracy_top1 } },
      ],
    });
    expect(evaluationsResponseSchema.safeParse(withAccuracy(0.8005)).success).toBe(true);
    expect(evaluationsResponseSchema.safeParse(withAccuracy(0.802)).success).toBe(false);
  });
});
