import { readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { summarizeEvaluation } from "../src/ml/evaluationSummary";
import { evaluationSchema } from "../src/ml/schemas";

/**
 * ML-10: lo que calcula la pantalla Evaluation con la evaluación final real debe
 * coincidir, a precisión completa, con el recálculo independiente de la auditoría
 * (`reports/quality/ml10_quality_audit.json`, hecho solo desde el CSV de predicciones).
 */
const reports = resolve(process.cwd(), "../reports");
const testDir = resolve(reports, "evaluations/test");

function realEvaluation() {
  const files = readdirSync(testDir).filter((name) => name.endsWith(".json"));
  expect(files).toHaveLength(1);
  const raw = JSON.parse(readFileSync(resolve(testDir, files[0] ?? ""), "utf-8"));
  return evaluationSchema.parse(raw);
}

type AuditClass = {
  class_name: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
  low_recall: boolean;
};

const audit = JSON.parse(
  readFileSync(resolve(reports, "quality/ml10_quality_audit.json"), "utf-8")
) as {
  verified: boolean;
  evaluation_id: string;
  metrics: {
    total: number;
    correct: number;
    accuracy: number;
    meets_target: boolean;
    f1_macro: number;
    per_class: AuditClass[];
    confusion_matrix: number[][];
    baseline: { classes: string[]; accuracy: number };
    most_confused_pairs: { true_class: string; predicted_class: string; count: number }[];
    error_crop_ids: string[];
  };
};

describe("ML-10 portal figures match the independent audit", () => {
  const evaluation = realEvaluation();
  const summary = summarizeEvaluation(evaluation);
  const metrics = audit.metrics;

  it("audits the same final test evaluation", () => {
    expect(audit.verified).toBe(true);
    expect(audit.evaluation_id).toBe(evaluation.evaluation_id);
    expect(evaluation.confusion_matrix).toEqual(metrics.confusion_matrix);
  });

  it("shows the same accuracy, target and F1 macro, without rounding", () => {
    expect(summary.total).toBe(metrics.total);
    expect(summary.correct).toBe(metrics.correct);
    expect(summary.accuracy).toBe(metrics.accuracy);
    expect(summary.meetsTarget).toBe(metrics.meets_target);
    expect(summary.f1Macro).toBe(metrics.f1_macro);
  });

  it("shows the same per-class precision, recall, F1, support and low-recall flag", () => {
    expect(
      summary.perClass.map(({ className, precision, recall, f1, support, lowRecall }) => ({
        class_name: className,
        precision,
        recall,
        f1,
        support,
        low_recall: lowRecall,
      }))
    ).toEqual(
      metrics.per_class.map(({ class_name, precision, recall, f1, support, low_recall }) => ({
        class_name,
        precision,
        recall,
        f1,
        support,
        low_recall,
      }))
    );
  });

  it("shows the same majority baseline and most confused pair", () => {
    expect(summary.baseline.className).toBe(metrics.baseline.classes[0]);
    expect(summary.baseline.accuracy).toBe(metrics.baseline.accuracy);
    const top = metrics.most_confused_pairs[0];
    expect(summary.mostConfused).toEqual(
      top === undefined
        ? null
        : { trueClass: top.true_class, predictedClass: top.predicted_class, count: top.count }
    );
  });

  it("lists as errors exactly the crops the audit found wrong", () => {
    const wrong = evaluation.predictions
      .filter((p) => p.true_class !== p.predicted_class)
      .map((p) => `img${p.image_id}-ann${p.annotation_id}`);
    expect(wrong).toEqual(metrics.error_crop_ids);
  });
});
