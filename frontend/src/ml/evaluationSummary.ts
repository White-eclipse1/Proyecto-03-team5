import type { Evaluation } from "./schemas";

/** Meta del proyecto (rúbrica 4.3): aciertos / recortes de test ≥ 0.85, sin redondear. */
export const TEST_ACCURACY_TARGET = { numerator: 85, denominator: 100 } as const;

export type ClassSummary = {
  className: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
  /** Recortes de la clase predichos correctamente (diagonal de la matriz). */
  hits: number;
  /** recall < 0.85, comparado con enteros (hits·100 < 85·support). */
  lowRecall: boolean;
};

export type EvaluationSummary = {
  total: number;
  correct: number;
  accuracy: number;
  meetsTarget: boolean;
  f1Macro: number;
  perClass: ClassSummary[];
  rowTotals: number[];
  columnTotals: number[];
  /** Clase más frecuente del mismo test: accuracy de predecirla siempre. */
  baseline: { className: string; accuracy: number };
  /** Celda fuera de la diagonal con más recortes; null si no hubo errores. */
  mostConfused: { trueClass: string; predictedClass: string; count: number } | null;
};

const sum = (values: readonly number[]) => values.reduce((total, value) => total + value, 0);
const ratio = (part: number, whole: number) => (whole === 0 ? 0 : part / whole);

function reachesTarget(part: number, whole: number): boolean {
  const { numerator, denominator } = TEST_ACCURACY_TARGET;
  return part * denominator >= numerator * whole;
}

/**
 * Todo se recalcula desde la matriz (filas: clase real, columnas: predicha). El
 * contrato ya exige que métricas guardadas y predicciones cuadren con ella, pero
 * las métricas guardadas van redondeadas: compararlas con 0.85 podría convertir
 * 0.8495 en una meta alcanzada.
 */
export function summarizeEvaluation(evaluation: Evaluation): EvaluationSummary {
  const names = evaluation.class_names;
  const matrix = evaluation.confusion_matrix;
  const cell = (row: number, column: number) => matrix[row]?.[column] ?? 0;
  const rowTotals = names.map((_, row) => sum(matrix[row] ?? []));
  const columnTotals = names.map((_, column) => sum(names.map((__, row) => cell(row, column))));
  const total = sum(rowTotals);
  const correct = sum(names.map((_, index) => cell(index, index)));

  const perClass = names.map((className, index) => {
    const hits = cell(index, index);
    const support = rowTotals[index] ?? 0;
    const precision = ratio(hits, columnTotals[index] ?? 0);
    const recall = ratio(hits, support);
    const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    return {
      className,
      precision,
      recall,
      f1,
      support,
      hits,
      lowRecall: !reachesTarget(hits, support),
    };
  });

  const majority = names.reduce(
    (best, className, index) =>
      (rowTotals[index] ?? 0) > best.support ? { className, support: rowTotals[index] ?? 0 } : best,
    { className: names[0] ?? "", support: -1 }
  );

  let mostConfused: EvaluationSummary["mostConfused"] = null;
  names.forEach((trueClass, row) => {
    names.forEach((predictedClass, column) => {
      const count = cell(row, column);
      if (row !== column && count > 0 && count > (mostConfused?.count ?? 0)) {
        mostConfused = { trueClass, predictedClass, count };
      }
    });
  });

  return {
    total,
    correct,
    accuracy: ratio(correct, total),
    meetsTarget: reachesTarget(correct, total),
    f1Macro: ratio(sum(perClass.map((entry) => entry.f1)), names.length),
    perClass,
    rowTotals,
    columnTotals,
    baseline: { className: majority.className, accuracy: ratio(majority.support, total) },
    mostConfused,
  };
}
