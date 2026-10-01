import { describe, expect, it } from "vitest";
import { type CurveSeries, labelledRunIds, niceAxis } from "../src/ml/components/RunCurvesChart";

// APP-04: escala del eje Y y etiquetas al final de línea de las curvas comparadas.
// Revisión visual: las etiquetas chocaban cuando las curvas convergen y el eje Y
// mostraba valores como 0.575 / 0.46 / 0.345.

function series(runId: string, slot: number, values: number[]): CurveSeries {
  return {
    slot,
    runId,
    label: runId,
    points: values.map((value, index) => ({ step: index + 1, value })),
  };
}

describe("niceAxis", () => {
  it("redondea el rango a décimas y marca cada 0.1", () => {
    const axis = niceAxis([series("a", 1, [0.58, 0.31, 0.19]), series("b", 2, [0.57, 0.2])]);
    expect(axis.domain).toEqual([0.1, 0.6]);
    expect(axis.ticks).toEqual([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
  });

  it("usa pasos de 0.2 cuando el rango es amplio", () => {
    const axis = niceAxis([series("a", 1, [0.05, 0.92])]);
    expect(axis.domain).toEqual([0, 1]);
    expect(axis.ticks).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
  });

  it("no colapsa un rango plano", () => {
    const axis = niceAxis([series("a", 1, [0.5, 0.5])]);
    expect(axis.domain[0]).toBeLessThan(0.5);
    expect(axis.domain[1]).toBeGreaterThan(0.5);
  });
});

describe("labelledRunIds", () => {
  it("etiqueta todas las líneas cuyos finales están separados", () => {
    const runs = [series("a", 1, [0.6, 0.9]), series("b", 2, [0.6, 0.75]), series("c", 3, [0.6, 0.6])];
    expect(labelledRunIds(runs, [0.5, 1])).toEqual(new Set(["a", "b", "c"]));
  });

  it("omite la etiqueta que chocaría; la leyenda y el tooltip la identifican", () => {
    const runs = [series("a", 1, [0.3, 0.188]), series("b", 2, [0.3, 0.179]), series("c", 3, [0.3, 0.27])];
    const labelled = labelledRunIds(runs, [0.1, 0.6]);
    expect(labelled.has("c")).toBe(true);
    expect(labelled.has("a") && labelled.has("b")).toBe(false);
    expect(labelled.size).toBe(2);
  });

  it("no etiqueta una sola línea: el título de la gráfica ya la nombra", () => {
    expect(labelledRunIds([series("a", 1, [0.3, 0.2])], [0, 1])).toEqual(new Set());
  });
});
