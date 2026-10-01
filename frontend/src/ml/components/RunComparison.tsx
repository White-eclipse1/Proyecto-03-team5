import { useRunCurves } from "../dataSource";
import type { ExperimentRun } from "../schemas";
import { type CurveSeries, RunCurvesChart, SeriesSwatch } from "./RunCurvesChart";

/** Métricas por época que registra ML-04 (ver app/tracking/run_schema.py). */
export const CURVE_METRICS = ["train_loss", "val_loss", "train_accuracy", "val_accuracy"] as const;

/** Parámetros en el orden de TrainingParams: los 7 de la rúbrica primero. */
const PARAM_ORDER = [
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
];

export type SlottedRun = { slot: number; run: ExperimentRun };

function orderedKeys(keys: Set<string>, preferred: readonly string[]): string[] {
  const first = preferred.filter((key) => keys.has(key));
  const rest = [...keys].filter((key) => !preferred.includes(key)).sort();
  return [...first, ...rest];
}

function DiffRow({
  label,
  values,
}: Readonly<{ label: string; values: (string | number | undefined)[] }>) {
  const differs = values.length > 1 && new Set(values.map(String)).size > 1;
  return (
    <tr className={differs ? "bg-accent-lilac-soft/60" : undefined}>
      <th scope="row" className="whitespace-nowrap px-3 py-1.5 font-mono text-xs font-normal">
        {label}
      </th>
      {values.map((value, index) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: columnas fijas por run seleccionado
        <td key={index} className="whitespace-nowrap px-3 py-1.5 font-mono text-xs">
          {value ?? <span className="text-ink-faint">—</span>}
        </td>
      ))}
      <td className="px-3 py-1.5 text-xs">
        {differs && <span className="font-medium text-ink">Distinto</span>}
      </td>
    </tr>
  );
}

/**
 * APP-04: compara hasta 3 runs (o muestra las curvas de uno). Las curvas vienen del
 * historial de MLflow de cada run; `version` las vuelve a pedir mientras alguno corre.
 */
export function RunComparison({
  selected,
  version,
}: Readonly<{ selected: SlottedRun[]; version: number }>) {
  const slot1 = selected.find((s) => s.slot === 1)?.run.run_id ?? null;
  const slot2 = selected.find((s) => s.slot === 2)?.run.run_id ?? null;
  const slot3 = selected.find((s) => s.slot === 3)?.run.run_id ?? null;
  const curves = [
    useRunCurves(slot1, version),
    useRunCurves(slot2, version),
    useRunCurves(slot3, version),
  ];

  const single = selected.length === 1 ? selected[0] : undefined;
  const title = single ? `Curvas de ${single.run.run_name}` : "Comparación de runs";
  const headingId = "run-comparison-heading";

  const paramKeys = orderedKeys(
    new Set(selected.flatMap((s) => Object.keys(s.run.params))),
    PARAM_ORDER
  );
  const metricKeys = orderedKeys(new Set(selected.flatMap((s) => Object.keys(s.run.metrics))), []);

  const errors = selected.flatMap(({ slot, run }) => {
    const state = curves[slot - 1];
    return state?.status === "error" ? [{ run, error: state.error }] : [];
  });
  const series = (metric: string): CurveSeries[] =>
    selected.map(({ slot, run }) => {
      const state = curves[slot - 1];
      return {
        slot,
        runId: run.run_id,
        label: run.run_name,
        points: state?.status === "success" ? (state.data.curves[metric] ?? []) : [],
      };
    });
  const loaded = selected.some(({ slot }) => curves[slot - 1]?.status === "success");
  const metricsWithCurves = CURVE_METRICS.filter((metric) =>
    series(metric).some((s) => s.points.length > 0)
  );

  return (
    <section aria-labelledby={headingId} className="flex flex-col gap-4">
      <h2 id={headingId} className="text-lg font-semibold">
        {title}
      </h2>

      {selected.length > 1 && (
        <div className="overflow-x-auto rounded-2xl border border-border bg-surface shadow-card">
          <table className="w-full text-left text-sm">
            <caption className="sr-only">Parámetros y métricas de los runs comparados</caption>
            <thead className="border-b border-border text-xs text-ink-muted">
              <tr>
                <th scope="col" className="px-3 py-2 font-medium">
                  Campo
                </th>
                {selected.map(({ slot, run }) => (
                  <th key={run.run_id} scope="col" className="px-3 py-2 font-medium">
                    <span data-series-slot={slot} className="flex items-center gap-1.5">
                      <SeriesSwatch slot={slot} />
                      <span>{run.run_name}</span>
                    </span>
                  </th>
                ))}
                <th scope="col" className="px-3 py-2 font-medium">
                  <span className="sr-only">Diferencia</span>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {paramKeys.map((key) => (
                <DiffRow key={key} label={key} values={selected.map((s) => s.run.params[key])} />
              ))}
              {metricKeys.map((key) => (
                <DiffRow key={key} label={key} values={selected.map((s) => s.run.metrics[key])} />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {errors.map(({ run, error }) => (
        <p key={run.run_id} role="alert" className="flex gap-1 text-sm text-red-600">
          <span className="font-medium">{`${run.run_name}:`}</span>
          <span>{error.message}</span>
        </p>
      ))}
      {!loaded && errors.length === 0 && <p className="text-sm text-ink-muted">Cargando curvas…</p>}
      {loaded && metricsWithCurves.length === 0 && (
        <p className="text-sm text-ink-muted">
          Estos runs todavía no registraron métricas por época.
        </p>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        {metricsWithCurves.map((metric) => (
          <RunCurvesChart key={metric} metric={metric} series={series(metric)} />
        ))}
      </div>
    </section>
  );
}
