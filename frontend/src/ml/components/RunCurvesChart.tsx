import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CurvePoint } from "../schemas";

/**
 * APP-04: colores de las series de la comparación. Son los 3 primeros slots de la
 * paleta categórica de referencia (dataviz), validados contra la superficie blanca del
 * portal con todos los pares: ΔE daltonismo 9.2, visión normal 24.0. El slot 3 queda
 * bajo 3:1 de contraste, por eso cada línea lleva su nombre al final y hay vista de tabla.
 * El color sigue al run (su slot), nunca a su posición en la lista.
 */
export const SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"] as const;

const GRID = "#E7E5E1"; // border
const AXIS_TEXT = "#8A8782"; // ink-muted
const LABEL_TEXT = "#1C1B1A"; // ink
const SURFACE = "#FFFFFF";

export type CurveSeries = {
  slot: number; // 1..3
  runId: string;
  label: string;
  points: CurvePoint[];
};

export function SeriesSwatch({ slot }: Readonly<{ slot: number }>) {
  return (
    <span
      aria-hidden
      className="inline-block h-0.5 w-4 rounded-full"
      style={{ backgroundColor: SERIES_COLORS[slot - 1] }}
    />
  );
}

type Row = { step: number } & Record<string, number>;

function mergeByStep(series: CurveSeries[]): Row[] {
  const rows = new Map<number, Row>();
  for (const { runId, points } of series) {
    for (const { step, value } of points) {
      const row = rows.get(step) ?? { step };
      row[runId] = value;
      rows.set(step, row);
    }
  }
  return [...rows.values()].sort((a, b) => a.step - b.step);
}

/** Alto aproximado del área de trazado (h-48 menos márgenes y eje X), en px. */
const PLOT_HEIGHT_PX = 150;
/** Separación vertical mínima entre dos etiquetas de final de línea, en px. */
const LABEL_GAP_PX = 13;

/**
 * Eje Y con valores redondos: el rango se amplía a décimas y se marca cada 0.1 (o cada
 * 0.2 si el rango pasa de 0.6). Antes, el rango automático daba 0.575 / 0.46 / 0.345.
 */
export function niceAxis(series: CurveSeries[]): { domain: [number, number]; ticks: number[] } {
  const values = series.flatMap((s) => s.points.map((p) => p.value));
  if (values.length === 0) return { domain: [0, 1], ticks: [0, 0.2, 0.4, 0.6, 0.8, 1] };
  let low = Math.floor(Math.min(...values) * 10 + 1e-9);
  let high = Math.ceil(Math.max(...values) * 10 - 1e-9);
  if (low === high) {
    low -= 1;
    high += 1;
  }
  const step = high - low > 6 ? 2 : 1;
  low = Math.floor(low / step) * step;
  high = Math.ceil(high / step) * step;
  const ticks: number[] = [];
  for (let tenth = low; tenth <= high; tenth += step) ticks.push(tenth / 10);
  return { domain: [low / 10, high / 10], ticks };
}

/**
 * Qué líneas llevan su nombre al final. Con 2-3 runs se etiquetan las que no chocan con
 * otra; la que chocaría se omite (la leyenda y el tooltip la identifican), en vez de
 * mover etiquetas lejos de su línea. Una sola línea no se etiqueta: la nombra el título.
 */
export function labelledRunIds(series: CurveSeries[], domain: [number, number]): Set<string> {
  const ends = series.flatMap((s) => {
    const last = s.points.at(-1);
    return last ? [{ runId: s.runId, value: last.value }] : [];
  });
  if (ends.length < 2) return new Set();
  const minGap = (LABEL_GAP_PX / PLOT_HEIGHT_PX) * (domain[1] - domain[0]);
  const kept: { runId: string; value: number }[] = [];
  for (const end of ends.sort((a, b) => b.value - a.value)) {
    if (kept.every((other) => Math.abs(other.value - end.value) >= minGap)) kept.push(end);
  }
  return new Set(kept.map((end) => end.runId));
}

/** Nombre del run al final de su línea (alivio de contraste además de la tabla). */
function endLabel(series: CurveSeries) {
  const lastIndex = series.points.length - 1;
  return ({ x, y, index }: { x?: number | string; y?: number | string; index?: number }) =>
    index === lastIndex && typeof x === "number" && typeof y === "number" ? (
      <text x={x + 6} y={y} dy={4} fontSize={11} fill={LABEL_TEXT}>
        {series.label}
      </text>
    ) : null;
}

/** Una métrica (p. ej. val_accuracy) por época, una línea por run comparado. */
export function RunCurvesChart({
  metric,
  series,
}: Readonly<{ metric: string; series: CurveSeries[] }>) {
  const data = mergeByStep(series);
  const withData = series.filter((s) => s.points.length > 0);
  const axis = niceAxis(withData);
  const labelled = labelledRunIds(withData, axis.domain);

  return (
    <figure
      aria-label={`${metric} por época`}
      className="flex flex-col gap-2 rounded-2xl border border-border bg-surface p-4 shadow-card"
    >
      <figcaption className="text-sm font-medium text-ink">{metric}</figcaption>
      {withData.length > 1 && (
        <ul className="flex flex-wrap gap-3 text-xs text-ink-muted">
          {withData.map((s) => (
            <li key={s.runId} data-legend-slot={s.slot} className="flex items-center gap-1.5">
              <SeriesSwatch slot={s.slot} />
              <span>{s.label}</span>
            </li>
          ))}
        </ul>
      )}
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 8, right: labelled.size > 0 ? 132 : 16, left: 0, bottom: 4 }}
          >
            <CartesianGrid vertical={false} stroke={GRID} strokeWidth={1} />
            <XAxis
              dataKey="step"
              type="number"
              domain={["dataMin", "dataMax"]}
              allowDecimals={false}
              tick={{ fontSize: 11, fill: AXIS_TEXT }}
              axisLine={{ stroke: GRID }}
              tickLine={false}
              label={{
                value: "época",
                position: "insideBottomRight",
                offset: -2,
                fontSize: 11,
                fill: AXIS_TEXT,
              }}
            />
            <YAxis
              tick={{ fontSize: 11, fill: AXIS_TEXT }}
              axisLine={false}
              tickLine={false}
              width={36}
              domain={axis.domain}
              ticks={axis.ticks}
              tickFormatter={(value: number) => value.toFixed(1)}
            />
            <Tooltip
              cursor={{ stroke: AXIS_TEXT, strokeWidth: 1 }}
              contentStyle={{ borderRadius: 12, border: `1px solid ${GRID}`, fontSize: 12 }}
              labelFormatter={(step) => `Época ${step}`}
              formatter={(value, runId) => [
                value,
                series.find((s) => s.runId === runId)?.label ?? String(runId),
              ]}
            />
            {withData.map((s) => (
              <Line
                key={s.runId}
                dataKey={s.runId}
                name={s.runId}
                type="linear"
                stroke={SERIES_COLORS[s.slot - 1]}
                strokeWidth={2}
                strokeLinecap="round"
                strokeLinejoin="round"
                dot={false}
                activeDot={{ r: 4, stroke: SURFACE, strokeWidth: 2 }}
                connectNulls
                isAnimationActive={false}
                label={labelled.has(s.runId) ? endLabel(s) : false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <details className="text-xs">
        <summary className="cursor-pointer text-ink-muted">Ver datos</summary>
        <table className="mt-2 w-full text-left">
          <caption className="sr-only">{`${metric} por época`}</caption>
          <thead className="text-ink-muted">
            <tr>
              <th scope="col" className="py-1 pr-3 font-medium">
                Época
              </th>
              {withData.map((s) => (
                <th key={s.runId} scope="col" className="py-1 pr-3 font-medium">
                  {s.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="font-mono">
            {data.map((row) => (
              <tr key={row.step}>
                <td className="py-0.5 pr-3">{row.step}</td>
                {withData.map((s) => (
                  <td key={s.runId} className="py-0.5 pr-3">
                    {row[s.runId] ?? "—"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
