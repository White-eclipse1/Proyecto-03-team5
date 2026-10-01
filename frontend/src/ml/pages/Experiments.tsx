import { ExternalLink } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Cell, Missing } from "../components/DataTable";
import { MlPage } from "../components/MlPage";
import { MlResourceBoundary } from "../components/MlResourceBoundary";
import { NonFinite } from "../components/NonFinite";
import { RunComparison, type SlottedRun } from "../components/RunComparison";
import { mlflowRunUrl, useExperimentRuns } from "../dataSource";
import type { ExperimentRun } from "../schemas";

/** Cada cuánto se vuelve a leer MLflow mientras haya runs en curso. */
export const RUNS_POLL_MS = 5000;

/** Máximo de runs comparados: los 3 colores validados para todos los pares. */
const MAX_COMPARED = 3;

/** Los 7 hiperparámetros de la rúbrica, en el orden de TrainingParams. */
const PARAM_COLUMNS = [
  "optimizer",
  "batch_size",
  "max_epochs",
  "learning_rate",
  "image_size",
  "hidden_layers",
  "dropout",
];

const isActive = (run: ExperimentRun) => run.status === "RUNNING" || run.status === "SCHEDULED";

type SortKey = "run_name" | "status" | "start_time" | `metric:${string}`;
type Sort = { key: SortKey; direction: "ascending" | "descending" };

function sortValue(run: ExperimentRun, key: SortKey): string | number | null | undefined {
  if (key.startsWith("metric:")) return run.metrics[key.slice("metric:".length)];
  // Por el instante real, no por el texto ISO: "…:20Z" > "…:20.500000Z" como texto.
  if (key === "start_time") return Date.parse(run.start_time);
  return run[key as "run_name" | "status"];
}

function compareRuns(a: ExperimentRun, b: ExperimentRun, sort: Sort): number {
  const left = sortValue(a, sort.key);
  const right = sortValue(b, sort.key);
  const leftMissing = left === undefined || left === null;
  const rightMissing = right === undefined || right === null;
  if (leftMissing || rightMissing) {
    // Sin valor (o no finito) siempre al final, en cualquier dirección.
    return leftMissing ? (rightMissing ? a.run_id.localeCompare(b.run_id) : 1) : -1;
  }
  const order =
    typeof left === "number" && typeof right === "number"
      ? left - right
      : String(left).localeCompare(String(right));
  // Empate exacto: por run_id, igual que la API.
  if (order === 0) return a.run_id.localeCompare(b.run_id);
  return sort.direction === "ascending" ? order : -order;
}

/** `?runs=a,,c`: cada posición es un color; quitar un run deja su hueco. */
function parseSlots(value: string | null): (string | null)[] {
  const slots = (value ?? "").split(",").slice(0, MAX_COMPARED);
  return Array.from({ length: MAX_COMPARED }, (_, index) => slots[index] || null);
}

function serializeSlots(slots: (string | null)[]): string {
  return slots
    .map((id) => id ?? "")
    .join(",")
    .replace(/,+$/, "");
}

function SortableHeader({
  label,
  sortKey,
  sort,
  onSort,
}: Readonly<{ label: string; sortKey: SortKey; sort: Sort; onSort: (key: SortKey) => void }>) {
  const active = sort.key === sortKey;
  return (
    <th
      scope="col"
      aria-sort={active ? sort.direction : "none"}
      className="whitespace-nowrap px-4 py-3 font-medium"
    >
      <button
        type="button"
        onClick={() => onSort(sortKey)}
        className="inline-flex items-center gap-1 uppercase hover:text-ink"
      >
        {label}
        <span aria-hidden className="text-ink-faint">
          {active ? (sort.direction === "ascending" ? "▲" : "▼") : "↕"}
        </span>
      </button>
    </th>
  );
}

function PlainHeader({ label }: Readonly<{ label: string }>) {
  return (
    <th scope="col" className="whitespace-nowrap px-4 py-3 font-medium">
      {label}
    </th>
  );
}

function RunsTable({
  runs,
  allRuns,
  slots,
  onToggle,
}: Readonly<{
  runs: ExperimentRun[];
  allRuns: ExperimentRun[];
  slots: (string | null)[];
  onToggle: (runId: string) => void;
}>) {
  const [sort, setSort] = useState<Sort>({ key: "start_time", direction: "descending" });
  const validationMetrics = useMemo(
    () =>
      [...new Set(allRuns.flatMap((run) => Object.keys(run.metrics)))]
        .filter((name) => name.startsWith("val_"))
        .sort(),
    [allRuns]
  );
  const sorted = [...runs].sort((a, b) => compareRuns(a, b, sort));
  const full = slots.every((id) => id !== null);

  const onSort = (key: SortKey) =>
    setSort((current) =>
      current.key === key
        ? { key, direction: current.direction === "ascending" ? "descending" : "ascending" }
        : {
            key,
            direction:
              key.startsWith("metric:") || key === "start_time" ? "descending" : "ascending",
          }
    );

  return (
    <div className="overflow-x-auto rounded-2xl border border-border bg-surface shadow-card">
      <table className="w-full text-left text-sm">
        <caption className="sr-only">Runs de MLflow</caption>
        <thead className="border-b border-border text-xs uppercase text-ink-muted">
          <tr>
            <PlainHeader label="Comparar" />
            <SortableHeader label="Run" sortKey="run_name" sort={sort} onSort={onSort} />
            <PlainHeader label="Run ID" />
            <SortableHeader label="Estado" sortKey="status" sort={sort} onSort={onSort} />
            <PlainHeader label="Dataset" />
            <PlainHeader label="Manifest" />
            <PlainHeader label="Commit" />
            {PARAM_COLUMNS.map((param) => (
              <PlainHeader key={param} label={param} />
            ))}
            {validationMetrics.map((metric) => (
              <SortableHeader
                key={metric}
                label={metric}
                sortKey={`metric:${metric}`}
                sort={sort}
                onSort={onSort}
              />
            ))}
            <SortableHeader label="Inicio" sortKey="start_time" sort={sort} onSort={onSort} />
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {sorted.map((run) => {
            const checked = slots.includes(run.run_id);
            return (
              <tr key={run.run_id}>
                <Cell>
                  <input
                    type="checkbox"
                    aria-label={`Comparar ${run.run_name}`}
                    checked={checked}
                    disabled={!checked && full}
                    onChange={() => onToggle(run.run_id)}
                    className="h-4 w-4 accent-accent-lilac"
                  />
                </Cell>
                <Cell>
                  <span className="inline-flex items-center gap-2">
                    {run.run_name}
                    <a
                      href={mlflowRunUrl(run)}
                      target="_blank"
                      rel="noreferrer"
                      aria-label={`Abrir ${run.run_name} en MLflow`}
                      title="Abrir en MLflow"
                      className="text-ink-faint hover:text-accent-lilac"
                    >
                      <ExternalLink aria-hidden className="h-3.5 w-3.5" />
                    </a>
                  </span>
                </Cell>
                <Cell mono>{run.run_id}</Cell>
                <Cell>{run.status}</Cell>
                <Cell mono>{run.dataset_version}</Cell>
                <Cell mono>{run.manifest_hash}</Cell>
                <Cell mono>
                  {run.git_commit ? (
                    <span title={run.git_commit}>{run.git_commit.slice(0, 7)}</span>
                  ) : (
                    <span title="Commit no registrado">
                      <Missing />
                    </span>
                  )}
                </Cell>
                {PARAM_COLUMNS.map((param) => (
                  <Cell key={param} mono>
                    {run.params[param] ?? <Missing />}
                  </Cell>
                ))}
                {validationMetrics.map((metric) => (
                  <Cell key={metric} mono>
                    {run.metrics[metric] === null ? (
                      <NonFinite />
                    ) : (
                      (run.metrics[metric] ?? <Missing />)
                    )}
                  </Cell>
                ))}
                <Cell mono>{run.start_time}</Cell>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/**
 * APP-04: runs reales de MLflow (vía ml-api), sin caché ni filas propias. La
 * comparación queda en la URL (`?runs=`), así que refrescar muestra lo mismo.
 */
export function ExperimentsPage() {
  const runs = useExperimentRuns();
  const { refresh } = runs;
  const [searchParams, setSearchParams] = useSearchParams();
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("");
  const [dataset, setDataset] = useState("");
  const [version, setVersion] = useState(0);

  const allRuns = runs.status === "success" ? runs.data.runs : [];
  const anyActive = allRuns.some(isActive);
  const slots = parseSlots(searchParams.get("runs"));

  useEffect(() => {
    if (!anyActive) return;
    const timer = setInterval(() => {
      refresh();
      setVersion((v) => v + 1);
    }, RUNS_POLL_MS);
    return () => clearInterval(timer);
  }, [anyActive, refresh]);

  const toggle = (runId: string) => {
    const next = [...slots];
    const current = next.indexOf(runId);
    if (current >= 0) {
      next[current] = null;
    } else {
      const free = next.indexOf(null);
      if (free < 0) return;
      next[free] = runId;
    }
    const serialized = serializeSlots(next);
    setSearchParams(serialized ? { runs: serialized } : {});
  };

  const selected: SlottedRun[] = slots.flatMap((id, index) => {
    const run = allRuns.find((r) => r.run_id === id);
    return run ? [{ slot: index + 1, run }] : [];
  });

  const needle = query.trim().toLowerCase();
  const visible = allRuns.filter(
    (run) =>
      (!status || run.status === status) &&
      (!dataset || run.dataset_version === dataset) &&
      (!needle ||
        [run.run_name, run.run_id, run.dataset_version, run.git_commit ?? ""].some((value) =>
          value.toLowerCase().includes(needle)
        ))
  );
  const statuses = [...new Set(allRuns.map((run) => run.status))].sort();
  const datasets = [...new Set(allRuns.map((run) => run.dataset_version))].sort();

  return (
    <MlPage
      title="Experiments"
      subtitle="Runs registrados en MLflow con sus parámetros y métricas."
    >
      <MlResourceBoundary state={runs} emptyMessage="Todavía no hay runs registrados en MLflow.">
        {() => (
          <>
            <div className="flex flex-wrap items-end gap-3">
              <label className="flex flex-col gap-1 text-xs text-ink-muted">
                Buscar
                <input
                  type="search"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Nombre, run_id, dataset o commit"
                  className="w-64 rounded-lg border border-border bg-surface px-3 py-2 text-sm text-ink"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs text-ink-muted">
                Estado
                <select
                  value={status}
                  onChange={(event) => setStatus(event.target.value)}
                  className="rounded-lg border border-border bg-surface px-3 py-2 text-sm text-ink"
                >
                  <option value="">Todos</option>
                  {statuses.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1 text-xs text-ink-muted">
                Dataset
                <select
                  value={dataset}
                  onChange={(event) => setDataset(event.target.value)}
                  className="rounded-lg border border-border bg-surface px-3 py-2 text-sm text-ink"
                >
                  <option value="">Todos</option>
                  {datasets.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
              <p className="ml-auto text-xs text-ink-muted">
                {`Marca hasta ${MAX_COMPARED} runs para comparar sus parámetros y curvas.`}
              </p>
            </div>
            {visible.length === 0 ? (
              <p className="rounded-2xl border border-border bg-surface px-6 py-10 text-center text-sm text-ink-muted">
                Ningún run coincide con el filtro.
              </p>
            ) : (
              <RunsTable runs={visible} allRuns={allRuns} slots={slots} onToggle={toggle} />
            )}
            {selected.length > 0 && <RunComparison selected={selected} version={version} />}
          </>
        )}
      </MlResourceBoundary>
    </MlPage>
  );
}
