import type { TrainingJob } from "../schemas";
import { Missing } from "./DataTable";

/** Métrica que se muestra junto a la barra, en orden de preferencia. */
const HEADLINE_METRICS = ["val_accuracy", "val_loss", "train_loss"];

function headlineMetric(metrics: Record<string, number | null>): string | null {
  const name = HEADLINE_METRICS.find((key) => key in metrics) ?? Object.keys(metrics)[0];
  if (name === undefined) return null;
  // null: el worker reportó NaN o ±inf; se dice tal cual, sin inventar un número.
  const value = metrics[name];
  return `${name} ${value === null ? "no finito" : value}`;
}

/**
 * APP-03: última época que reportó el worker. Un job sin `progress` (queued o
 * recién iniciado) no muestra barra: nada de progreso inventado.
 */
export function JobProgress({ job }: Readonly<{ job: TrainingJob }>) {
  const { progress } = job;
  if (progress === null) return <Missing />;

  const percent = Math.round((progress.epoch / progress.max_epochs) * 100);
  const metric = headlineMetric(progress.metrics);

  return (
    <div className="flex min-w-36 flex-col gap-1">
      <div
        role="progressbar"
        aria-label={`Progreso de ${job.job_id}`}
        aria-valuemin={0}
        aria-valuemax={progress.max_epochs}
        aria-valuenow={progress.epoch}
        className="h-1.5 w-full overflow-hidden rounded-full bg-border"
      >
        <div className="h-full rounded-full bg-accent-lilac" style={{ width: `${percent}%` }} />
      </div>
      <span className="text-xs text-ink">{`Época ${progress.epoch}/${progress.max_epochs}`}</span>
      {metric && <span className="font-mono text-xs text-ink-muted">{metric}</span>}
    </div>
  );
}
