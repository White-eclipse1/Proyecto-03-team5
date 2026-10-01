import { Button } from "@/components/ui/Button";
import { useTrainingJobLogs } from "../dataSource";
import type { TrainingLogEntry } from "../schemas";

const LEVEL_CLASSES: Record<TrainingLogEntry["level"], string> = {
  info: "text-ink-muted",
  warning: "text-amber-600",
  error: "text-red-600",
};

/** APP-03: logs de un job, leídos de la API (persisten en MariaDB, no en el navegador). */
export function TrainingJobLogs({
  jobId,
  active,
  pollMs,
  onClose,
}: Readonly<{ jobId: string; active: boolean; pollMs: number; onClose: () => void }>) {
  const logs = useTrainingJobLogs(jobId, active, pollMs);
  const headingId = `logs-${jobId}`;

  return (
    <section
      aria-labelledby={headingId}
      className="flex flex-col gap-3 rounded-2xl border border-border bg-surface p-4 shadow-card"
    >
      <div className="flex items-center justify-between gap-2">
        <h2 id={headingId} className="text-lg font-semibold">
          {`Logs de ${jobId}`}
        </h2>
        <Button variant="ghost" size="sm" onClick={onClose}>
          Cerrar
        </Button>
      </div>

      {logs.error && (
        <p role="alert" className="text-sm text-red-600">
          {logs.error.message}
        </p>
      )}
      {!logs.loaded && <p className="text-sm text-ink-muted">Cargando logs…</p>}
      {logs.loaded && !logs.error && logs.entries.length === 0 && (
        <p className="text-sm text-ink-muted">Este job todavía no tiene logs.</p>
      )}

      {logs.entries.length > 0 && (
        <ol className="max-h-80 overflow-y-auto rounded-lg bg-sidebar p-3 font-mono text-xs">
          {logs.entries.map((entry) => (
            <li key={entry.seq} className="flex gap-3 py-0.5">
              <span className="shrink-0 text-ink-faint">{entry.timestamp}</span>
              <span className={`shrink-0 uppercase ${LEVEL_CLASSES[entry.level]}`}>
                {entry.level}
              </span>
              <span className="whitespace-pre-wrap break-words text-ink">{entry.message}</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
