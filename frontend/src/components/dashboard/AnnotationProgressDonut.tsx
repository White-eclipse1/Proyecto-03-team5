import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";
import type { AnnotationProgress } from "../../types/dashboard";

type AnnotationProgressDonutProps = {
  progress: AnnotationProgress;
};

const COLORS = {
  annotated: "#2FAF87", // accent-mint
  pending: "#E7E5E1", // border
} as const;

const segments = [
  { key: "annotated", label: "Anotadas" },
  { key: "pending", label: "Pendientes" },
] as const;

export function AnnotationProgressDonut({ progress }: AnnotationProgressDonutProps) {
  const total = progress.annotated + progress.pending;

  if (total === 0) {
    return (
      <div className="flex h-72 items-center justify-center rounded-2xl border border-border bg-surface text-sm text-ink-faint">
        Aún no hay imágenes para anotar.
      </div>
    );
  }

  const data = segments.map((segment) => ({ ...segment, value: progress[segment.key] }));
  const annotatedPct = Math.round((progress.annotated / total) * 100);

  return (
    <div className="rounded-2xl border border-border bg-surface p-5 shadow-card">
      <h3 className="text-sm font-medium text-ink">Progreso de anotación</h3>
      <div className="mt-3 flex items-center gap-6">
        <div className="relative h-40 w-40 shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                dataKey="value"
                nameKey="label"
                innerRadius={48}
                outerRadius={70}
                paddingAngle={2}
                stroke="none"
              >
                {data.map((entry) => (
                  <Cell key={entry.key} fill={COLORS[entry.key]} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
          <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-2xl font-semibold text-ink">{annotatedPct}%</span>
            <span className="text-xs text-ink-faint">anotado</span>
          </div>
        </div>
        <ul className="space-y-3 text-sm">
          {data.map((entry) => (
            <li key={entry.key} className="flex items-center gap-2">
              <span
                className="h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: COLORS[entry.key] }}
                aria-hidden="true"
              />
              <span className="text-ink-muted">{entry.label}</span>
              <span className="ml-auto font-medium text-ink">{entry.value}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
