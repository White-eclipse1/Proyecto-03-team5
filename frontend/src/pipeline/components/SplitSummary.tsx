import { StatCard } from "@/components/dashboard/StatCard";
import type { SplitsReport } from "../schemas";

export function SplitSummary({ report }: Readonly<{ report: SplitsReport }>) {
  const distribution = Object.entries(report.class_distribution);
  const categories = [...new Set(distribution.flatMap(([, classes]) => Object.keys(classes)))];
  const leakage = report.leakage;
  return (
    <section aria-label={`Splits de ${report.dataset_version}`} className="space-y-5">
      <p className="mb-4">
        {report.total_images} imágenes · {report.dataset_version}
      </p>
      <div className="grid gap-4 sm:grid-cols-3">
        {(["train", "validation", "test"] as const).map((name) => (
          <StatCard
            key={name}
            label={`${name} (${Number((report.splits[name].ratio * 100).toFixed(4))}%)`}
            value={report.splits[name].image_count}
            accent="lilac"
          />
        ))}
      </div>
      {distribution.length > 0 && (
        <div className="overflow-x-auto rounded-2xl border border-border bg-surface">
          <table className="w-full text-left text-sm">
            <caption className="px-5 py-3 text-left font-semibold text-ink">
              Distribución por clase
            </caption>
            <thead className="border-y border-border bg-sidebar text-xs uppercase text-ink-muted">
              <tr>
                <th className="px-5 py-3 font-medium">Clase</th>
                {Object.keys(report.splits).map((split) => (
                  <th key={split} className="px-5 py-3 font-medium">
                    {split}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {categories.map((category) => (
                <tr key={category}>
                  <th className="px-5 py-3 font-medium">{category}</th>
                  {Object.keys(report.splits).map((split) => (
                    <td key={split} className="px-5 py-3 text-ink-muted">
                      {report.class_distribution[split]?.[category] ?? 0}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {Object.keys(leakage).length > 0 && (
        <div className="rounded-2xl border border-border bg-surface p-5 text-sm">
          <h3 className="font-semibold text-ink">Leakage entre splits</h3>
          <p className="mt-2 text-ink-muted">
            Estado: {leakage.status ?? "no disponible"} · grupos cruzados:{" "}
            {leakage.cross_split_groups ?? "—"} · cobertura: {leakage.coverage ?? "—"}
          </p>
        </div>
      )}
    </section>
  );
}
