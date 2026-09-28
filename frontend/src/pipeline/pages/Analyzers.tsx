import { CheckCriterion, CheckDetails } from "../components/CheckDetails";
import { PageHeader } from "../components/PageHeader";
import { ReportBoundary } from "../components/ReportBoundary";
import { StatusBadge } from "../components/StatusBadge";
import { useQualityReport } from "../dataSource";

export function AnalyzersPage() {
  const quality = useQualityReport();

  return (
    <main className="flex-1 px-6 py-6 lg:px-10 lg:py-8">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <ReportBoundary state={quality}>
          {(report) => (
            <>
              <PageHeader
                title="Analyzers"
                subtitle={`${report.checks.length} checks del dataset ${report.dataset_version}`}
              >
                <StatusBadge label={report.status} />
              </PageHeader>

              <div className="overflow-hidden rounded-2xl border border-border bg-surface shadow-card">
                <table className="w-full text-left text-sm">
                  <thead className="border-b border-border bg-sidebar text-xs uppercase text-ink-muted">
                    <tr>
                      <th className="px-5 py-3 font-medium">Check</th>
                      <th className="px-5 py-3 font-medium">Resultado</th>
                      <th className="px-5 py-3 font-medium">Métrica</th>
                      <th className="px-5 py-3 font-medium">Criterio</th>
                      <th className="px-5 py-3 font-medium">Acción si falla</th>
                      <th className="px-5 py-3 font-medium">Detalles</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {report.checks.map((check) => (
                      <tr key={check.check_name}>
                        <td className="px-5 py-3 font-medium text-ink">{check.check_name}</td>
                        <td className="px-5 py-3">
                          <StatusBadge label={check.passed ? "passed" : check.action} />
                        </td>
                        <td className="px-5 py-3 text-ink-muted">
                          {check.check_name === "max_imbalance_ratio" &&
                          check.details.ratio_defined === false
                            ? "Indefinido: hay categorías sin imágenes"
                            : check.metric_value}
                        </td>
                        <td className="px-5 py-3">
                          <CheckCriterion check={check} />
                        </td>
                        <td className="px-5 py-3 text-ink-muted">{check.action}</td>
                        <td className="px-5 py-3">
                          <CheckDetails check={check} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </ReportBoundary>
      </div>
    </main>
  );
}
