import { useState } from "react";
import { PageHeader } from "../components/PageHeader";
import { ReportBoundary } from "../components/ReportBoundary";
import { SplitSummary } from "../components/SplitSummary";
import { StatusBadge } from "../components/StatusBadge";
import { useReleaseReport, useVersionsReport } from "../dataSource";
import type { DatasetRelease, QualityReport } from "../schemas";

function categoryCounts(report: QualityReport): Record<string, number> {
  const check = report.checks.find((entry) => entry.check_name === "max_imbalance_ratio");
  if (!check || !Array.isArray(check.details.images_per_category)) return {};
  return Object.fromEntries(
    check.details.images_per_category.flatMap((entry) => {
      if (!entry || typeof entry !== "object") return [];
      const value = entry as Record<string, unknown>;
      return typeof value.category_name === "string" && typeof value.image_count === "number"
        ? [[value.category_name, value.image_count]]
        : [];
    })
  );
}

function annotationCount(report: QualityReport) {
  const check = report.checks.find((entry) => entry.check_name === "degenerate_boxes");
  return typeof check?.details.total_annotations === "number" ? check.details.total_annotations : 0;
}

function classesCrossingMinimum(before: QualityReport, after: QualityReport) {
  const beforeCheck = before.checks.find((entry) => entry.check_name === "min_images_per_class");
  const afterCheck = after.checks.find((entry) => entry.check_name === "min_images_per_class");
  const beforeThreshold = beforeCheck?.details.criterion;
  const afterThreshold = afterCheck?.details.criterion;
  if (
    !beforeThreshold ||
    typeof beforeThreshold !== "object" ||
    !afterThreshold ||
    typeof afterThreshold !== "object"
  ) {
    return { entered: [], left: [] };
  }
  const threshold = (beforeThreshold as Record<string, unknown>).threshold;
  if (typeof threshold !== "number") return { entered: [], left: [] };
  const beforeCounts = categoryCounts(before);
  const afterCounts = categoryCounts(after);
  const entered: string[] = [];
  const left: string[] = [];
  for (const category of new Set([...Object.keys(beforeCounts), ...Object.keys(afterCounts)])) {
    const wasPassing = (beforeCounts[category] ?? 0) >= threshold;
    const isPassing = (afterCounts[category] ?? 0) >= threshold;
    if (!wasPassing && isPassing) entered.push(category);
    if (wasPassing && !isPassing) left.push(category);
  }
  return { entered, left };
}

function ReleaseComparisonContent({
  releases,
  firstRelease,
}: Readonly<{ releases: DatasetRelease[]; firstRelease: DatasetRelease }>) {
  const [fromVersion, setFromVersion] = useState(firstRelease.dataset_version);
  const [toVersion, setToVersion] = useState(releases[1]?.dataset_version ?? "");
  const from = releases.find((release) => release.dataset_version === fromVersion) ?? firstRelease;
  const to =
    releases.find((release) => release.dataset_version === toVersion) ??
    releases[1] ??
    firstRelease;
  const fromQuality = useReleaseReport("quality", from);
  const toQuality = useReleaseReport("quality", to);

  return (
    <section className="mt-6 rounded-2xl border border-border bg-surface p-5 shadow-card">
      <h2 className="text-lg font-semibold">Comparar releases</h2>
      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <label>
          Desde
          <select
            className="mt-1 block"
            value={from.dataset_version}
            onChange={(event) => setFromVersion(event.target.value)}
          >
            {releases.map((release) => (
              <option key={release.dataset_version} value={release.dataset_version}>
                {release.dataset_version}
              </option>
            ))}
          </select>
        </label>
        <label>
          Hasta
          <select
            className="mt-1 block"
            value={to.dataset_version}
            onChange={(event) => setToVersion(event.target.value)}
          >
            {releases.map((release) => (
              <option key={release.dataset_version} value={release.dataset_version}>
                {release.dataset_version}
              </option>
            ))}
          </select>
        </label>
      </div>
      <ReportBoundary state={fromQuality}>
        {(before) => (
          <ReportBoundary state={toQuality}>
            {(after) => {
              const beforeCounts = categoryCounts(before);
              const afterCounts = categoryCounts(after);
              const categories = [
                ...new Set([...Object.keys(beforeCounts), ...Object.keys(afterCounts)]),
              ];
              const crossing = classesCrossingMinimum(before, after);
              const checkNames = [
                ...new Set([
                  ...before.checks.map((check) => check.check_name),
                  ...after.checks.map((check) => check.check_name),
                ]),
              ];
              return (
                <div className="mt-5 space-y-4">
                  <p className="text-sm text-ink-muted">
                    Cajas: {annotationCount(before)} → {annotationCount(after)} (delta{" "}
                    {annotationCount(after) - annotationCount(before)})
                  </p>
                  <table className="w-full text-left text-sm">
                    <thead className="border-y border-border bg-sidebar text-xs uppercase text-ink-muted">
                      <tr>
                        <th className="px-4 py-2">Clase</th>
                        <th className="px-4 py-2">Desde</th>
                        <th className="px-4 py-2">Hasta</th>
                        <th className="px-4 py-2">Delta</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {categories.map((category) => (
                        <tr key={category}>
                          <th className="px-4 py-2 font-medium">{category}</th>
                          <td className="px-4 py-2">{beforeCounts[category] ?? 0}</td>
                          <td className="px-4 py-2">{afterCounts[category] ?? 0}</td>
                          <td className="px-4 py-2">
                            {(afterCounts[category] ?? 0) - (beforeCounts[category] ?? 0)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="text-sm text-ink-muted">
                    Cruzan el mínimo: entran {crossing.entered.join(", ") || "ninguna"}; salen{" "}
                    {crossing.left.join(", ") || "ninguna"}.
                  </p>
                  <table className="w-full text-left text-sm">
                    <caption className="px-4 py-2 text-left font-semibold text-ink">
                      Variación de checks
                    </caption>
                    <tbody className="divide-y divide-border">
                      {checkNames.map((name) => {
                        const beforeCheck = before.checks.find(
                          (check) => check.check_name === name
                        );
                        const afterCheck = after.checks.find((check) => check.check_name === name);
                        const beforeMetric = beforeCheck?.metric_value;
                        const afterMetric = afterCheck?.metric_value;
                        return (
                          <tr key={name}>
                            <th className="px-4 py-2 font-medium">{name}</th>
                            <td className="px-4 py-2">
                              {beforeCheck?.passed === undefined
                                ? "—"
                                : beforeCheck.passed
                                  ? "passed"
                                  : "failed"}
                            </td>
                            <td className="px-4 py-2">
                              {afterCheck?.passed === undefined
                                ? "—"
                                : afterCheck.passed
                                  ? "passed"
                                  : "failed"}
                            </td>
                            <td className="px-4 py-2">
                              {typeof beforeMetric === "number" && typeof afterMetric === "number"
                                ? `${beforeMetric} → ${afterMetric} (Δ ${afterMetric - beforeMetric})`
                                : "—"}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  <p className="text-sm text-ink-muted">
                    DEV: requiere dvc status -r dev · PROD: requiere dvc status -r prod. El catálogo
                    conserva los reportes publicados, no una sesión AWS.
                  </p>
                </div>
              );
            }}
          </ReportBoundary>
        )}
      </ReportBoundary>
    </section>
  );
}

function ReleaseComparison({ releases }: Readonly<{ releases: DatasetRelease[] }>) {
  const firstRelease = releases.at(0);
  return firstRelease ? (
    <ReleaseComparisonContent releases={releases} firstRelease={firstRelease} />
  ) : null;
}

function ReleaseCard({ release }: Readonly<{ release: DatasetRelease }>) {
  const quality = useReleaseReport("quality", release);
  const splits = useReleaseReport("splits", release);
  return (
    <li className="rounded-2xl border border-border bg-surface p-5 space-y-4">
      <h2 className="font-semibold">{release.dataset_version}</h2>
      <ReportBoundary state={quality}>
        {(report) => (
          <p>
            Calidad: <StatusBadge label={report.status} />
          </p>
        )}
      </ReportBoundary>
      <ReportBoundary state={splits}>{(report) => <SplitSummary report={report} />}</ReportBoundary>
      <p className="flex gap-4">
        <a href={`/reports/${release.quality_file}`}>Reporte de calidad</a>
        <a href={`/reports/${release.splits_file}`}>Reporte de splits</a>
      </p>
    </li>
  );
}

export function VersionsPage() {
  const versions = useVersionsReport();
  return (
    <main className="flex-1 px-6 py-6 lg:px-10">
      <PageHeader
        title="Versions"
        subtitle="Catálogo de releases publicadas; sin orden cronológico definido"
      />
      <ReportBoundary state={versions}>
        {(catalog) =>
          catalog.releases.length === 0 ? (
            <p>Todavía no hay releases publicadas.</p>
          ) : (
            <>
              {catalog.releases.length > 1 && <ReleaseComparison releases={catalog.releases} />}
              <ul className="mt-6 space-y-4">
                {catalog.releases.map((release) => (
                  <ReleaseCard key={release.dataset_version} release={release} />
                ))}
              </ul>
            </>
          )
        }
      </ReportBoundary>
    </main>
  );
}
