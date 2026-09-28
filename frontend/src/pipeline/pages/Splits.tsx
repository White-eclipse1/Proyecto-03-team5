import { useState } from "react";
import { PageHeader } from "../components/PageHeader";
import { ReportBoundary } from "../components/ReportBoundary";
import { SplitSummary } from "../components/SplitSummary";
import { useReleaseReport, useVersionsReport } from "../dataSource";
import type { DatasetRelease } from "../schemas";

function SelectedSplits({ release }: Readonly<{ release: DatasetRelease }>) {
  const state = useReleaseReport("splits", release);
  return (
    <ReportBoundary state={state}>{(report) => <SplitSummary report={report} />}</ReportBoundary>
  );
}

export function SplitsPage() {
  const versions = useVersionsReport();
  const [selected, setSelected] = useState("");
  return (
    <main className="flex-1 px-6 py-6 lg:px-10">
      <PageHeader title="Splits" subtitle="Conjuntos generados por release" />
      <ReportBoundary state={versions}>
        {(catalog) => {
          const release = catalog.releases.find((entry) => entry.dataset_version === selected);
          return (
            <div className="mt-6 space-y-6">
              {catalog.releases.length === 0 ? (
                <p>No hay releases publicadas.</p>
              ) : (
                <>
                  <label htmlFor="split-release">Release</label>
                  <select
                    id="split-release"
                    value={selected}
                    onChange={(event) => setSelected(event.target.value)}
                  >
                    <option value="">Selecciona una release</option>
                    {catalog.releases.map((entry) => (
                      <option key={entry.dataset_version} value={entry.dataset_version}>
                        {entry.dataset_version}
                      </option>
                    ))}
                  </select>
                  {release && <SelectedSplits key={release.dataset_version} release={release} />}
                </>
              )}
            </div>
          );
        }}
      </ReportBoundary>
    </main>
  );
}
