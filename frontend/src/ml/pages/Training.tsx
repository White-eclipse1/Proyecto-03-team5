import { useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { Button } from "@/components/ui/Button";
import { Cell, DataTable, Missing } from "../components/DataTable";
import { JobProgress } from "../components/JobProgress";
import { MlPage } from "../components/MlPage";
import { MlResourceBoundary } from "../components/MlResourceBoundary";
import { TrainingForm } from "../components/TrainingForm";
import { TrainingJobLogs } from "../components/TrainingJobLogs";
import { useTrainingJobs } from "../dataSource";
import type { TrainingJob } from "../schemas";

/** Cada cuánto se vuelven a pedir jobs y logs mientras haya alguno activo. */
export const JOBS_POLL_MS = 3000;

const isActive = (job: TrainingJob) => job.status === "queued" || job.status === "running";

/**
 * APP-03: el entrenamiento corre en el worker, fuera del request HTTP. El estado
 * vive en el servidor, así que refrescar la página muestra lo mismo; el job cuyos
 * logs se están viendo queda en la URL (`?job=`).
 */
export function TrainingPage() {
  const jobs = useTrainingJobs();
  const { refresh } = jobs;
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedId = searchParams.get("job");

  const loadedJobs = jobs.status === "success" ? jobs.data.jobs : [];
  const anyActive = loadedJobs.some(isActive);
  const selectedJob = loadedJobs.find((job) => job.job_id === selectedId);

  useEffect(() => {
    if (!anyActive) return;
    const timer = setInterval(refresh, JOBS_POLL_MS);
    return () => clearInterval(timer);
  }, [anyActive, refresh]);

  return (
    <MlPage
      title="Training"
      subtitle="Configura un entrenamiento sobre un release aprobado del Proyecto 2 y sigue sus jobs."
    >
      <TrainingForm onCreated={jobs.reload} />
      <h2 className="text-lg font-semibold">Jobs de entrenamiento</h2>
      <MlResourceBoundary state={jobs} emptyMessage="Todavía no hay jobs de entrenamiento.">
        {(response) => (
          <DataTable
            caption="Jobs de entrenamiento"
            headers={[
              "Job",
              "Estado",
              "Progreso",
              "Dataset",
              "Manifest",
              "Run MLflow",
              "Checkpoint",
              "Creado",
              "Error",
              "Logs",
            ]}
          >
            {response.jobs.map((job) => (
              <tr key={job.job_id}>
                <Cell mono>{job.job_id}</Cell>
                <Cell>{job.status}</Cell>
                <Cell>
                  <JobProgress job={job} />
                </Cell>
                <Cell mono>{job.dataset_version}</Cell>
                <Cell mono>{job.manifest_hash}</Cell>
                <Cell mono>{job.run_id ?? <Missing />}</Cell>
                <Cell mono>{job.checkpoint ?? <Missing />}</Cell>
                <Cell mono>{job.created_at}</Cell>
                <Cell>
                  {job.error ? (
                    <div className="flex flex-col gap-0.5">
                      <span className="font-mono text-xs text-red-600">{job.error.code}</span>
                      <span>{job.error.message}</span>
                    </div>
                  ) : (
                    <Missing />
                  )}
                </Cell>
                <Cell>
                  <Button
                    variant="secondary"
                    size="sm"
                    aria-pressed={job.job_id === selectedId}
                    onClick={() => setSearchParams({ job: job.job_id })}
                  >
                    Ver logs
                  </Button>
                </Cell>
              </tr>
            ))}
          </DataTable>
        )}
      </MlResourceBoundary>
      {selectedId && (
        <TrainingJobLogs
          key={selectedId}
          jobId={selectedId}
          active={selectedJob !== undefined && isActive(selectedJob)}
          pollMs={JOBS_POLL_MS}
          onClose={() => setSearchParams({})}
        />
      )}
    </MlPage>
  );
}
