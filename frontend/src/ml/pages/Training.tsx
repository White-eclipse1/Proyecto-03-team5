import { Cell, DataTable, Missing } from "../components/DataTable";
import { MlPage } from "../components/MlPage";
import { MlResourceBoundary } from "../components/MlResourceBoundary";
import { useTrainingJobs } from "../dataSource";

export function TrainingPage() {
  const jobs = useTrainingJobs();

  return (
    <MlPage title="Training" subtitle="Jobs de entrenamiento y su trazabilidad dataset → run.">
      <MlResourceBoundary state={jobs} emptyMessage="Todavía no hay jobs de entrenamiento.">
        {(response) => (
          <DataTable
            caption="Jobs de entrenamiento"
            headers={[
              "Job",
              "Estado",
              "Dataset",
              "Manifest",
              "Run MLflow",
              "Checkpoint",
              "Creado",
              "Error",
            ]}
          >
            {response.jobs.map((job) => (
              <tr key={job.job_id}>
                <Cell mono>{job.job_id}</Cell>
                <Cell>{job.status}</Cell>
                <Cell mono>{job.dataset_version}</Cell>
                <Cell mono>{job.manifest_hash}</Cell>
                <Cell mono>{job.run_id ?? <Missing />}</Cell>
                <Cell mono>{job.checkpoint ?? <Missing />}</Cell>
                <Cell mono>{job.created_at}</Cell>
                <Cell>{job.error ? job.error.message : <Missing />}</Cell>
              </tr>
            ))}
          </DataTable>
        )}
      </MlResourceBoundary>
    </MlPage>
  );
}
