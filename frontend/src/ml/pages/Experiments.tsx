import { Cell, DataTable, Missing } from "../components/DataTable";
import { MlPage } from "../components/MlPage";
import { MlResourceBoundary } from "../components/MlResourceBoundary";
import { useExperimentRuns } from "../dataSource";

export function ExperimentsPage() {
  const runs = useExperimentRuns();

  return (
    <MlPage
      title="Experiments"
      subtitle="Runs registrados en MLflow con sus parámetros y métricas."
    >
      <MlResourceBoundary state={runs} emptyMessage="Todavía no hay runs registrados en MLflow.">
        {(response) => (
          <DataTable
            caption="Runs de MLflow"
            headers={["Run", "Run ID", "Estado", "Experimento", "Dataset", "Métricas", "Inicio"]}
          >
            {response.runs.map((run) => {
              const metrics = Object.entries(run.metrics);
              return (
                <tr key={run.run_id}>
                  <Cell>{run.run_name}</Cell>
                  <Cell mono>{run.run_id}</Cell>
                  <Cell>{run.status}</Cell>
                  <Cell mono>{run.experiment_id}</Cell>
                  <Cell mono>{run.dataset_version}</Cell>
                  <Cell>
                    {metrics.length === 0 ? (
                      <Missing />
                    ) : (
                      <ul>
                        {metrics.map(([name, value]) => (
                          <li key={name} className="font-mono text-xs">
                            {`${name}: ${value}`}
                          </li>
                        ))}
                      </ul>
                    )}
                  </Cell>
                  <Cell mono>{run.start_time}</Cell>
                </tr>
              );
            })}
          </DataTable>
        )}
      </MlResourceBoundary>
    </MlPage>
  );
}
