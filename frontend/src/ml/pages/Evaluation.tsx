import { Cell, DataTable } from "../components/DataTable";
import { MlPage } from "../components/MlPage";
import { MlResourceBoundary } from "../components/MlResourceBoundary";
import { useEvaluations } from "../dataSource";

const metric = (value: number) => value.toFixed(3);

export function EvaluationPage() {
  const evaluations = useEvaluations();

  return (
    <MlPage title="Evaluation" subtitle="Métricas de cada checkpoint sobre validation o test.">
      <MlResourceBoundary state={evaluations} emptyMessage="Todavía no hay evaluaciones.">
        {(response) => (
          <DataTable
            caption="Evaluaciones"
            headers={[
              "Evaluación",
              "Modelo",
              "Dataset",
              "Split",
              "mAP50",
              "mAP50-95",
              "Precision",
              "Recall",
              "Run MLflow",
            ]}
          >
            {response.evaluations.map((evaluation) => (
              <tr key={evaluation.evaluation_id}>
                <Cell mono>{evaluation.evaluation_id}</Cell>
                <Cell>
                  {evaluation.model_name === null ? (
                    <span className="text-ink-muted">Sin registrar</span>
                  ) : (
                    `${evaluation.model_name} v${evaluation.model_version}`
                  )}
                </Cell>
                <Cell mono>{evaluation.dataset_version}</Cell>
                <Cell>{evaluation.split}</Cell>
                <Cell mono>{metric(evaluation.metrics.map50)}</Cell>
                <Cell mono>{metric(evaluation.metrics.map50_95)}</Cell>
                <Cell mono>{metric(evaluation.metrics.precision)}</Cell>
                <Cell mono>{metric(evaluation.metrics.recall)}</Cell>
                <Cell mono>{evaluation.run_id}</Cell>
              </tr>
            ))}
          </DataTable>
        )}
      </MlResourceBoundary>
    </MlPage>
  );
}
