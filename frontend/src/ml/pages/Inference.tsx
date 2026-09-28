import { Cell, DataTable } from "../components/DataTable";
import { MlPage } from "../components/MlPage";
import { MlResourceBoundary } from "../components/MlResourceBoundary";
import { useReadyModels } from "../dataSource";
import type { InferenceResponse } from "../schemas";

/**
 * Resultado de una inferencia (contrato `InferenceResponse`). La petición real
 * todavía no está conectada; este componente es el consumidor del contrato y
 * se renderiza cuando exista una respuesta validada.
 */
export function InferenceResult({ response }: Readonly<{ response: InferenceResponse }>) {
  return (
    <section className="flex flex-col gap-3">
      <h2 className="text-lg font-semibold">Resultado</h2>
      <dl className="grid gap-2 text-sm sm:grid-cols-2">
        <div>
          <dt className="text-ink-muted">Modelo</dt>
          <dd className="font-mono">{`${response.model_name} v${response.model_version}`}</dd>
        </div>
        <div>
          <dt className="text-ink-muted">Dataset de entrenamiento</dt>
          <dd className="font-mono">{response.dataset_version}</dd>
        </div>
        <div>
          <dt className="text-ink-muted">Run MLflow</dt>
          <dd className="font-mono">{response.run_id}</dd>
        </div>
        <div>
          <dt className="text-ink-muted">Latencia</dt>
          <dd className="font-mono">{`${response.latency_ms} ms`}</dd>
        </div>
      </dl>
      <DataTable caption="Predicciones" headers={["Categoría", "Score", "BBox [x, y, w, h]"]}>
        {response.predictions.map((prediction) => (
          <tr key={`${prediction.category_name}:${prediction.bbox.join(",")}`}>
            <Cell>{prediction.category_name}</Cell>
            <Cell mono>{`${(prediction.score * 100).toFixed(1)}%`}</Cell>
            <Cell mono>{`[${prediction.bbox.join(", ")}]`}</Cell>
          </tr>
        ))}
      </DataTable>
    </section>
  );
}

export function InferencePage() {
  const models = useReadyModels();

  return (
    <MlPage title="Inference" subtitle="Modelos READY del registry disponibles para inferencia.">
      <MlResourceBoundary
        state={models}
        emptyMessage="No hay modelos READY disponibles para inferencia."
      >
        {(response) => (
          <DataTable
            caption="Modelos disponibles para inferencia"
            headers={["Modelo", "Aliases", "Dataset", "Run MLflow"]}
          >
            {response.models
              .filter((model) => model.status === "READY")
              .map((model) => (
                <tr key={`${model.model_name}:${model.model_version}`}>
                  <Cell mono>{`${model.model_name} v${model.model_version}`}</Cell>
                  <Cell>{model.aliases.join(", ")}</Cell>
                  <Cell mono>{model.dataset_version}</Cell>
                  <Cell mono>{model.run_id}</Cell>
                </tr>
              ))}
          </DataTable>
        )}
      </MlResourceBoundary>
    </MlPage>
  );
}
