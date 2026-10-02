import { Cell, DataTable } from "../components/DataTable";
import { MlPage } from "../components/MlPage";
import { MlResourceBoundary } from "../components/MlResourceBoundary";
import { useRegisteredModels } from "../dataSource";

export function ModelsPage() {
  const models = useRegisteredModels();

  return (
    <MlPage
      title="Models"
      subtitle="Versiones del Model Registry y el dataset con el que se entrenó cada una."
    >
      <MlResourceBoundary state={models} emptyMessage="Todavía no hay modelos registrados.">
        {(response) => (
          <DataTable
            caption="Versiones de modelos registradas"
            headers={["Modelo", "Versión", "S3", "Dataset", "Manifest", "Run MLflow"]}
          >
            {response.models.map((model) => (
              <tr key={`${model.model_name}:${model.model_version}`}>
                <Cell mono>{model.model_name}</Cell>
                <Cell mono>{model.model_version}</Cell>
                <Cell>{model.publication.status}</Cell>
                <Cell mono>{model.dataset_version}</Cell>
                <Cell mono>{model.manifest_hash}</Cell>
                <Cell mono>{model.run_id}</Cell>
              </tr>
            ))}
          </DataTable>
        )}
      </MlResourceBoundary>
    </MlPage>
  );
}
