import { Cell, DataTable, Missing } from "../components/DataTable";
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
            headers={[
              "Modelo",
              "Versión",
              "Estado",
              "Aliases",
              "Dataset",
              "Manifest",
              "Run MLflow",
            ]}
          >
            {response.models.map((model) => (
              <tr key={`${model.model_name}:${model.model_version}`}>
                <Cell mono>{model.model_name}</Cell>
                <Cell mono>{model.model_version}</Cell>
                <Cell>{model.status}</Cell>
                <Cell>
                  {model.aliases.length === 0 ? (
                    <Missing />
                  ) : (
                    <span className="flex gap-1">
                      {model.aliases.map((alias) => (
                        <span
                          key={alias}
                          className="rounded-full bg-accent-lilac-soft px-2 py-0.5 text-xs text-accent-lilac"
                        >
                          {alias}
                        </span>
                      ))}
                    </span>
                  )}
                </Cell>
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
