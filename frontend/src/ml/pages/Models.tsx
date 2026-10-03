import { useId, useState } from "react";
import { Link } from "react-router-dom";
import { Cell, DataTable } from "../components/DataTable";
import { MlPage } from "../components/MlPage";
import { MlResourceBoundary } from "../components/MlResourceBoundary";
import { modelFileUrl, useRegisteredModels } from "../dataSource";
import type { ModelPublication, RegisteredModelVersion } from "../schemas";

const PUBLICATION_LABEL: Record<ModelPublication["status"], string> = {
  published: "Publicado en S3",
  not_published: "No publicado",
  inconsistent: "Inconsistente",
  unverifiable: "No verificable",
};

/** Más nueva primero, comparando MAJOR.MINOR.PATCH como números (0.10.0 > 0.9.0). */
export function bySemverDesc(a: RegisteredModelVersion, b: RegisteredModelVersion): number {
  const parts = (version: string) => version.split(".").map(Number);
  const [left, right] = [parts(a.model_version), parts(b.model_version)];
  for (let index = 0; index < 3; index++) {
    const difference = (right[index] ?? 0) - (left[index] ?? 0);
    if (difference !== 0) return difference;
  }
  return 0;
}

const metric = (value: number) => value.toFixed(4);

/** En la tabla, `sha256:<64 hex>` se acorta; el detalle muestra el hash completo. */
const shortHash = (hash: string) =>
  hash.startsWith("sha256:") ? `${hash.slice(0, "sha256:".length + 12)}…` : hash;

function Field({ label, children }: Readonly<{ label: string; children: React.ReactNode }>) {
  return (
    <div>
      <dt className="text-xs uppercase text-ink-muted">{label}</dt>
      <dd className="mt-1 break-all text-sm text-ink">{children}</dd>
    </div>
  );
}

function Badge({ tone, children }: Readonly<{ tone: "ok" | "warn" | "muted"; children: string }>) {
  const colors = {
    ok: "bg-status-done-soft text-status-done",
    warn: "bg-status-pending-soft text-status-pending",
    muted: "bg-sidebar text-ink-muted",
  };
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${colors[tone]}`}>
      {children}
    </span>
  );
}

function PublicationBadge({ publication }: Readonly<{ publication: ModelPublication }>) {
  const tone = {
    published: "ok",
    not_published: "muted",
    inconsistent: "warn",
    unverifiable: "warn",
  } as const;
  return <Badge tone={tone[publication.status]}>{PUBLICATION_LABEL[publication.status]}</Badge>;
}

function VersionsTable({
  models,
  selected,
  onSelect,
}: Readonly<{
  models: RegisteredModelVersion[];
  selected: string;
  onSelect: (version: string) => void;
}>) {
  return (
    <DataTable
      caption="Versiones del modelo"
      headers={["Versión", "Release P2", "Run MLflow", "Manifest", "Paquete", "S3", ""]}
    >
      {models.map((model) => (
        <tr
          key={model.model_version}
          className={model.model_version === selected ? "bg-accent-lilac-soft" : undefined}
        >
          <Cell mono>{model.model_version}</Cell>
          <Cell mono>{model.dataset_version}</Cell>
          <Cell mono>{model.run_id}</Cell>
          <Cell mono>
            <span title={model.manifest_hash}>{shortHash(model.manifest_hash)}</span>
          </Cell>
          <Cell>
            <Badge tone={model.servable ? "ok" : "muted"}>
              {model.servable ? "Paquete disponible" : "Paquete no descargado"}
            </Badge>
          </Cell>
          <Cell>
            <PublicationBadge publication={model.publication} />
          </Cell>
          <Cell>
            <button
              type="button"
              aria-pressed={model.model_version === selected}
              className="text-sm font-medium text-accent-lilac underline-offset-2 hover:underline"
              onClick={() => onSelect(model.model_version)}
            >
              Ver {model.model_version}
            </button>
          </Cell>
        </tr>
      ))}
    </DataTable>
  );
}

function Publication({ publication }: Readonly<{ publication: ModelPublication }>) {
  if (publication.status === "not_published") {
    return <p className="text-sm text-ink-muted">Esta versión no está publicada en S3.</p>;
  }
  if (publication.status === "inconsistent") {
    return (
      <p role="alert" className="rounded-xl bg-status-pending-soft px-4 py-3 text-sm text-ink">
        <span className="font-medium text-status-pending">No se considera publicada: </span>
        {publication.problem}
      </p>
    );
  }
  if (publication.status === "unverifiable") {
    return (
      <p role="alert" className="rounded-xl bg-status-pending-soft px-4 py-3 text-sm text-ink">
        <span className="font-medium text-status-pending">No se pudo confirmar en S3: </span>
        {publication.problem}
      </p>
    );
  }
  return (
    <div className="flex flex-col gap-3">
      <dl className="grid gap-4 sm:grid-cols-3">
        <Field label="Bucket">
          <span className="font-mono text-xs">{publication.bucket}</span>
        </Field>
        <Field label="Región">
          <span className="font-mono text-xs">{publication.region}</span>
        </Field>
        <Field label="Publicado">
          <span className="font-mono text-xs">{publication.published_at}</span>
        </Field>
      </dl>
      <DataTable caption="Objetos en S3" headers={["Archivo", "Key", "VersionId", "sha256"]}>
        {publication.objects.map((object) => (
          <tr key={object.name}>
            <Cell mono>{object.name}</Cell>
            <Cell mono>{object.key}</Cell>
            <Cell mono>{object.version_id ?? "—"}</Cell>
            <Cell mono>{object.sha256}</Cell>
          </tr>
        ))}
      </DataTable>
      <p className="text-xs text-ink-muted">
        S3 confirmó cada objeto con su ChecksumSHA256 al responder esta página, y OPS-07 lo verificó
        además con una descarga.
      </p>
    </div>
  );
}

function VersionDetail({ model }: Readonly<{ model: RegisteredModelVersion }>) {
  const id = useId();
  const label = `${model.model_name} ${model.model_version}`;
  return (
    <section
      aria-labelledby={id}
      className="flex flex-col gap-6 rounded-2xl border border-border bg-surface p-6 shadow-card"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 id={id} className="text-base font-semibold text-ink">
          {label}
        </h2>
        {model.servable && (
          <Link
            to={`/ml/inference?model_version=${encodeURIComponent(model.model_version)}`}
            className="rounded-lg bg-ink px-3 py-1.5 text-sm font-medium text-surface"
          >
            Usar en Inference
          </Link>
        )}
      </div>

      <dl className="grid gap-4 sm:grid-cols-2">
        <Field label="Checkpoint">
          <span className="font-mono text-xs">{model.checkpoint}</span>
        </Field>
        <Field label="sha256 del checkpoint">
          <span className="font-mono text-xs">{model.checkpoint_sha256}</span>
        </Field>
        <Field label="Release P2">
          <span className="font-mono text-xs">{model.dataset_version}</span>
        </Field>
        <Field label="Manifest hash">
          <span className="font-mono text-xs">{model.manifest_hash}</span>
        </Field>
        <Field label="Arquitectura">
          <span className="font-mono text-xs">{`${model.architecture} · ${model.image_size}×${model.image_size}`}</span>
        </Field>
        <Field label="Run MLflow">
          <span className="font-mono text-xs">{model.run_id}</span>
        </Field>
        <Field label="Accuracy top-1 (test)">
          <span className="font-mono">{metric(model.test_metrics.accuracy_top1)}</span>
        </Field>
        <Field label="F1 macro (test)">
          <span className="font-mono">{metric(model.test_metrics.f1_macro)}</span>
        </Field>
      </dl>

      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-semibold text-ink">Model card</h3>
        <p className="text-sm text-ink">{model.model_card.purpose}</p>
        <ul className="list-disc pl-5 text-sm text-ink">
          {model.model_card.limitations.map((limitation) => (
            <li key={limitation}>{limitation}</li>
          ))}
        </ul>
        <p className="text-xs text-ink-muted">
          Pesos preentrenados:{" "}
          <span className="font-mono">{model.model_card.pretrained_weights ?? "ninguno"}</span>
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-semibold text-ink">Paquete</h3>
        <ul className="flex flex-col gap-1 text-sm">
          {model.files.map((file) => (
            <li key={file.name} className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-xs">{file.name}</span>
              {file.available ? (
                <a
                  href={modelFileUrl(model.model_version, file.name)}
                  download
                  aria-label={`Descargar ${file.name}`}
                  className="text-xs font-medium text-accent-lilac underline-offset-2 hover:underline"
                >
                  Descargar
                </a>
              ) : (
                <span className="text-xs text-ink-muted">No está en este servidor</span>
              )}
            </li>
          ))}
        </ul>
        {!model.servable && (
          <p className="text-xs text-ink-muted">
            Para usar esta versión, descarga su paquete con{" "}
            <span className="font-mono">dvc pull data/models.dvc</span>.
          </p>
        )}
      </div>

      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-semibold text-ink">Publicación en S3</h3>
        <Publication publication={model.publication} />
      </div>
    </section>
  );
}

function ModelsOverview({ models }: Readonly<{ models: RegisteredModelVersion[] }>) {
  const sorted = [...models].sort(bySemverDesc);
  const [selected, setSelected] = useState(sorted[0]?.model_version ?? "");
  const model = sorted.find((entry) => entry.model_version === selected) ?? sorted[0];
  return (
    <div className="flex flex-col gap-6">
      <VersionsTable models={sorted} selected={model?.model_version ?? ""} onSelect={setSelected} />
      {model && <VersionDetail key={model.model_version} model={model} />}
    </div>
  );
}

export function ModelsPage() {
  const models = useRegisteredModels();

  return (
    <MlPage
      title="Models"
      subtitle="Versiones del modelo (OPS-06), su paquete, su model card y su publicación en S3 (OPS-07)."
    >
      <MlResourceBoundary
        state={models}
        emptyMessage="Todavía no hay versiones registradas del modelo."
      >
        {(response) => <ModelsOverview models={response.models} />}
      </MlResourceBoundary>
    </MlPage>
  );
}
