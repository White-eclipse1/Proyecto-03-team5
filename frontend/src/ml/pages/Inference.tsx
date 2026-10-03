import { useEffect, useId, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { Cell, DataTable } from "../components/DataTable";
import { MlPage } from "../components/MlPage";
import { MlResourceBoundary } from "../components/MlResourceBoundary";
import {
  classifyImage,
  cropUrl,
  enqueueInferenceResult,
  type InferenceInput,
  useReadyModels,
} from "../dataSource";
import type { ContractError, InferenceResponse, RegisteredModelVersion } from "../schemas";

/** Igual que `MAX_UPLOAD_BYTES` de `training/inference.py`; el servidor vuelve a validar. */
export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/png", "image/jpeg"];

/** Validación inmediata en el navegador; `ml-api` decide por el contenido del archivo. */
export function uploadProblem(file: File): string | null {
  if (!ACCEPTED_TYPES.includes(file.type)) return "Solo se aceptan imágenes PNG o JPEG.";
  if (file.size > MAX_UPLOAD_BYTES) return "El archivo pesa más de 10 MB.";
  if (file.size === 0) return "El archivo está vacío.";
  return null;
}

function Field({ label, children }: Readonly<{ label: string; children: React.ReactNode }>) {
  return (
    <div>
      <dt className="text-ink-muted">{label}</dt>
      <dd className="break-all">{children}</dd>
    </div>
  );
}

/** Resultado de clasificar una imagen (contrato `InferenceResponse`). */
export function InferenceResult({ response }: Readonly<{ response: InferenceResponse }>) {
  const { crop, upload } = response;
  const ranked = Object.entries(response.probabilities).sort(([, a], [, b]) => b - a);
  const id = useId();

  return (
    <section aria-labelledby={id} className="flex flex-col gap-3">
      <h2 id={id} className="text-lg font-semibold">
        Resultado
      </h2>
      <dl className="grid gap-2 text-sm sm:grid-cols-2">
        <Field label="Clase predicha">
          <span className="font-semibold">{response.predicted_class}</span>
        </Field>
        <Field label={crop === null ? "Imagen subida" : "Recorte"}>
          <span className="font-mono">
            {crop === null
              ? upload?.filename
              : `${crop.dataset_version} · img ${crop.image_id} · ann ${crop.annotation_id}`}
          </span>
        </Field>
        <Field label="Modelo">
          <span className="font-mono">{`${response.model_name} v${response.model_version}`}</span>
        </Field>
        <Field label="Dataset de entrenamiento">
          <span className="font-mono">{response.dataset_version}</span>
        </Field>
        <Field label="Run MLflow">
          <span className="font-mono">{response.run_id}</span>
        </Field>
        <Field label="Checkpoint cargado">
          <span className="font-mono text-xs">{response.checkpoint}</span>
        </Field>
        <Field label="sha256 del checkpoint">
          <span className="font-mono text-xs">{response.checkpoint_sha256}</span>
        </Field>
        <Field label="Preprocesamiento">
          <span className="font-mono">{`evaluación, ${response.image_size}×${response.image_size}`}</span>
        </Field>
        <Field label="Latencia">
          <span className="font-mono">{`${response.latency_ms.toFixed(1)} ms`}</span>
        </Field>
      </dl>
      <DataTable caption="Probabilidades por clase" headers={["Clase", "Probabilidad"]}>
        {ranked.map(([className, probability]) => (
          <tr key={className}>
            <Cell>{className}</Cell>
            <Cell mono>{`${(probability * 100).toFixed(1)}%`}</Cell>
          </tr>
        ))}
      </DataTable>
    </section>
  );
}

/** IDs COCO enteros ≥ 0 de los dos campos, o null si falta o sobra algo. */
function parseCropIds(
  imageId: string,
  annotationId: string
): { image_id: number; annotation_id: number } | null {
  const parse = (value: string) => {
    const number = Number(value);
    return value.trim() !== "" && Number.isInteger(number) && number >= 0 ? number : null;
  };
  const image = parse(imageId);
  const annotation = parse(annotationId);
  return image === null || annotation === null
    ? null
    : { image_id: image, annotation_id: annotation };
}

const modelKey = (model: RegisteredModelVersion) => `${model.model_name}:${model.model_version}`;
const inputClass =
  "mt-1 block w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm font-mono";

function UploadPreview({ file }: Readonly<{ file: File }>) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);
  if (url === null) return null;
  return (
    <img
      src={url}
      alt="Vista previa de la imagen"
      className="h-40 w-40 rounded-lg bg-sidebar object-contain"
    />
  );
}

function InferenceForm({ models }: Readonly<{ models: RegisteredModelVersion[] }>) {
  const [params] = useSearchParams();
  const fromUrl = params.has("image_id") || params.has("annotation_id");
  const requested = models.find((entry) => entry.model_version === params.get("model_version"));
  const initial = requested ?? models[0];
  const [selected, setSelected] = useState(initial ? modelKey(initial) : "");
  const model = models.find((entry) => modelKey(entry) === selected) ?? models[0];
  const [mode, setMode] = useState<"upload" | "crop">(fromUrl ? "crop" : "upload");
  const [file, setFile] = useState<File | null>(null);
  const [fileProblem, setFileProblem] = useState<string | null>(null);
  const [release, setRelease] = useState(params.get("dataset_version") ?? "");
  const [imageId, setImageId] = useState(params.get("image_id") ?? "");
  const [annotationId, setAnnotationId] = useState(params.get("annotation_id") ?? "");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [error, setError] = useState<ContractError | null>(null);
  const [queueBusy, setQueueBusy] = useState(false);
  const [queueMessage, setQueueMessage] = useState<string | null>(null);
  const [queueError, setQueueError] = useState<ContractError | null>(null);

  if (model === undefined) return null;
  const cropRelease = release.trim() || model.dataset_version;
  const cropIds = parseCropIds(imageId, annotationId);
  const ready =
    !busy && (mode === "upload" ? file !== null && fileProblem === null : cropIds !== null);

  const chooseFile = (next: File | null) => {
    setFile(next);
    setFileProblem(next === null ? null : uploadProblem(next));
    setResult(null);
    setError(null);
  };

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!ready) return;
    let input: InferenceInput;
    if (mode === "upload" && file !== null) {
      input = {
        kind: "upload",
        modelName: model.model_name,
        modelVersion: model.model_version,
        file,
      };
    } else if (cropIds !== null) {
      input = {
        kind: "crop",
        request: {
          schema_version: "1.0",
          model_name: model.model_name,
          model_version: model.model_version,
          crop: { dataset_version: cropRelease, ...cropIds },
        },
      };
    } else {
      return;
    }
    setBusy(true);
    setResult(null);
    setError(null);
    const outcome = await classifyImage(input);
    setBusy(false);
    if (outcome.ok) setResult(outcome.response);
    else setError(outcome.error);
  };

  const sendToAnnotationQueue = async () => {
    if (result === null || queueBusy) return;

    setQueueBusy(true);
    setQueueMessage(null);
    setQueueError(null);

    const outcome = await enqueueInferenceResult(result, result.source === "upload" ? file : null);

    setQueueBusy(false);

    if (outcome.ok) {
      setQueueMessage(
        outcome.created
          ? `Enviado a la cola de anotación como imagen ${outcome.imageId}.`
          : `La inferencia ya estaba en la cola de anotación como imagen ${outcome.imageId}.`
      );
    } else {
      setQueueError(outcome.error);
    }
  };

  const tab = (active: boolean) =>
    `rounded-lg px-3 py-1.5 text-sm ${active ? "bg-accent-lilac-soft text-ink" : "text-ink-muted"}`;

  return (
    <div className="flex flex-col gap-6">
      <form
        onSubmit={submit}
        className="flex flex-col gap-5 rounded-2xl border border-border bg-surface p-6 shadow-card"
      >
        <label className="text-sm font-medium text-ink">
          Model version
          <select
            className={inputClass}
            value={selected}
            onChange={(event) => {
              setSelected(event.target.value);
              setResult(null);
              setError(null);
            }}
          >
            {models.map((entry) => (
              <option key={modelKey(entry)} value={modelKey(entry)}>
                {`${entry.model_name} v${entry.model_version}`}
              </option>
            ))}
          </select>
          <span className="mt-1 block text-xs font-normal text-ink-muted">
            {`Run ${model.run_id} · release ${model.dataset_version}`}
          </span>
        </label>

        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            aria-pressed={mode === "upload"}
            className={tab(mode === "upload")}
            onClick={() => setMode("upload")}
          >
            Subir imagen
          </button>
          <button
            type="button"
            aria-pressed={mode === "crop"}
            className={tab(mode === "crop")}
            onClick={() => setMode("crop")}
          >
            Recorte del portal
          </button>
        </div>

        {mode === "upload" ? (
          <div className="flex flex-col gap-3">
            <label className="text-sm font-medium text-ink">
              Imagen (PNG o JPEG)
              <input
                type="file"
                accept={ACCEPTED_TYPES.join(",")}
                className="mt-1 block text-sm"
                aria-invalid={fileProblem !== null}
                onChange={(event) => chooseFile(event.target.files?.[0] ?? null)}
              />
            </label>
            <p className="text-xs text-ink-muted">
              Hasta 10 MB. Sube el recorte de un solo animal: el modelo clasifica la imagen completa
              como dog o cat.
            </p>
            {fileProblem !== null && (
              <p className="text-xs font-medium text-status-pending">{fileProblem}</p>
            )}
            {file !== null && fileProblem === null && <UploadPreview file={file} />}
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            <div className="grid gap-3 sm:grid-cols-3">
              <label className="text-sm font-medium text-ink">
                Release del recorte
                <input
                  className={inputClass}
                  value={release}
                  placeholder={model.dataset_version}
                  onChange={(event) => setRelease(event.target.value)}
                />
              </label>
              <label className="text-sm font-medium text-ink">
                image_id
                <input
                  type="number"
                  min={0}
                  className={inputClass}
                  value={imageId}
                  onChange={(event) => setImageId(event.target.value)}
                />
              </label>
              <label className="text-sm font-medium text-ink">
                annotation_id
                <input
                  type="number"
                  min={0}
                  className={inputClass}
                  value={annotationId}
                  onChange={(event) => setAnnotationId(event.target.value)}
                />
              </label>
            </div>
            {cropIds !== null && (
              <img
                src={cropUrl(cropIds.image_id, cropIds.annotation_id)}
                alt={`Recorte img${cropIds.image_id}-ann${cropIds.annotation_id}`}
                className="h-40 w-40 rounded-lg bg-sidebar object-contain"
              />
            )}
          </div>
        )}

        <div>
          <button
            type="submit"
            disabled={!ready}
            className="rounded-lg bg-ink px-4 py-2 text-sm font-medium text-surface disabled:opacity-40"
          >
            {busy ? "Clasificando…" : "Clasificar"}
          </button>
        </div>
      </form>

      {error !== null && (
        <div role="alert" className="rounded-xl bg-status-pending-soft px-4 py-3 text-sm text-ink">
          <p>{error.message}</p>
          <p className="mt-1 font-mono text-xs text-ink-muted">{error.code}</p>
        </div>
      )}
      {result !== null && (
        <>
          <InferenceResult response={result} />

          <section className="rounded-2xl border border-border bg-surface p-4 shadow-card">
            <button
              type="button"
              disabled={queueBusy}
              onClick={() => void sendToAnnotationQueue()}
              className="rounded-lg bg-ink px-4 py-2 text-sm font-medium text-surface disabled:opacity-40"
            >
              {queueBusy ? "Enviando…" : "Enviar a cola de anotación"}
            </button>

            {queueMessage !== null && (
              <p className="mt-3 text-sm font-medium text-ink">{queueMessage}</p>
            )}

            {queueError !== null && (
              <div
                role="alert"
                className="mt-3 rounded-xl bg-status-pending-soft px-4 py-3 text-sm text-ink"
              >
                <p>{queueError.message}</p>
                <p className="mt-1 font-mono text-xs text-ink-muted">{queueError.code}</p>
              </div>
            )}
          </section>
        </>
      )}
    </div>
  );
}

export function InferencePage() {
  const models = useReadyModels();

  return (
    <MlPage
      title="Inference"
      subtitle="Clasifica una imagen o un recorte del portal con una versión del modelo registrada."
    >
      <MlResourceBoundary
        state={models}
        emptyMessage="No hay versiones del modelo con su paquete disponible para inferencia."
      >
        {(response) => <InferenceForm models={response.models.filter((model) => model.servable)} />}
      </MlResourceBoundary>
    </MlPage>
  );
}
