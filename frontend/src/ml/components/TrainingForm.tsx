import { type FormEvent, useCallback, useState } from "react";
import { useVersionsReport } from "@/pipeline/dataSource";
import { createTrainingJob } from "../dataSource";
import { TRAINING_PARAM_NAMES, type TrainingJob, type TrainingParamName } from "../schemas";
import {
  DEFAULT_TRAINING_PARAMS,
  type RawTrainingParams,
  TRAINING_PARAM_HINTS,
  type TrainingFormErrors,
  validateTrainingForm,
} from "../trainingForm";
import { type GateState, ReleaseDetails } from "./ReleaseDetails";

type SubmitState =
  | { status: "idle" }
  | { status: "invalid" }
  | { status: "submitting" }
  | { status: "failed"; message: string }
  | { status: "created"; job: TrainingJob };

const OPTIMIZERS = ["adam", "adamw", "sgd"] as const;
const inputClass =
  "mt-1 block w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm font-mono aria-[invalid=true]:border-status-pending";

function ParamInput({
  name,
  value,
  error,
  onChange,
}: Readonly<{
  name: TrainingParamName;
  value: string;
  error: string | undefined;
  onChange: (name: TrainingParamName, value: string) => void;
}>) {
  const id = `param-${name}`;
  const describedBy = error ? `${id}-hint ${id}-error` : `${id}-hint`;
  return (
    <div>
      <label htmlFor={id} className="font-mono text-sm font-medium text-ink">
        {name}
      </label>
      {name === "optimizer" ? (
        <select
          id={id}
          value={value}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy}
          onChange={(event) => onChange(name, event.target.value)}
          className={inputClass}
        >
          {OPTIMIZERS.map((optimizer) => (
            <option key={optimizer} value={optimizer}>
              {optimizer}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={id}
          type="text"
          inputMode={name === "hidden_layers" ? "text" : "decimal"}
          value={value}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy}
          onChange={(event) => onChange(name, event.target.value)}
          className={inputClass}
        />
      )}
      <p id={`${id}-hint`} className="mt-1 text-xs text-ink-muted">
        {TRAINING_PARAM_HINTS[name]}
      </p>
      {error && (
        <p id={`${id}-error`} className="mt-1 text-xs font-medium text-status-pending">
          {error}
        </p>
      )}
    </div>
  );
}

/**
 * APP-02: configura un training job sobre un release publicado de P2. Nada se
 * envía si el Quality Gate del release falló o si algún parámetro no cumple
 * `trainingJobRequestSchema` (el mismo contrato que valida el backend).
 */
export function TrainingForm({ onCreated }: Readonly<{ onCreated: () => void }>) {
  const versions = useVersionsReport();
  const [datasetVersion, setDatasetVersion] = useState("");
  const [gate, setGate] = useState<GateState>({ kind: "checking" });
  const [params, setParams] = useState<RawTrainingParams>(DEFAULT_TRAINING_PARAMS);
  const [errors, setErrors] = useState<TrainingFormErrors>({});
  const [submit, setSubmit] = useState<SubmitState>({ status: "idle" });

  const releases = versions.status === "success" ? versions.data.releases : [];
  const release = releases.find((entry) => entry.dataset_version === datasetVersion);
  const onGateChange = useCallback((next: GateState) => setGate(next), []);

  function updateParam(name: TrainingParamName, value: string) {
    setParams((current) => ({ ...current, [name]: value }));
    setErrors(({ [name]: _cleared, ...rest }) => rest);
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (release && gate.kind !== "ok") return;
    const validation = validateTrainingForm(datasetVersion, params);
    if (!validation.ok) {
      setErrors(validation.errors);
      setSubmit({ status: "invalid" });
      return;
    }
    setErrors({});
    setSubmit({ status: "submitting" });
    const result = await createTrainingJob(validation.request);
    if (result.ok) {
      setSubmit({ status: "created", job: result.job });
      onCreated();
    } else {
      setSubmit({ status: "failed", message: result.error.message });
    }
  }

  const blocked = release !== undefined && gate.kind !== "ok";

  return (
    <section aria-labelledby="new-training" className="flex flex-col gap-4">
      <h2 id="new-training" className="text-lg font-semibold">
        Nuevo entrenamiento
      </h2>
      <form noValidate onSubmit={handleSubmit} className="flex flex-col gap-5">
        <div>
          <label htmlFor="training-release" className="text-sm font-medium text-ink">
            Release del dataset
          </label>
          <select
            id="training-release"
            value={datasetVersion}
            aria-invalid={errors.release ? true : undefined}
            aria-describedby={errors.release ? "training-release-error" : undefined}
            onChange={(event) => {
              setDatasetVersion(event.target.value);
              setGate({ kind: "checking" });
              setErrors(({ release: _cleared, ...rest }) => rest);
              setSubmit({ status: "idle" });
            }}
            className={inputClass}
          >
            <option value="">Selecciona un release</option>
            {releases.map((entry) => (
              <option key={entry.dataset_version} value={entry.dataset_version}>
                {entry.dataset_version}
              </option>
            ))}
          </select>
          {versions.status === "loading" && (
            <p className="mt-1 text-xs text-ink-muted">Cargando releases publicados…</p>
          )}
          {versions.status === "error" && (
            <p className="mt-1 text-xs text-status-pending">
              No se pudieron leer los releases publicados del Proyecto 2: {versions.message}
            </p>
          )}
          {errors.release && (
            <p id="training-release-error" className="mt-1 text-xs font-medium text-status-pending">
              {errors.release}
            </p>
          )}
        </div>

        {release && (
          <ReleaseDetails
            key={release.dataset_version}
            release={release}
            onGateChange={onGateChange}
          />
        )}
        {release && gate.kind === "blocked" && (
          <p role="alert" className="text-sm font-medium text-status-pending">
            {gate.reason}
          </p>
        )}
        {release && gate.kind === "unknown" && (
          <p role="alert" className="text-sm font-medium text-status-pending">
            No se pudo verificar el Quality Gate de este release; no se puede entrenar.
          </p>
        )}

        <fieldset className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <legend className="mb-2 text-sm font-medium text-ink">Hiperparámetros</legend>
          {TRAINING_PARAM_NAMES.map((name) => (
            <ParamInput
              key={name}
              name={name}
              value={params[name]}
              error={errors[name]}
              onChange={updateParam}
            />
          ))}
        </fieldset>

        {submit.status === "invalid" && (
          <p role="alert" className="text-sm font-medium text-status-pending">
            Revisa los parámetros marcados; no se creó ningún job.
          </p>
        )}
        {submit.status === "failed" && (
          <p role="alert" className="text-sm font-medium text-status-pending">
            {submit.message}
          </p>
        )}
        {submit.status === "created" && (
          <p className="text-sm font-medium text-ink">
            {`Job ${submit.job.job_id} creado (${submit.job.status}).`}
          </p>
        )}

        <div>
          <button
            type="submit"
            disabled={blocked || submit.status === "submitting"}
            className="rounded-lg bg-ink px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-ink/90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Crear training job
          </button>
        </div>
      </form>
    </section>
  );
}
