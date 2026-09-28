import { type FormEvent, useEffect, useState } from "react";
import {
  getSettings,
  type QualitySettings,
  qualitySettingsSchema,
  type SplitSettings,
  saveQualitySettings,
  saveSplitSettings,
  splitSettingsSchema,
} from "../../lib/api/settings";
import { PageHeader } from "../components/PageHeader";

const ruleLabels: Record<keyof QualitySettings, string> = {
  min_images_per_class: "Mínimo de imágenes distintas por clase",
  max_imbalance_ratio: "Ratio máximo de desbalance",
  max_small_object_ratio: "Proporción máxima de objetos pequeños (0–1)",
  degenerate_boxes: "Máximo de anotaciones inválidas",
  duplicate_similarity_threshold: "Similitud mínima pHash para detectar pares (0–1)",
  min_spatial_dispersion: "Dispersión espacial mínima (0–0.5)",
};
const saved = "Guardado. Se aplicará en la próxima ejecución del pipeline/release.";
const inputClass = "rounded border border-border bg-surface p-2";
function message(error: unknown) {
  return error instanceof Error ? error.message : "No se pudo guardar la configuración.";
}

function NumberField({
  id,
  label,
  value,
  onChange,
}: {
  id: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label htmlFor={id} className="flex flex-col gap-1">
      {label}
      <input
        id={id}
        className={inputClass}
        type="number"
        step="any"
        value={Number.isFinite(value) ? value : ""}
        onChange={(event) =>
          onChange(event.target.value === "" ? Number.NaN : Number(event.target.value))
        }
      />
    </label>
  );
}

function QualityForm({ initial }: { initial: QualitySettings }) {
  const [value, setValue] = useState(initial);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setSuccess("");
    const parsed = qualitySettingsSchema.safeParse(value);
    if (!parsed.success) {
      setError(
        "Revisa los umbrales, tamaños y acciones: " +
          parsed.error.issues.map((i) => i.path.join(".")).join(", ")
      );
      return;
    }
    setSaving(true);
    try {
      setValue(await saveQualitySettings(parsed.data));
      setSuccess(saved);
    } catch (err) {
      setError(message(err));
    } finally {
      setSaving(false);
    }
  }
  return (
    <form
      aria-label="Política de calidad"
      onSubmit={submit}
      onChange={() => setSuccess("")}
      noValidate
      className="space-y-4"
    >
      <h2 className="text-xl font-semibold">Política de calidad</h2>
      <fieldset disabled={saving} className="space-y-4">
        {(Object.keys(ruleLabels) as (keyof QualitySettings)[]).map((name) => (
          <div key={name} className="grid gap-3 sm:grid-cols-2">
            <NumberField
              id={name}
              label={ruleLabels[name]}
              value={value[name].threshold}
              onChange={(threshold) =>
                setValue((current) => ({ ...current, [name]: { ...current[name], threshold } }))
              }
            />
            <label htmlFor={`${name}-action`} className="flex flex-col gap-1">
              Acción: {ruleLabels[name]}
              <select
                id={`${name}-action`}
                className={inputClass}
                value={value[name].action}
                onChange={(event) => {
                  const action = event.target.value as "warn" | "fail";
                  setValue((current) => ({ ...current, [name]: { ...current[name], action } }));
                }}
              >
                <option value="warn">warn — advertencia</option>
                <option value="fail">fail — fallo de calidad</option>
              </select>
            </label>
          </div>
        ))}
        <NumberField
          id="width_px"
          label="Ancho límite de objeto pequeño (px)"
          value={value.max_small_object_ratio.width_px}
          onChange={(width_px) =>
            setValue((current) => ({
              ...current,
              max_small_object_ratio: { ...current.max_small_object_ratio, width_px },
            }))
          }
        />
        <NumberField
          id="height_px"
          label="Alto límite de objeto pequeño (px)"
          value={value.max_small_object_ratio.height_px}
          onChange={(height_px) =>
            setValue((current) => ({
              ...current,
              max_small_object_ratio: { ...current.max_small_object_ratio, height_px },
            }))
          }
        />
        <p>
          Una caja es pequeña cuando ambos lados están por debajo de sus límites. pHash usa la
          similitud para detectar pares; el cumplimiento exige cero pares.
        </p>
        <button type="submit" disabled={saving} className={inputClass}>
          {saving ? "Guardando calidad…" : "Guardar calidad"}
        </button>
      </fieldset>
      {error && <p role="alert">{error}</p>}
      {success && <p role="status">{success}</p>}
    </form>
  );
}

function SplitsForm({ initial }: { initial: SplitSettings }) {
  const [value, setValue] = useState(initial);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setSuccess("");
    const parsed = splitSettingsSchema.safeParse(value);
    if (!parsed.success) {
      setError(
        "Los ratios deben estar entre 0 y 1 y sumar 1 (100%, tolerancia 0.000001). La seed debe ser un entero seguro."
      );
      return;
    }
    setSaving(true);
    try {
      setValue(await saveSplitSettings(parsed.data));
      setSuccess(saved);
    } catch (err) {
      setError(message(err));
    } finally {
      setSaving(false);
    }
  }
  return (
    <form
      aria-label="Configuración de splits"
      onSubmit={submit}
      onChange={() => setSuccess("")}
      noValidate
      className="space-y-4"
    >
      <h2 className="text-xl font-semibold">Configuración de splits</h2>
      <p>Proporciones de 0 a 1: train + validation + test deben sumar 1 (100%).</p>
      <fieldset disabled={saving} className="space-y-3">
        {(["train", "val", "test", "seed"] as const).map((name) => (
          <NumberField
            key={name}
            id={`split-${name}`}
            label={
              name === "val"
                ? "Validation"
                : name === "seed"
                  ? "Seed reproducible"
                  : name === "train"
                    ? "Train"
                    : "Test"
            }
            value={value[name]}
            onChange={(number) => setValue((current) => ({ ...current, [name]: number }))}
          />
        ))}
        <button type="submit" disabled={saving} className={inputClass}>
          {saving ? "Guardando splits…" : "Guardar splits"}
        </button>
      </fieldset>
      {error && <p role="alert">{error}</p>}
      {success && <p role="status">{success}</p>}
    </form>
  );
}

function SettingsContent({ onRetry }: { onRetry: () => void }) {
  const [config, setConfig] = useState<Awaited<ReturnType<typeof getSettings>>>();
  const [error, setError] = useState("");
  useEffect(() => {
    const request = new AbortController();
    let cancelled = false;
    setError("");
    getSettings(request.signal)
      .then((value) => {
        if (!cancelled) setConfig(value);
      })
      .catch((err) => {
        if (!cancelled) setError(message(err));
      });
    return () => {
      cancelled = true;
      request.abort();
    };
  }, []);
  return (
    <main className="flex-1 px-6 py-6 lg:px-10">
      <PageHeader title="Settings" subtitle="Configuración persistente del pipeline" />
      <p className="my-4">
        Guardar no ejecuta el pipeline ni cambia reportes o releases existentes.
      </p>
      {error ? (
        <div role="alert">
          {error}{" "}
          <button type="button" onClick={onRetry}>
            Reintentar
          </button>
        </div>
      ) : config ? (
        <div className="space-y-10">
          <QualityForm initial={config.quality} />
          <SplitsForm initial={config.splits} />
        </div>
      ) : (
        <p role="status">Cargando configuración…</p>
      )}
    </main>
  );
}

export function SettingsPage() {
  const [attempt, setAttempt] = useState(0);
  return <SettingsContent key={attempt} onRetry={() => setAttempt((value) => value + 1)} />;
}
