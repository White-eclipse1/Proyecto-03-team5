import { useEffect, useMemo } from "react";
import { useReleaseReport } from "@/pipeline/dataSource";
import { type DatasetRelease, type QualityReport, reportReferenceSchema } from "@/pipeline/schemas";
import { releaseProvenanceSchema, trainingBlockedReason, trainingManifestSchema } from "../schemas";
import { useOptionalReport } from "../useOptionalReport";

export type GateState =
  | { kind: "checking" }
  | { kind: "unknown"; message: string }
  | { kind: "blocked"; reason: string }
  | { kind: "ok" };

/** Clases con imágenes en el release (las categorías vacías no son entrenables). */
function trainableClasses(report: QualityReport): { name: string; images: number }[] {
  const check = report.checks.find((entry) => entry.check_name === "max_imbalance_ratio");
  const categories = check?.details.images_per_category;
  if (!Array.isArray(categories)) return [];
  return categories.flatMap((entry) => {
    if (!entry || typeof entry !== "object") return [];
    const { category_name: name, image_count: images } = entry as Record<string, unknown>;
    return typeof name === "string" && typeof images === "number" && images > 0
      ? [{ name, images }]
      : [];
  });
}

/** `releases/<v>/quality.json` → `releases/<v>/<file>`, validando la referencia. */
function siblingReport(release: DatasetRelease, file: string): string | null {
  const reference = reportReferenceSchema("quality.json").safeParse(release.quality_file);
  return reference.success
    ? `/reports/${reference.data.slice(0, -"quality.json".length)}${file}`
    : null;
}

function Field({ label, children }: Readonly<{ label: string; children: React.ReactNode }>) {
  return (
    <div>
      <dt className="text-xs uppercase text-ink-muted">{label}</dt>
      <dd className="mt-1 text-sm text-ink">{children}</dd>
    </div>
  );
}

export function ReleaseDetails({
  release,
  onGateChange,
}: Readonly<{ release: DatasetRelease; onGateChange: (gate: GateState) => void }>) {
  const quality = useReleaseReport("quality", release);
  const version = release.dataset_version;
  const provenanceSchema = useMemo(
    () =>
      releaseProvenanceSchema.refine(
        (provenance) => provenance.dataset_version === version,
        "La procedencia no corresponde al release"
      ),
    [version]
  );
  const manifestSchema = useMemo(
    () =>
      trainingManifestSchema.refine(
        (manifest) => manifest.dataset_version === version,
        "El manifiesto no corresponde al release"
      ),
    [version]
  );
  const provenance = useOptionalReport(siblingReport(release, "provenance.json"), provenanceSchema);
  const manifest = useOptionalReport(siblingReport(release, "manifest.json"), manifestSchema);

  const qualityStatus = quality.status === "success" ? quality.data.status : null;
  const qualityError = quality.status === "error" ? quality.message : null;
  useEffect(() => {
    if (qualityError !== null) {
      onGateChange({ kind: "unknown", message: qualityError });
    } else if (qualityStatus === null) {
      onGateChange({ kind: "checking" });
    } else {
      const reason = trainingBlockedReason(qualityStatus);
      onGateChange(reason === null ? { kind: "ok" } : { kind: "blocked", reason });
    }
  }, [qualityStatus, qualityError, onGateChange]);

  return (
    <section
      aria-label="Release seleccionado"
      className="rounded-2xl border border-border bg-surface p-5 shadow-card"
    >
      <dl className="grid gap-4 sm:grid-cols-2">
        <Field label="Dataset version">
          <span className="font-mono">{version}</span>
        </Field>
        <Field label="Quality Gate">
          {quality.status === "success" && <span className="font-mono">{quality.data.status}</span>}
          {quality.status === "loading" && <span className="text-ink-muted">Verificando…</span>}
          {quality.status === "error" && (
            <span className="text-status-pending">No se pudo leer: {quality.message}</span>
          )}
        </Field>
        <Field label="Procedencia DVC">
          {provenance.status === "success" && (
            <ul className="space-y-1 font-mono text-xs">
              {provenance.data.dvc_outputs.map((output) => (
                <li key={output.path}>{`${output.path}: ${output.md5}`}</li>
              ))}
            </ul>
          )}
          {provenance.status === "missing" && (
            <span className="text-ink-muted">Hash DVC no registrado para este release.</span>
          )}
          {provenance.status === "loading" && <span className="text-ink-muted">Cargando…</span>}
          {provenance.status === "error" && (
            <span className="text-status-pending">
              No se pudo leer la procedencia: {provenance.message}
            </span>
          )}
        </Field>
        <Field label="Clases">
          {quality.status === "success" ? (
            <span className="flex flex-wrap gap-1">
              {trainableClasses(quality.data).map((entry) => (
                <span
                  key={entry.name}
                  className="rounded-full bg-accent-lilac-soft px-2 py-0.5 text-xs text-accent-lilac"
                >
                  {`${entry.name} (${entry.images})`}
                </span>
              ))}
            </span>
          ) : (
            <span className="text-ink-muted">—</span>
          )}
        </Field>
        <div className="sm:col-span-2">
          <Field label="Manifiesto">
            {manifest.status === "success" && (
              <>
                <ul aria-label="Manifiesto de splits" className="space-y-1">
                  {Object.entries(manifest.data.splits).map(([name, split]) => (
                    <li key={name}>
                      {`${name}: ${split.image_count} imágenes (${Math.round(split.ratio * 100)}%)`}
                    </li>
                  ))}
                </ul>
                <p className="mt-1 font-mono text-xs text-ink-muted">
                  {manifest.data.manifest_hash}
                </p>
              </>
            )}
            {manifest.status === "missing" && (
              <span className="text-ink-muted">
                El manifiesto 70/20/10 de este release todavía no está disponible.
              </span>
            )}
            {manifest.status === "loading" && <span className="text-ink-muted">Cargando…</span>}
            {manifest.status === "error" && (
              <span className="text-status-pending">
                No se pudo leer el manifiesto: {manifest.message}
              </span>
            )}
          </Field>
        </div>
      </dl>
    </section>
  );
}
