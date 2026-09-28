import { z } from "zod";
import type { QualityCheck } from "../schemas";

const criterionSchema = z.object({ threshold: z.number(), operator: z.string() });
const classesSchema = z.array(
  z.object({
    category_name: z.string(),
    image_count: z.number(),
    image_ids: z.array(z.number()).optional(),
  })
);
const samplesSchema = z.array(
  z.object({ image_id: z.number(), annotation_id: z.number().optional() })
);
const smallBoxSchema = z.object({
  width_px: z.number(),
  height_px: z.number(),
  operator: z.string(),
  combination: z.literal("and"),
});
const pairsSchema = z.array(
  z.object({
    image_id_a: z.number(),
    image_id_b: z.number(),
    similarity: z.number(),
    split_a: z.string().optional(),
    split_b: z.string().optional(),
    hamming_distance: z.number().optional(),
  })
);

export function CheckCriterion({ check }: Readonly<{ check: QualityCheck }>) {
  const criterion = criterionSchema.safeParse(check.details.criterion);
  return (
    <>
      {criterion.success
        ? `${criterion.data.operator} ${criterion.data.threshold}`
        : "Criterio no disponible"}
    </>
  );
}

export function CheckDetails({ check }: Readonly<{ check: QualityCheck }>) {
  const details = check.details;
  const criterion = criterionSchema.safeParse(details.criterion);
  const classes = classesSchema.safeParse(
    details.images_per_category ?? details.classes_below_minimum
  );
  const samples = samplesSchema.safeParse(details.offending_samples);
  const pairs = pairsSchema.safeParse(details.image_pairs);
  const smallBox = smallBoxSchema.safeParse(details.small_box_detection);
  const maxClassCount = classes.success
    ? Math.max(...classes.data.map((entry) => entry.image_count), 1)
    : 1;
  const metricRatio =
    criterion.success && criterion.data.threshold !== 0
      ? Math.min(1, Math.abs(check.metric_value / criterion.data.threshold))
      : check.passed
        ? 1
        : 0;
  return (
    <div className="space-y-2 text-sm">
      <div
        role="img"
        aria-label={`Métrica ${check.metric_value} respecto al criterio`}
        className="h-2 rounded-full bg-sidebar"
      >
        <div
          className={`h-2 rounded-full ${check.passed ? "bg-accent-mint" : "bg-status-error"}`}
          style={{ width: `${metricRatio * 100}%` }}
        />
      </div>
      {classes.success && (
        <ul className="space-y-2">
          {classes.data.map((entry) => (
            <li key={entry.category_name} className="space-y-1">
              <div className="h-2 rounded-full bg-sidebar" aria-hidden="true">
                <div
                  className="h-2 rounded-full bg-accent-lilac"
                  style={{ width: `${(entry.image_count / maxClassCount) * 100}%` }}
                />
              </div>
              {entry.category_name}: {entry.image_count} imágenes
            </li>
          ))}
        </ul>
      )}
      {smallBox.success && (
        <p>
          Caja pequeña: ancho {smallBox.data.operator} {smallBox.data.width_px} px y alto{" "}
          {smallBox.data.operator} {smallBox.data.height_px} px.
        </p>
      )}
      {typeof details.similarity_threshold === "number" && (
        <p>
          Detección pHash: similitud ≥ {details.similarity_threshold}. El criterio de aprobación
          cuenta pares, no similitud.
        </p>
      )}
      {pairs.success && pairs.data.length > 0 && (
        <details>
          <summary>Pares detectados ({pairs.data.length})</summary>
          <ul>
            {pairs.data.map((pair) => (
              <li key={`${pair.image_id_a}-${pair.image_id_b}`}>
                <a href={`/annotate/${pair.image_id_a}`} target="_blank" rel="noreferrer">
                  Imagen {pair.image_id_a}
                </a>{" "}
                /{" "}
                <a href={`/annotate/${pair.image_id_b}`} target="_blank" rel="noreferrer">
                  {pair.image_id_b}
                </a>
                : similitud {pair.similarity}
                {pair.split_a && pair.split_b && (
                  <span>
                    {" "}
                    · {pair.split_a} → {pair.split_b}
                  </span>
                )}
                {pair.hamming_distance !== undefined && (
                  <span> · Hamming: {pair.hamming_distance}</span>
                )}
              </li>
            ))}
          </ul>
        </details>
      )}
      {samples.success && samples.data.length > 0 && (
        <details>
          <summary>Muestras afectadas ({samples.data.length})</summary>
          <ul>
            {samples.data.map((sample) => (
              <li key={`${sample.image_id}-${sample.annotation_id}`}>
                <a href={`/annotate/${sample.image_id}`} target="_blank" rel="noreferrer">
                  Imagen {sample.image_id}
                </a>
                , anotación {sample.annotation_id ?? "no indicada"}
              </li>
            ))}
          </ul>
        </details>
      )}
      {typeof details.std_center_x === "number" && (
        <p>Dispersión horizontal: {details.std_center_x}</p>
      )}
      {typeof details.std_center_y === "number" && (
        <p>Dispersión vertical: {details.std_center_y}</p>
      )}
    </div>
  );
}
