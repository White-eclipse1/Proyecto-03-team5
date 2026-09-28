import { z } from "zod";

const id = z.number().int().min(0).max(Number.MAX_SAFE_INTEGER);
const point = z.strictObject({
  image_id: id,
  file_name: z
    .string()
    .min(1)
    .refine(
      (value) =>
        !/[\\:]/.test(value) &&
        !value.includes(String.fromCharCode(0)) &&
        value.split("/").every((part) => !["", ".", ".."].includes(part))
    ),
  category_ids: z.array(id).refine((ids) => new Set(ids).size === ids.length),
  x: z.number().finite(),
  y: z.number().finite(),
});

/** Mirrors presentation/projections_contracts.py; unrelated to quality/splits contracts. */
export const projectionsReportSchema = z
  .strictObject({
    schema_version: z.literal("1.0"),
    dataset_version: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]*$/),
    dataset_fingerprint: z.string().regex(/^[0-9a-f]{64}$/),
    total_images: z.number().int().positive(),
    categories: z.array(z.strictObject({ id, name: z.string().min(1) })).min(1),
    features: z.strictObject({
      representation: z.literal("rgb_pixels"),
      width: z.literal(16),
      height: z.literal(16),
      channels: z.literal(3),
      resize_filter: z.literal("LANCZOS"),
      exif_orientation: z.literal("transpose"),
      normalization: z.literal("divide_by_255"),
    }),
    pca: z.strictObject({
      parameters: z.strictObject({
        n_components: z.literal(2),
        svd_solver: z.literal("full"),
        whiten: z.boolean(),
      }),
      explained_variance_ratio: z
        .array(z.number().min(0).max(1))
        .length(2)
        .refine((values) => values.reduce((sum, value) => sum + value, 0) <= 1 + 1e-6),
      points: z.array(point),
    }),
    tsne: z.strictObject({
      parameters: z.strictObject({
        n_components: z.literal(2),
        random_state: z
          .number()
          .int()
          .min(0)
          .max(2 ** 32 - 1),
        init: z.literal("pca"),
        learning_rate: z.literal("auto"),
        max_iter: z.number().int().min(250),
        perplexity: z.number().positive(),
        method: z.literal("barnes_hut"),
      }),
      points: z.array(point),
    }),
  })
  .superRefine((report, context) => {
    const fail = (message: string) => context.addIssue({ code: "custom", message });
    const categories = new Set(report.categories.map((category) => category.id));
    if (categories.size !== report.categories.length) fail("Categorías repetidas");
    for (const method of [report.pca, report.tsne]) {
      if (method.points.length !== report.total_images) fail("Cobertura incompleta");
      if (new Set(method.points.map((item) => item.image_id)).size !== method.points.length)
        fail("IDs repetidos");
      if (method.points.some((item) => item.category_ids.some((value) => !categories.has(value))))
        fail("Categoría desconocida");
    }
    const pca = new Map(report.pca.points.map((item) => [item.image_id, item]));
    for (const item of report.tsne.points) {
      const other = pca.get(item.image_id);
      if (
        !other ||
        other.file_name !== item.file_name ||
        other.category_ids.length !== item.category_ids.length ||
        item.category_ids.some((value) => !other.category_ids.includes(value))
      )
        fail("PCA/t-SNE no coinciden en imágenes y etiquetas");
    }
    if (report.tsne.parameters.perplexity >= report.total_images)
      fail("Perplexity debe ser menor que total_images");
  });
export type ProjectionsReport = z.infer<typeof projectionsReportSchema>;
export type ProjectionPoint = z.infer<typeof point>;
