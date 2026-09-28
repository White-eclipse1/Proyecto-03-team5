import { z } from "zod";

/**
 * Espeja `app/presentation/contracts.py` (P2-12, forma v1.0 congelada).
 * Igual que con el backend real: nunca se confía en el tipo que TypeScript
 * infiere del JSON importado, se valida con estos schemas y el tipo de cada
 * componente sale de z.infer<typeof schema>.
 */

const identifierSchema = z
  .string()
  .regex(/^[A-Za-z0-9][A-Za-z0-9._-]*$/, "dataset_version/check_name tiene un formato inválido");

export const qualityCheckSchema = z.object({
  check_name: identifierSchema,
  passed: z.boolean(),
  metric_value: z.number(),
  details: z.record(z.string(), z.unknown()).default({}),
  action: z.enum(["warn", "fail"]),
});
export type QualityCheck = z.infer<typeof qualityCheckSchema>;

export const qualityReportSchema = z.object({
  schema_version: z.literal("1.0"),
  dataset_version: identifierSchema,
  status: z.enum(["passed", "warning", "failed"]),
  checks: z.array(qualityCheckSchema).min(1),
});
export type QualityReport = z.infer<typeof qualityReportSchema>;

const splitSummarySchema = z.object({
  image_count: z.number().int().min(0),
  ratio: z.number().min(0).max(1),
});

const classDistributionSchema = z.record(z.string(), z.record(z.string(), z.number().int().min(0)));

const leakageSchema = z.object({
  status: z.enum(["passed", "warning", "failed"]).optional(),
  checked_groups: z.number().int().min(0).optional(),
  duplicate_groups: z.number().int().min(0).optional(),
  cross_split_groups: z.number().int().min(0).optional(),
  coverage: z.number().int().min(0).optional(),
});

export const splitsReportSchema = z
  .object({
    schema_version: z.literal("1.0"),
    dataset_version: identifierSchema,
    total_images: z.number().int().positive(),
    splits: z.object({
      train: splitSummarySchema,
      validation: splitSummarySchema,
      test: splitSummarySchema,
    }),
    class_distribution: classDistributionSchema.default({}),
    leakage: leakageSchema.default({}),
  })
  .superRefine((report, context) => {
    const entries = Object.entries(report.splits);
    if (
      entries.reduce((total, [, split]) => total + split.image_count, 0) !== report.total_images
    ) {
      context.addIssue({
        code: "custom",
        message: "Los conteos deben sumar total_images",
        path: ["splits"],
      });
    }
    if (report.total_images <= 0) return;
    for (const [name, split] of entries) {
      // Python: math.isclose(rel_tol=0, abs_tol=1e-6).
      if (Math.abs(split.ratio - split.image_count / report.total_images) > 1e-6) {
        context.addIssue({
          code: "custom",
          message: "El ratio debe coincidir con image_count / total_images",
          path: ["splits", name, "ratio"],
        });
      }
    }
  });
export type SplitsReport = z.infer<typeof splitsReportSchema>;

export function reportReferenceSchema(filename: string) {
  return z.string().refine((value) => {
    const parts = value.split("/");
    return (
      parts.at(-1) === filename &&
      parts.every((part) => part !== "." && part !== ".." && /^[A-Za-z0-9._-]+$/.test(part))
    );
  }, "Referencia de reporte inválida");
}

export const datasetReleaseSchema = z.object({
  dataset_version: identifierSchema,
  quality_file: reportReferenceSchema("quality.json"),
  splits_file: reportReferenceSchema("splits.json"),
});

export const versionsReportSchema = z
  .object({
    schema_version: z.literal("1.0"),
    releases: z.array(datasetReleaseSchema),
  })
  .refine(
    (report) =>
      new Set(report.releases.map((release) => release.dataset_version)).size ===
      report.releases.length,
    "dataset_version debe ser único en el catálogo"
  );
export type VersionsReport = z.infer<typeof versionsReportSchema>;

export type DatasetRelease = z.infer<typeof datasetReleaseSchema>;
