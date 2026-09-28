import { expect, it } from "vitest";
import { qualityReportSchema, splitsReportSchema, versionsReportSchema } from "../src/pipeline/schemas";
import catalog from "./fixtures/catalog.json";
import quality from "./fixtures/quality.json";
import splits from "./fixtures/splits.json";

it("accepts the three contractual test fixtures", () => {
  expect(qualityReportSchema.safeParse(quality).success).toBe(true);
  expect(splitsReportSchema.safeParse(splits).success).toBe(true);
  expect(versionsReportSchema.safeParse(catalog).success).toBe(true);
});
it("rejects counts that do not sum to the total", () => {
  const report = structuredClone(splits);
  report.splits.train.image_count = 999;
  expect(splitsReportSchema.safeParse(report).success).toBe(false);
});
it.each(["train", "validation", "test"] as const)("rejects inconsistent %s ratio", (name) => {
  const report = structuredClone(splits);
  report.splits[name].ratio += 0.01;
  expect(splitsReportSchema.safeParse(report).success).toBe(false);
});
it.each([0.5e-6, 2e-6])("uses Python absolute tolerance (%s)", (difference) => {
  const report = structuredClone(splits);
  report.splits.train.ratio += difference;
  expect(splitsReportSchema.safeParse(report).success).toBe(difference < 1e-6);
});
it("rejects an empty dataset even when counts sum to zero", () => {
  const report = structuredClone(splits);
  report.total_images = 0;
  for (const split of Object.values(report.splits)) { split.image_count = 0; split.ratio = 0; }
  expect(splitsReportSchema.safeParse(report).success).toBe(false);
});
it("accepts distinct versions without imposing order", () => {
  const first = catalog.releases[0]!;
  expect(versionsReportSchema.safeParse({ ...catalog, releases: [{ ...first, dataset_version: "v9.0.0" }, first] }).success).toBe(true);
});
it("rejects duplicate versions even with different references", () => {
  const first = catalog.releases[0]!;
  expect(versionsReportSchema.safeParse({ ...catalog, releases: [first, { ...first, quality_file: "other/quality.json", splits_file: "other/splits.json" }] }).success).toBe(false);
});
