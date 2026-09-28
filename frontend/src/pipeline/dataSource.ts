import { useMemo } from "react";
import type { ZodType } from "zod";
import { projectionsReportSchema } from "./projectionSchemas";
import type { QualityReport, SplitsReport } from "./schemas";
import {
  type DatasetRelease,
  qualityReportSchema,
  reportReferenceSchema,
  splitsReportSchema,
  versionsReportSchema,
} from "./schemas";
import { useReportFetch } from "./useReportFetch";

/** Real reports published from the repository reports directory. */
export function useQualityReport() {
  return useReportFetch("/reports/quality.json", qualityReportSchema);
}

export function useSplitsReport() {
  return useReportFetch("/reports/splits.json", splitsReportSchema);
}

export function useVersionsReport() {
  return useReportFetch("/reports/versions.json", versionsReportSchema);
}

/** References are validated again at the request boundary, before any fetch. */
export function useReleaseReport(
  kind: "quality",
  release: DatasetRelease
): ReturnType<typeof useQualityReport>;
export function useReleaseReport(
  kind: "splits",
  release: DatasetRelease
): ReturnType<typeof useSplitsReport>;
export function useReleaseReport(kind: "quality" | "splits", release: DatasetRelease) {
  const reference = release[`${kind}_file`];
  const parsed = reportReferenceSchema(`${kind}.json`).safeParse(reference);
  const schema = useMemo(
    () =>
      (
        (kind === "quality" ? qualityReportSchema : splitsReportSchema) as ZodType<
          QualityReport | SplitsReport
        >
      ).refine(
        (report) => report.dataset_version === release.dataset_version,
        "La versión del reporte no coincide con el catálogo"
      ),
    [kind, release.dataset_version]
  );
  return useReportFetch(parsed.success ? `/reports/${parsed.data}` : null, schema);
}

export function useProjectionsReport() {
  return useReportFetch("/reports/projections.json", projectionsReportSchema);
}
