import { z } from "zod";

export const objectPerClassSchema = z.object({
  categoryId: z.number(),
  name: z.string(),
  color: z.string(),
  count: z.number(),
});

export const recentUploadSchema = z.object({
  id: z.number(),
  thumbnailUrl: z.string(),
  status: z.enum(["pending", "in_progress", "completed"]),
});

export const annotationProgressSchema = z.object({
  annotated: z.number(),
  pending: z.number(),
});

export const dashboardSummarySchema = z.object({
  imagesUploaded: z.number(),
  imagesAnnotated: z.number(),
  boundingBoxes: z.number(),
  categoriesCount: z.number(),
  objectsPerClass: z.array(objectPerClassSchema),
  annotationProgress: annotationProgressSchema,
  recentUploads: z.array(recentUploadSchema),
});

// Types are always derived from the schemas (z.infer), never declared by hand,
// so a change in the contract can only ever be made in one place.
export type ObjectPerClass = z.infer<typeof objectPerClassSchema>;
export type RecentUpload = z.infer<typeof recentUploadSchema>;
export type AnnotationProgress = z.infer<typeof annotationProgressSchema>;
export type DashboardSummary = z.infer<typeof dashboardSummarySchema>;
