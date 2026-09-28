import { z } from "zod";

/**
 * Todas las respuestas del backend se validan aquí. Los tipos de dominio
 * usados por el resto de la app se derivan con z.infer, nunca se declaran
 * a mano por separado (para evitar que el tipo y la validación diverjan).
 */

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

export const imageStatusSchema = z.enum(["pending", "in_progress", "completed"]);
export type ImageStatus = z.infer<typeof imageStatusSchema>;

export const imageSchema = z.object({
  id: z.number(),
  filename: z.string(),
  storageKey: z.string(),
  mimeType: z.string(),
  width: z.number(),
  height: z.number(),
  sizeBytes: z.number(),
  status: imageStatusSchema,
  createdAt: z.string(),
  updatedAt: z.string(),
});
export type ImageRecord = z.infer<typeof imageSchema>;

// Respuesta real de POST /images (Fase 1-2, campos limitados)
export const imageUploadResponseSchema = z.object({
  id: z.number(),
  filename: z.string(),
  storageKey: z.string(),
  width: z.number(),
  height: z.number(),
});
export type ImageUploadResponse = z.infer<typeof imageUploadResponseSchema>;

export const paginationSchema = z.object({
  page: z.number(),
  pageSize: z.number(),
  total: z.number(),
  totalPages: z.number(),
});
export type Pagination = z.infer<typeof paginationSchema>;

// Respuesta real de GET /images/search
export const imageSearchResponseSchema = z.object({
  data: z.array(imageSchema),
  pagination: paginationSchema,
});
export type ImageSearchResponse = z.infer<typeof imageSearchResponseSchema>;

// ASUNCIÓN OPCIONAL (no está en el prompt original y el resto del código NO
// depende de ella para funcionar): si el backend expone GET /images/:id, se
// usa solo para enriquecer el badge de status/filename cuando no llegan por
// estado de navegación (ver useImageAnnotations). Las dimensiones reales de
// la imagen se obtienen SIEMPRE cargando el binario real de
// GET /images/:id/file (naturalWidth/naturalHeight), que sí es un endpoint
// del prompt. Si el backend real no expone GET /images/:id, no hace falta
// tocar nada: la llamada falla en silencio y se usan valores por defecto.
export const imageDetailResponseSchema = imageSchema;

// ---------------------------------------------------------------------------
// Categories
// ---------------------------------------------------------------------------

export const categorySchema = z.object({
  id: z.number(),
  name: z.string(),
  color: z.string().regex(/^#[0-9A-Fa-f]{6}$/, "El color debe ser un hex #RRGGBB"),
  createdAt: z.string(),
});
export type Category = z.infer<typeof categorySchema>;

export const categoriesResponseSchema = z.array(categorySchema);

// ---------------------------------------------------------------------------
// Annotations
// ---------------------------------------------------------------------------

const categorySummarySchema = categorySchema.pick({
  id: true,
  name: true,
  color: true,
});

export const annotationSchema = z.object({
  id: z.number(),
  imageId: z.number(),
  categoryId: z.number(),
  bboxX: z.number(),
  bboxY: z.number(),
  bboxWidth: z.number(),
  bboxHeight: z.number(),
  area: z.number(),
  iscrowd: z.boolean(),
  category: categorySummarySchema,
});
export type Annotation = z.infer<typeof annotationSchema>;

export const annotationsResponseSchema = z.array(annotationSchema);

export const createAnnotationBodySchema = z.object({
  categoryId: z.number(),
  bboxX: z.number().nonnegative(),
  bboxY: z.number().nonnegative(),
  bboxWidth: z.number().positive(),
  bboxHeight: z.number().positive(),
  iscrowd: z.boolean().optional(),
});
export type CreateAnnotationBody = z.infer<typeof createAnnotationBodySchema>;

export const patchAnnotationBodySchema = createAnnotationBodySchema.partial();
export type PatchAnnotationBody = z.infer<typeof patchAnnotationBodySchema>;

// ---------------------------------------------------------------------------
// Image status
// ---------------------------------------------------------------------------

export const patchImageStatusBodySchema = z.object({
  status: z.enum(["in_progress", "completed"]),
});
export type PatchImageStatusBody = z.infer<typeof patchImageStatusBodySchema>;

// ---------------------------------------------------------------------------
// Bounding box (forma común usada en UI, subconjunto de Annotation)
// ---------------------------------------------------------------------------

export interface BBox {
  bboxX: number;
  bboxY: number;
  bboxWidth: number;
  bboxHeight: number;
}
