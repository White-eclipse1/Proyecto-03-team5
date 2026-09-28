/**
 * SPEC-DASH-001 ÔÇö Construcci├│n del resumen del dashboard.
 *
 * M├│dulo puro: recibe los agregados ya calculados en SQL y los transforma en
 * el contrato que consume el frontend. No importa la capa de datos ni la
 * configuraci├│n, por lo que puede probarse en aislamiento total.
 */

export type ImageStatus = 'pending' | 'in_progress' | 'completed';

export interface ObjectPerClass {
  categoryId: number;
  name: string;
  color: string;
  count: number;
}

export interface RecentUpload {
  id: number;
  thumbnailUrl: string;
  status: ImageStatus;
}

export interface AnnotationProgress {
  annotated: number;
  pending: number;
}

export interface DashboardSummary {
  imagesUploaded: number;
  imagesAnnotated: number;
  boundingBoxes: number;
  categoriesCount: number;
  objectsPerClass: ObjectPerClass[];
  annotationProgress: AnnotationProgress;
  recentUploads: RecentUpload[];
}

/** Agregados que entrega la capa Data. */
export interface DashboardSourceData {
  statusCounts: { status: ImageStatus; count: number }[];
  boundingBoxes: number;
  categoriesCount: number;
  objectsPerClass: ObjectPerClass[];
  recentImages: { id: number; status: ImageStatus }[];
}

/**
 * URL del binario servida por el backend.
 * El bucket de MinIO es privado, as├¡ que su URL nunca se expone al navegador.
 */
export function buildThumbnailUrl(imageId: number): string {
  return `/images/${imageId}/file`;
}

/**
 * Transforma los agregados de SQL en el resumen del dashboard.
 *
 * `imagesAnnotated` cuenta solo las im├ígenes terminadas (`completed`).
 * `annotationProgress.pending` agrupa `pending` + `in_progress`, es decir
 * todo lo que a├║n falta por terminar, de modo que
 * `annotated + pending === imagesUploaded`.
 */
export function buildDashboardSummary(source: DashboardSourceData): DashboardSummary {
  const byStatus = new Map(source.statusCounts.map((row) => [row.status, row.count]));

  const pending = byStatus.get('pending') ?? 0;
  const inProgress = byStatus.get('in_progress') ?? 0;
  const completed = byStatus.get('completed') ?? 0;

  return {
    imagesUploaded: pending + inProgress + completed,
    imagesAnnotated: completed,
    boundingBoxes: source.boundingBoxes,
    categoriesCount: source.categoriesCount,
    objectsPerClass: source.objectsPerClass,
    annotationProgress: {
      annotated: completed,
      pending: pending + inProgress,
    },
    recentUploads: source.recentImages.map((image) => ({
      id: image.id,
      thumbnailUrl: buildThumbnailUrl(image.id),
      status: image.status,
    })),
  };
}
