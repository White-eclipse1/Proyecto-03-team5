import {
  countAllAnnotations,
  countAllCategories,
  countAnnotationsByCategory,
  countImagesByStatus,
  findRecentImages,
} from '../data/index.js';
import { buildDashboardSummary, type DashboardSummary } from './dashboard.builder.js';

/** Cuántas imágenes recientes se muestran en el panel de actividad. */
const RECENT_UPLOADS_LIMIT = 8;

/**
 * SPEC-DASH-001 — Calcula el resumen del dashboard.
 *
 * Todas las cifras salen de consultas agregadas en SQL (`COUNT` / `GROUP BY`);
 * ninguna está fija en el código. La transformación al contrato del frontend
 * la hace el builder puro.
 */
export async function getDashboardSummary(): Promise<DashboardSummary> {
  const [statusCounts, boundingBoxes, categoriesCount, objectsPerClass, recentImages] =
    await Promise.all([
      countImagesByStatus(),
      countAllAnnotations(),
      countAllCategories(),
      countAnnotationsByCategory(),
      findRecentImages(RECENT_UPLOADS_LIMIT),
    ]);

  return buildDashboardSummary({
    statusCounts,
    boundingBoxes,
    categoriesCount,
    objectsPerClass,
    recentImages: recentImages.map((image) => ({ id: image.id, status: image.status })),
  });
}
