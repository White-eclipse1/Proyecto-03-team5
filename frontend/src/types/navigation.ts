import type { ImageStatus } from "./schemas";

/**
 * Estado de navegación OPCIONAL que Upload (o Search) puede pasar al navegar
 * a /annotate/:imageId, para que el Canvas conozca filename/status reales
 * sin depender de ningún endpoint adicional. Si no se pasa nada, el Canvas
 * usa valores por defecto seguros y, si el backend expone GET /images/:id,
 * los corrige después en segundo plano (ver useImageAnnotations).
 */
export interface AnnotateNavigationState {
  filenames?: Record<number, string>;
  statuses?: Record<number, ImageStatus>;
  /** Pantalla desde la que se navegó, para que "Volver" regrese al mismo lugar. */
  from?: "upload" | "search";
}
