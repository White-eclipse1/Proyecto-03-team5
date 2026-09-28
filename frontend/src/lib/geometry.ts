import type { BBox } from "../types/schemas";

export const MIN_BOX_SIZE_PX = 3;

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Garantiza que una caja quede completamente dentro de [0,0,imgW,imgH],
 * igual que valida el backend (bboxX+bboxWidth<=width, etc.).
 */
export function clampBBox(bbox: BBox, imageWidth: number, imageHeight: number): BBox {
  const width = clamp(bbox.bboxWidth, MIN_BOX_SIZE_PX, imageWidth);
  const height = clamp(bbox.bboxHeight, MIN_BOX_SIZE_PX, imageHeight);
  const x = clamp(bbox.bboxX, 0, imageWidth - width);
  const y = clamp(bbox.bboxY, 0, imageHeight - height);
  return { bboxX: x, bboxY: y, bboxWidth: width, bboxHeight: height };
}

/**
 * Convierte coordenadas de pantalla (clientX/clientY) a coordenadas absolutas
 * de la imagen original, deshaciendo el `displayScale` (zoom) aplicado a la
 * capa que envuelve <img> + overlay de cajas. El resultado NUNCA depende del
 * zoom: siempre son píxeles reales de la imagen.
 */
export function clientToImageCoords(
  clientX: number,
  clientY: number,
  containerRect: DOMRect,
  displayScale: number
): { x: number; y: number } {
  const x = (clientX - containerRect.left) / displayScale;
  const y = (clientY - containerRect.top) / displayScale;
  return { x, y };
}

export function bboxesEqual(a: BBox, b: BBox): boolean {
  return (
    a.bboxX === b.bboxX &&
    a.bboxY === b.bboxY &&
    a.bboxWidth === b.bboxWidth &&
    a.bboxHeight === b.bboxHeight
  );
}
