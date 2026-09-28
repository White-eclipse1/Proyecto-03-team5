import {
  type AnnotationWithCategory,
  countAnnotationsByImage,
  createAnnotationRow,
  deleteAnnotationRow,
  findAnnotationById,
  findImageById,
  listAnnotationsByImage,
  updateAnnotationRow,
  updateImageStatus,
} from '../data/index.js';
import {
  createAnnotationForImageSchema,
  mergeAnnotationPatch,
  patchAnnotationSchema,
  validateBboxWithBounds,
} from './annotation.validation.js';
import { NotFoundError, ValidationError } from './errors.js';

export interface AnnotationBoxInput {
  categoryId: number;
  bboxX: number;
  bboxY: number;
  bboxWidth: number;
  bboxHeight: number;
  iscrowd?: boolean;
}

export async function getAnnotationsForImage(imageId: number): Promise<AnnotationWithCategory[]> {
  const image = await findImageById(imageId);
  if (!image) {
    throw new NotFoundError('La imagen no existe.');
  }

  return listAnnotationsByImage(imageId);
}

/**
 * SPEC-ANNOT-001 / SPEC-VALID-001 — Crea una bounding box.
 *
 * Reglas aplicadas:
 *  - El body se valida con Zod antes de tocar la base de datos.
 *  - La imagen debe existir (aporta las dimensiones para validar límites).
 *  - bboxX/bboxY >= 0 y bboxWidth/bboxHeight > 0.
 *  - La caja no puede salirse del canvas de la imagen.
 *  - El área se calcula en el backend: bboxWidth × bboxHeight.
 *  - Al crear la primera caja, la imagen pasa de `pending` a `in_progress`.
 */
export async function createAnnotationForImage(
  imageId: number,
  input: unknown,
): Promise<AnnotationWithCategory> {
  const parsed = createAnnotationForImageSchema.safeParse(input);
  if (!parsed.success) {
    throw new ValidationError(
      parsed.error.issues[0]?.message ?? 'Los datos de la anotación son inválidos.',
    );
  }
  const box = parsed.data;

  const image = await findImageById(imageId);
  if (!image) {
    throw new NotFoundError('La imagen no existe.');
  }

  // La caja debe caber dentro de las dimensiones reales de la imagen.
  const bounds = validateBboxWithBounds(
    {
      categoryId: box.categoryId,
      bboxX: box.bboxX,
      bboxY: box.bboxY,
      bboxWidth: box.bboxWidth,
      bboxHeight: box.bboxHeight,
    },
    image.width,
    image.height,
  );
  if (!bounds.success) {
    throw new ValidationError(bounds.error ?? 'La caja no cabe dentro de la imagen.');
  }

  const id = await createAnnotationRow({
    imageId,
    categoryId: box.categoryId,
    bboxX: box.bboxX,
    bboxY: box.bboxY,
    bboxWidth: box.bboxWidth,
    bboxHeight: box.bboxHeight,
    area: box.bboxWidth * box.bboxHeight,
    isCrowd: box.iscrowd ?? false,
  });

  // Primera caja de la imagen: entra al flujo de anotación.
  if (image.status === 'pending') {
    await updateImageStatus(imageId, 'in_progress');
  }

  const created = await findAnnotationById(id);
  if (!created) {
    throw new Error('No se pudo leer la anotación recién creada.');
  }

  return created;
}

/**
 * Mueve, redimensiona o reclasifica una anotación. Acepta cambios parciales
 * y revalida la caja resultante contra los límites de la imagen.
 */
export async function updateAnnotation(
  id: number,
  changes: unknown,
): Promise<AnnotationWithCategory> {
  const parsed = patchAnnotationSchema.safeParse(changes);
  if (!parsed.success) {
    throw new ValidationError(
      parsed.error.issues[0]?.message ?? 'Los datos de la anotación son inválidos.',
    );
  }
  const patch = parsed.data;

  const existing = await findAnnotationById(id);
  if (!existing) {
    throw new NotFoundError('La anotación no existe.');
  }

  const image = await findImageById(existing.imageId);
  if (!image) {
    throw new NotFoundError('La imagen asociada a la anotación no existe.');
  }

  // Se combina lo recibido con lo ya guardado para validar la caja final.
  const merged = mergeAnnotationPatch(existing, patch);

  const bounds = validateBboxWithBounds(merged, image.width, image.height);
  if (!bounds.success) {
    throw new ValidationError(bounds.error ?? 'La caja no cabe dentro de la imagen.');
  }

  // El área siempre la recalcula el backend a partir de la caja resultante.
  const row: Record<string, number | boolean> = {
    ...merged,
    area: merged.bboxWidth * merged.bboxHeight,
  };
  if (patch.iscrowd !== undefined) row.isCrowd = patch.iscrowd;

  await updateAnnotationRow(id, row);

  const updated = await findAnnotationById(id);
  if (!updated) {
    throw new NotFoundError('La anotación no existe.');
  }

  return updated;
}

/**
 * Elimina una anotación. Si la imagen queda sin cajas, vuelve a `pending`
 * para regresar a la cola de trabajo.
 */
export async function deleteAnnotation(id: number): Promise<void> {
  const existing = await findAnnotationById(id);
  if (!existing) {
    throw new NotFoundError('La anotación no existe.');
  }

  await deleteAnnotationRow(id);

  const remaining = await countAnnotationsByImage(existing.imageId);
  if (remaining === 0) {
    const image = await findImageById(existing.imageId);
    if (image && image.status !== 'pending') {
      await updateImageStatus(existing.imageId, 'pending');
    }
  }
}
