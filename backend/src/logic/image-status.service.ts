import { countAnnotationsByImage, findImageById, updateImageStatus } from '../data/index.js';
import { imageStatusTransitionSchema } from './annotation.validation.js';
import { NotFoundError, ValidationError } from './errors.js';

/**
 * Cambia el status de una imagen (pending -> in_progress -> completed).
 *
 * Solo se permiten las transiciones que dispara la UI de anotación. Pasar a
 * `completed` exige al menos una caja: un dataset no debe contener imágenes
 * marcadas como terminadas sin ninguna anotación.
 */
export async function setImageStatus(imageId: number, status: unknown): Promise<void> {
  const parsed = imageStatusTransitionSchema.safeParse({ status });
  if (!parsed.success) {
    throw new ValidationError(parsed.error.issues[0]?.message ?? 'Status inválido.');
  }
  const nextStatus = parsed.data.status;

  const image = await findImageById(imageId);
  if (!image) {
    throw new NotFoundError('La imagen no existe.');
  }

  if (nextStatus === 'completed') {
    const total = await countAnnotationsByImage(imageId);
    if (total === 0) {
      throw new ValidationError('No se puede completar una imagen sin anotaciones.');
    }
  }

  if (image.status === nextStatus) {
    return; // Idempotente.
  }

  await updateImageStatus(imageId, nextStatus);
}
