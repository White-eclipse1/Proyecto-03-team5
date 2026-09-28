import type { Readable } from 'node:stream';

import { findImageById, getImageObjectStream } from '../data/index.js';

export interface ImageFile {
  stream: Readable;
  mimeType: string;
}

/**
 * Obtiene el binario de una imagen desde MinIO junto con su mime type.
 * Devuelve null si la imagen no existe en MariaDB.
 */
export async function getImageFile(imageId: number): Promise<ImageFile | null> {
  const image = await findImageById(imageId);
  if (!image) {
    return null;
  }

  const stream = await getImageObjectStream(image.storageKey);
  return { stream, mimeType: image.mimeType };
}
