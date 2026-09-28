import { minioBucket, minioClient } from './minio.client.js';

/**
 * Verifica que el bucket exista.
 * Si no existe, lo crea automáticamente.
 */
export async function ensureMinioBucket(): Promise<void> {
  const exists = await minioClient.bucketExists(minioBucket);

  if (!exists) {
    await minioClient.makeBucket(minioBucket);
  }
}

/**
 * Guarda un archivo en MinIO.
 */
export async function uploadImageObject(
  storageKey: string,
  buffer: Buffer,
  mimeType: string,
): Promise<void> {
  await minioClient.putObject(minioBucket, storageKey, buffer, buffer.length, {
    'Content-Type': mimeType,
  });
}

/**
 * Elimina un archivo almacenado en MinIO.
 */
export async function deleteImageObject(storageKey: string): Promise<void> {
  await minioClient.removeObject(minioBucket, storageKey);
}

/**
 * Obtiene un stream de lectura del archivo guardado en MinIO.
 */
export async function getImageObjectStream(storageKey: string) {
  return minioClient.getObject(minioBucket, storageKey);
}
