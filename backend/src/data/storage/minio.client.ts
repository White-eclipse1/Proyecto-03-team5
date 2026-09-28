import * as Minio from 'minio';

import { env } from '../../config/env.js';

/**
 * Cliente de MinIO configurado desde variables de entorno.
 */
export const minioClient = new Minio.Client({
  endPoint: env.MINIO_ENDPOINT,
  port: env.MINIO_PORT,
  useSSL: env.MINIO_USE_SSL,
  accessKey: env.MINIO_ACCESS_KEY,
  secretKey: env.MINIO_SECRET_KEY,
});

// Bucket donde se almacenan las imágenes.
export const minioBucket = env.MINIO_BUCKET;
