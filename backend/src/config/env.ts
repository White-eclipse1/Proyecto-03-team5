import 'dotenv/config';
import { fileURLToPath } from 'node:url';
import { z } from 'zod';

/**
 * Valida las variables de entorno usadas por la aplicación.
 */
const envSchema = z.object({
  PIPELINE_CONFIG_ROOT: z
    .string()
    .min(1)
    .default(fileURLToPath(new URL('../../../app/', import.meta.url))),

  NODE_ENV: z.enum(['development', 'test', 'production']).default('development'),

  PORT: z.coerce.number().int().positive().default(3000),

  DATABASE_URL: z.string().min(1),

  MINIO_ENDPOINT: z.string().min(1),

  MINIO_PORT: z.coerce.number().int().positive().default(9000),

  MINIO_USE_SSL: z
    .enum(['true', 'false'])
    .default('false')
    .transform((value) => value === 'true'),

  MINIO_ACCESS_KEY: z.string().min(1),

  MINIO_SECRET_KEY: z.string().min(1),

  MINIO_BUCKET: z.string().min(3),

  MAX_UPLOAD_SIZE_BYTES: z.coerce
    .number()
    .int()
    .positive()
    .default(5 * 1024 * 1024),
});

/**
 * Variables ya validadas y tipadas.
 * Si alguna configuración requerida falta, la aplicación falla al iniciar.
 */
export const env = envSchema.parse(process.env);
