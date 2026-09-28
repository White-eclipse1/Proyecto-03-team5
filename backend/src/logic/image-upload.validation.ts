import { z } from 'zod';

/**
 * Tipos de imagen permitidos para el portal.
 */
const imageUploadSchema = z.object({
  mimeType: z.enum(['image/jpeg', 'image/png', 'image/webp']),
  sizeBytes: z.number().int().positive(),
});

export type ImageUploadInput = z.infer<typeof imageUploadSchema>;

export interface ImageUploadValidationResult {
  success: boolean;
}

/**
 * Agrega al esquema base la validación del tamaño máximo.
 */
function createImageUploadSchema(maxSizeBytes: number) {
  return imageUploadSchema.refine((image) => image.sizeBytes <= maxSizeBytes, {
    message: 'La imagen excede el tamaño máximo permitido.',
    path: ['sizeBytes'],
  });
}

/**
 * Valida el tipo y tamaño de una imagen antes de subirla.
 */
export function validateImageUpload(
  input: unknown,
  maxSizeBytes: number,
): ImageUploadValidationResult {
  const schema = createImageUploadSchema(maxSizeBytes);

  return {
    success: schema.safeParse(input).success,
  };
}
