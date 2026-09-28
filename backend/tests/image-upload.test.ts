import { describe, expect, it } from 'vitest';

import { validateImageUpload } from '../src/logic/image-upload.validation.js';

/**
 * Pruebas asociadas a SPEC-UPLOAD-001.
 */
describe('SPEC-UPLOAD-001 - validación de imágenes', () => {
  const maxSizeBytes = 5 * 1024 * 1024;

  it('acepta una imagen JPEG válida', () => {
    const result = validateImageUpload(
      {
        mimeType: 'image/jpeg',
        sizeBytes: 1024,
      },
      maxSizeBytes,
    );

    expect(result.success).toBe(true);
  });

  it('rechaza un archivo que no sea imagen', () => {
    const result = validateImageUpload(
      {
        mimeType: 'application/pdf',
        sizeBytes: 1024,
      },
      maxSizeBytes,
    );

    expect(result.success).toBe(false);
  });

  it('rechaza una imagen que exceda el tamaño máximo', () => {
    const result = validateImageUpload(
      {
        mimeType: 'image/png',
        sizeBytes: maxSizeBytes + 1,
      },
      maxSizeBytes,
    );

    expect(result.success).toBe(false);
  });
});
