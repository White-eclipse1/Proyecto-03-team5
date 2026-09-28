import { describe, expect, it } from 'vitest';

import { idParamSchema } from '../src/logic/annotation.validation.js';

/**
 * Pruebas asociadas a SPEC-VALID-001.
 * Trazabilidad: features/http-validation.feature → src/logic/annotation.validation.ts
 *
 * Estos esquemas cierran la frontera HTTP: validan el dato externo antes de
 * que llegue a la capa de datos.
 */
describe('SPEC-VALID-001 - validación de la frontera HTTP', () => {
  // -------------------------------------------------------------------------
  // Route param :id
  // -------------------------------------------------------------------------

  describe('idParamSchema', () => {
    it('acepta y convierte un id numérico en string', () => {
      const result = idParamSchema.safeParse('7');

      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data).toBe(7);
      }
    });

    it('acepta un id numérico', () => {
      const result = idParamSchema.safeParse(3);

      expect(result.success).toBe(true);
    });

    it('rechaza un id no numérico', () => {
      const result = idParamSchema.safeParse('abc');

      expect(result.success).toBe(false);
    });

    it('rechaza el id cero', () => {
      const result = idParamSchema.safeParse('0');

      expect(result.success).toBe(false);
    });

    it('rechaza un id negativo', () => {
      const result = idParamSchema.safeParse('-5');

      expect(result.success).toBe(false);
    });

    it('rechaza un id decimal', () => {
      const result = idParamSchema.safeParse('2.7');

      expect(result.success).toBe(false);
    });

    it('rechaza una cadena vacía', () => {
      const result = idParamSchema.safeParse('');

      expect(result.success).toBe(false);
    });
  });
});
