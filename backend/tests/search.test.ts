import { describe, expect, it } from 'vitest';

import { parseSearchQuery } from '../src/logic/search-query.parser.js';

/**
 * Pruebas asociadas a SPEC-SEARCH-001.
 * Trazabilidad: features/search.feature ÔåÆ src/logic/search-query.parser.ts
 *
 * El parser es una funci├│n pura: no toca la base de datos.
 */
describe('SPEC-SEARCH-001 - parseo de consultas con operadores', () => {
  describe('consultas v├ílidas', () => {
    it('parsea un solo t├®rmino y usa AND por defecto', () => {
      const result = parseSearchQuery('car');

      expect(result).toEqual({ terms: ['car'], operator: 'AND' });
    });

    it('parsea dos t├®rminos con AND', () => {
      const result = parseSearchQuery('car AND person');

      expect(result).toEqual({ terms: ['car', 'person'], operator: 'AND' });
    });

    it('parsea dos t├®rminos con OR', () => {
      const result = parseSearchQuery('car OR person');

      expect(result).toEqual({ terms: ['car', 'person'], operator: 'OR' });
    });

    it('parsea tres t├®rminos con AND', () => {
      const result = parseSearchQuery('car AND person AND dog');

      expect(result).toEqual({ terms: ['car', 'person', 'dog'], operator: 'AND' });
    });

    it('descarta espacios sobrantes', () => {
      const result = parseSearchQuery('   car    AND   person  ');

      expect(result).toEqual({ terms: ['car', 'person'], operator: 'AND' });
    });

    it('normaliza los t├®rminos a min├║sculas', () => {
      const result = parseSearchQuery('Car AND PERSON');

      expect(result).toEqual({ terms: ['car', 'person'], operator: 'AND' });
    });

    it('no parte una clase que contiene la palabra AND', () => {
      const result = parseSearchQuery('android');

      expect(result).toEqual({ terms: ['android'], operator: 'AND' });
    });

    it('no parte una clase que contiene la palabra OR', () => {
      const result = parseSearchQuery('tractor');

      expect(result).toEqual({ terms: ['tractor'], operator: 'AND' });
    });

    it('no trata "and" en min├║sculas como operador', () => {
      const result = parseSearchQuery('and');

      expect(result).toEqual({ terms: ['and'], operator: 'AND' });
    });
  });

  describe('consultas sin b├║squeda', () => {
    it('devuelve null para una cadena vac├¡a', () => {
      expect(parseSearchQuery('')).toBeNull();
    });

    it('devuelve null para una cadena de solo espacios', () => {
      expect(parseSearchQuery('   ')).toBeNull();
    });
  });

  describe('consultas inv├ílidas', () => {
    it('rechaza mezclar AND con OR', () => {
      expect(() => parseSearchQuery('car AND person OR dog')).toThrow();
    });

    it('rechaza un operador sin t├®rmino a la derecha', () => {
      expect(() => parseSearchQuery('car AND')).toThrow();
    });

    it('rechaza un operador sin t├®rmino a la izquierda', () => {
      expect(() => parseSearchQuery('AND car')).toThrow();
    });

    it('rechaza dos operadores consecutivos', () => {
      expect(() => parseSearchQuery('car AND AND person')).toThrow();
    });

    it('rechaza un OR sin t├®rmino a la derecha', () => {
      expect(() => parseSearchQuery('person OR')).toThrow();
    });
  });
});
