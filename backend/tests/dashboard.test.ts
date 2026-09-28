import { describe, expect, it } from 'vitest';
import type { DashboardSourceData } from '../src/logic/dashboard.builder.js';
import { buildDashboardSummary } from '../src/logic/dashboard.builder.js';

/**
 * Pruebas asociadas a SPEC-DASH-001.
 * Trazabilidad: features/dashboard.feature ÔåÆ src/logic/dashboard.builder.ts
 *
 * Se prueba la funci├│n pura, que transforma los agregados de SQL en el
 * resumen que consume el frontend, sin depender de la base de datos.
 */

function makeSource(overrides: Partial<DashboardSourceData> = {}): DashboardSourceData {
  return {
    statusCounts: [],
    boundingBoxes: 0,
    categoriesCount: 0,
    objectsPerClass: [],
    recentImages: [],
    ...overrides,
  };
}

describe('SPEC-DASH-001 - resumen del dashboard', () => {
  describe('totales de im├ígenes', () => {
    it('imagesUploaded suma todos los estados', () => {
      const summary = buildDashboardSummary(
        makeSource({
          statusCounts: [
            { status: 'pending', count: 2 },
            { status: 'in_progress', count: 1 },
            { status: 'completed', count: 3 },
          ],
        }),
      );

      expect(summary.imagesUploaded).toBe(6);
    });

    it('imagesAnnotated cuenta solo las completadas', () => {
      const summary = buildDashboardSummary(
        makeSource({
          statusCounts: [
            { status: 'pending', count: 2 },
            { status: 'in_progress', count: 1 },
            { status: 'completed', count: 3 },
          ],
        }),
      );

      expect(summary.imagesAnnotated).toBe(3);
    });

    it('un dataset vac├¡o devuelve ceros sin fallar', () => {
      const summary = buildDashboardSummary(makeSource());

      expect(summary.imagesUploaded).toBe(0);
      expect(summary.imagesAnnotated).toBe(0);
      expect(summary.boundingBoxes).toBe(0);
      expect(summary.categoriesCount).toBe(0);
      expect(summary.objectsPerClass).toEqual([]);
      expect(summary.recentUploads).toEqual([]);
    });
  });

  describe('progreso de anotaci├│n', () => {
    it('annotated son las completadas y pending agrupa el resto', () => {
      const summary = buildDashboardSummary(
        makeSource({
          statusCounts: [
            { status: 'pending', count: 2 },
            { status: 'in_progress', count: 1 },
            { status: 'completed', count: 3 },
          ],
        }),
      );

      expect(summary.annotationProgress.annotated).toBe(3);
      expect(summary.annotationProgress.pending).toBe(3);
    });

    it('annotated m├ís pending siempre iguala imagesUploaded', () => {
      const summary = buildDashboardSummary(
        makeSource({
          statusCounts: [
            { status: 'pending', count: 5 },
            { status: 'in_progress', count: 7 },
            { status: 'completed', count: 11 },
          ],
        }),
      );

      const { annotated, pending } = summary.annotationProgress;
      expect(annotated + pending).toBe(summary.imagesUploaded);
    });

    it('sin im├ígenes completadas el progreso anotado es cero', () => {
      const summary = buildDashboardSummary(
        makeSource({ statusCounts: [{ status: 'pending', count: 4 }] }),
      );

      expect(summary.annotationProgress.annotated).toBe(0);
      expect(summary.annotationProgress.pending).toBe(4);
    });
  });

  describe('objetos por clase', () => {
    it('conserva categoryId, nombre, color y conteo', () => {
      const summary = buildDashboardSummary(
        makeSource({
          objectsPerClass: [{ categoryId: 2, name: 'car', color: '#3498DB', count: 4 }],
        }),
      );

      expect(summary.objectsPerClass[0]).toEqual({
        categoryId: 2,
        name: 'car',
        color: '#3498DB',
        count: 4,
      });
    });

    it('refleja el total de bounding boxes', () => {
      const summary = buildDashboardSummary(makeSource({ boundingBoxes: 17 }));

      expect(summary.boundingBoxes).toBe(17);
    });

    it('refleja el total de categor├¡as', () => {
      const summary = buildDashboardSummary(makeSource({ categoriesCount: 3 }));

      expect(summary.categoriesCount).toBe(3);
    });
  });

  describe('actividad reciente', () => {
    it('la miniatura se sirve por el backend, no por MinIO', () => {
      const summary = buildDashboardSummary(
        makeSource({ recentImages: [{ id: 7, status: 'pending' }] }),
      );

      expect(summary.recentUploads[0]).toEqual({
        id: 7,
        thumbnailUrl: '/images/7/file',
        status: 'pending',
      });
    });

    it('conserva el orden recibido de la consulta', () => {
      const summary = buildDashboardSummary(
        makeSource({
          recentImages: [
            { id: 3, status: 'completed' },
            { id: 2, status: 'in_progress' },
            { id: 1, status: 'pending' },
          ],
        }),
      );

      expect(summary.recentUploads.map((upload) => upload.id)).toEqual([3, 2, 1]);
    });
  });
});
