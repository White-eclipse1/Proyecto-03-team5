import { findAllAnnotationRows, findAllImages, listCategories } from '../data/index.js';
import { buildCocoDataset, type CocoDataset } from './coco-export.builder.js';

/**
 * SPEC-COCO-001 — Servicio de exportación COCO.
 *
 * Trae todo el dataset desde la capa Data y delega la transformación al
 * builder puro. Los IDs se conservan tal cual, consistentes entre secciones.
 */
export async function exportCocoDataset(): Promise<CocoDataset> {
  const [images, annotations, categories] = await Promise.all([
    findAllImages(),
    findAllAnnotationRows(),
    listCategories(),
  ]);

  return buildCocoDataset({ images, annotations, categories });
}
