import { describe, expect, it } from 'vitest';
import type { Annotation, Category, Image } from '../src/data/db/schema.js';
import { buildCocoDataset } from '../src/logic/coco-export.builder.js';

/**
 * Pruebas asociadas a SPEC-COCO-001.
 * Trazabilidad: features/coco-export.feature ÔåÆ src/logic/coco-export.service.ts
 *
 * Se prueba la funci├│n pura buildCocoDataset, que transforma filas de BD
 * en un documento COCO sin depender de la base de datos.
 */

function makeImage(overrides: Partial<Image> = {}): Image {
  return {
    id: 1,
    filename: 'sample.png',
    storageKey: 'images/sample.png',
    mimeType: 'image/png',
    width: 640,
    height: 480,
    sizeBytes: 1000,
    status: 'completed',
    createdAt: new Date(),
    updatedAt: new Date(),
    ...overrides,
  };
}

function makeCategory(overrides: Partial<Category> = {}): Category {
  return {
    id: 1,
    name: 'person',
    color: '#FF5733',
    createdAt: new Date(),
    ...overrides,
  };
}

function makeAnnotation(overrides: Partial<Annotation> = {}): Annotation {
  return {
    id: 1,
    imageId: 1,
    categoryId: 1,
    bboxX: 10,
    bboxY: 20,
    bboxWidth: 100,
    bboxHeight: 50,
    area: 5000,
    isCrowd: false,
    createdAt: new Date(),
    updatedAt: new Date(),
    ...overrides,
  };
}

describe('SPEC-COCO-001 - exportaci├│n en formato COCO', () => {
  it('el documento tiene las tres secciones obligatorias', () => {
    const coco = buildCocoDataset({
      images: [makeImage()],
      categories: [makeCategory()],
      annotations: [makeAnnotation()],
    });

    expect(coco).toHaveProperty('images');
    expect(coco).toHaveProperty('annotations');
    expect(coco).toHaveProperty('categories');
  });

  it('cada imagen expone id, file_name, width y height', () => {
    const coco = buildCocoDataset({
      images: [makeImage({ id: 7, filename: 'foto.jpg', width: 800, height: 600 })],
      categories: [],
      annotations: [],
    });

    expect(coco.images[0]).toEqual({
      id: 7,
      file_name: 'foto.jpg',
      width: 800,
      height: 600,
    });
  });

  it('el bbox est├í en orden [x, y, width, height]', () => {
    const coco = buildCocoDataset({
      images: [makeImage()],
      categories: [makeCategory()],
      annotations: [makeAnnotation({ bboxX: 10, bboxY: 20, bboxWidth: 100, bboxHeight: 50 })],
    });

    expect(coco.annotations[0]?.bbox).toEqual([10, 20, 100, 50]);
  });

  it('el area es coherente con el bbox (width ├ù height)', () => {
    const coco = buildCocoDataset({
      images: [makeImage()],
      categories: [makeCategory()],
      annotations: [makeAnnotation({ bboxWidth: 100, bboxHeight: 50 })],
    });

    const ann = coco.annotations[0];
    expect(ann?.area).toBe(5000);
    expect(ann?.area).toBe((ann?.bbox[2] ?? 0) * (ann?.bbox[3] ?? 0));
  });

  it('iscrowd se convierte de boolean a 0 cuando es false', () => {
    const coco = buildCocoDataset({
      images: [makeImage()],
      categories: [makeCategory()],
      annotations: [makeAnnotation({ isCrowd: false })],
    });

    expect(coco.annotations[0]?.iscrowd).toBe(0);
  });

  it('iscrowd se convierte de boolean a 1 cuando es true', () => {
    const coco = buildCocoDataset({
      images: [makeImage()],
      categories: [makeCategory()],
      annotations: [makeAnnotation({ isCrowd: true })],
    });

    expect(coco.annotations[0]?.iscrowd).toBe(1);
  });

  it('cada annotation.image_id existe en la secci├│n images', () => {
    const coco = buildCocoDataset({
      images: [makeImage({ id: 1 }), makeImage({ id: 2, filename: 'b.png' })],
      categories: [makeCategory()],
      annotations: [makeAnnotation({ id: 1, imageId: 2 })],
    });

    const imageIds = new Set(coco.images.map((i) => i.id));
    for (const ann of coco.annotations) {
      expect(imageIds.has(ann.image_id)).toBe(true);
    }
  });

  it('cada annotation.category_id existe en la secci├│n categories', () => {
    const coco = buildCocoDataset({
      images: [makeImage()],
      categories: [makeCategory({ id: 1 }), makeCategory({ id: 2, name: 'car' })],
      annotations: [makeAnnotation({ categoryId: 2 })],
    });

    const categoryIds = new Set(coco.categories.map((c) => c.id));
    for (const ann of coco.annotations) {
      expect(categoryIds.has(ann.category_id)).toBe(true);
    }
  });

  it('las im├ígenes sin anotaciones tambi├®n se incluyen', () => {
    const coco = buildCocoDataset({
      images: [makeImage({ id: 99, filename: 'sin-cajas.png' })],
      categories: [makeCategory()],
      annotations: [],
    });

    expect(coco.images).toHaveLength(1);
    expect(coco.images[0]?.id).toBe(99);
    expect(coco.annotations).toHaveLength(0);
  });

  it('cada categor├¡a expone id y name', () => {
    const coco = buildCocoDataset({
      images: [],
      categories: [makeCategory({ id: 3, name: 'dog' })],
      annotations: [],
    });

    expect(coco.categories[0]).toEqual({ id: 3, name: 'dog' });
  });

  it('segmentation est├í presente como arreglo vac├¡o', () => {
    const coco = buildCocoDataset({
      images: [makeImage()],
      categories: [makeCategory()],
      annotations: [makeAnnotation()],
    });

    expect(coco.annotations[0]?.segmentation).toEqual([]);
  });
});
