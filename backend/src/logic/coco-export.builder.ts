import type { Annotation, Category, Image } from '../data/db/schema.js';

/**
 * SPEC-COCO-001 ÔÇö Estructuras y construcci├│n del formato COCO.
 *
 * Este m├│dulo es una funci├│n pura: no importa la capa de datos ni la
 * configuraci├│n, por lo que puede probarse en aislamiento total.
 */

export interface CocoImage {
  id: number;
  file_name: string;
  width: number;
  height: number;
}

export interface CocoAnnotation {
  id: number;
  image_id: number;
  category_id: number;
  /** [x, y, width, height] en p├¡xeles absolutos */
  bbox: [number, number, number, number];
  area: number;
  iscrowd: 0 | 1;
  /** Pol├¡gonos no soportados: siempre vac├¡o */
  segmentation: number[];
}

export interface CocoCategory {
  id: number;
  name: string;
}

export interface CocoDataset {
  images: CocoImage[];
  annotations: CocoAnnotation[];
  categories: CocoCategory[];
}

export interface CocoSourceData {
  images: Image[];
  annotations: Annotation[];
  categories: Category[];
}

/**
 * Funci├│n pura: transforma filas de la base de datos en un documento COCO.
 *
 * Los IDs de BD (enteros autoincrementales) se reutilizan tal cual para
 * mantener consistencia entre las tres secciones.
 */
export function buildCocoDataset(source: CocoSourceData): CocoDataset {
  const images: CocoImage[] = source.images.map((image) => ({
    id: image.id,
    file_name: image.filename,
    width: image.width,
    height: image.height,
  }));

  const categories: CocoCategory[] = source.categories.map((category) => ({
    id: category.id,
    name: category.name,
  }));

  const annotations: CocoAnnotation[] = source.annotations.map(toCocoAnnotation);

  return { images, annotations, categories };
}

/**
 * Convierte una anotaci├│n de BD a su representaci├│n COCO.
 * El ├írea se recalcula desde el bbox para garantizar coherencia
 * (area === width ├ù height), sin confiar en el valor almacenado.
 */
function toCocoAnnotation(annotation: Annotation): CocoAnnotation {
  const width = annotation.bboxWidth;
  const height = annotation.bboxHeight;

  return {
    id: annotation.id,
    image_id: annotation.imageId,
    category_id: annotation.categoryId,
    bbox: [annotation.bboxX, annotation.bboxY, width, height],
    area: width * height,
    iscrowd: annotation.isCrowd ? 1 : 0,
    segmentation: [],
  };
}
