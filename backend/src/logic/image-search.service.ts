import {
  countAnnotationsForImages,
  findCategoriesForImages,
  findImages,
  type Image,
} from '../data/index.js';
import { buildThumbnailUrl } from './dashboard.builder.js';
import { ValidationError } from './errors.js';
import { parseSearchQuery } from './search-query.parser.js';

/** Categoría resumida que acompaña a cada resultado. */
export interface SearchResultCategory {
  id: number;
  name: string;
  color: string;
}

/**
 * Elemento del listado de búsqueda, enriquecido con lo que muestra la UI:
 * miniatura, conteo de anotaciones y clases presentes.
 */
export interface ImageSearchItem extends Image {
  thumbnailUrl: string;
  annotationsCount: number;
  categories: SearchResultCategory[];
}

export interface SearchImagesInput {
  /** Expresión de clases con operadores, ej. "car AND person". */
  q?: string;
  status?: Image['status'] | Image['status'][];
  categoryIds?: number[];
  dateFrom?: Date;
  dateTo?: Date;
  page: number;
  pageSize: number;
}

export interface SearchImagesResult {
  data: ImageSearchItem[];
  pagination: {
    page: number;
    pageSize: number;
    total: number;
    totalPages: number;
  };
}

/**
 * SPEC-SEARCH-001 — Busca imágenes con filtros combinables resueltos en SQL,
 * y enriquece cada resultado con su miniatura, conteo de cajas y clases.
 *
 * La expresión `q` se parsea a operadores AND/OR; la sintaxis ambigua
 * (mezclar AND con OR) produce un ValidationError → 400.
 */
export async function searchImages(input: SearchImagesInput): Promise<SearchImagesResult> {
  let classSearch: ReturnType<typeof parseSearchQuery> = null;
  if (input.q) {
    try {
      classSearch = parseSearchQuery(input.q);
    } catch (error) {
      throw new ValidationError(
        error instanceof Error ? error.message : 'Consulta de búsqueda inválida.',
      );
    }
  }

  const { data, total } = await findImages({
    status: input.status,
    classSearch: classSearch ?? undefined,
    categoryIds: input.categoryIds,
    dateFrom: input.dateFrom,
    dateTo: input.dateTo,
    page: input.page,
    pageSize: input.pageSize,
  });

  const imageIds = data.map((image) => image.id);
  const [annotationCounts, categoriesByImage] = await Promise.all([
    countAnnotationsForImages(imageIds),
    findCategoriesForImages(imageIds),
  ]);

  const items: ImageSearchItem[] = data.map((image) => ({
    ...image,
    thumbnailUrl: buildThumbnailUrl(image.id),
    annotationsCount: annotationCounts.get(image.id) ?? 0,
    categories: categoriesByImage.get(image.id) ?? [],
  }));

  return {
    data: items,
    pagination: {
      page: input.page,
      pageSize: input.pageSize,
      total,
      totalPages: Math.max(1, Math.ceil(total / input.pageSize)),
    },
  };
}
