import { and, count, desc, eq, exists, gte, inArray, like, lte, type SQL, sql } from 'drizzle-orm';
import { db } from '../db/client.js';
import { annotations, categories, type Image, images, type NewImage } from '../db/schema.js';

type ImageStatusValue = Image['status'];

/**
 * Guarda en MariaDB los metadatos de una imagen.
 */
export async function createImageMetadata(image: NewImage): Promise<number> {
  const result = await db.insert(images).values(image).$returningId();

  const createdImage = result[0];

  if (!createdImage) {
    throw new Error('No se pudo guardar la imagen en MariaDB.');
  }

  return createdImage.id;
}

/** SPEC-SEARCH-001: búsqueda por clases ya parseada. */
export interface ClassSearch {
  terms: string[];
  operator: 'AND' | 'OR';
}

export interface FindImagesOptions {
  /** Uno o varios status a incluir. Sin filtro si se omite. */
  status?: ImageStatusValue | ImageStatusValue[];
  /** Búsqueda por clases con operadores, resuelta con EXISTS en SQL. */
  classSearch?: ClassSearch;
  /** Restringe a imágenes con anotaciones de alguna de estas categorías. */
  categoryIds?: number[];
  /** Rango sobre created_at. */
  dateFrom?: Date;
  dateTo?: Date;
  page: number;
  pageSize: number;
}

export interface FindImagesResult {
  data: Image[];
  total: number;
}

/**
 * Subconsulta correlacionada: comprueba si la imagen de la consulta externa
 * tiene al menos una anotación cuya categoría cumple la condición dada.
 * Se usa con `exists()` para que el filtrado ocurra dentro de SQL.
 */
function hasAnnotationMatching(categoryCondition: SQL | undefined) {
  return db
    .select({ one: sql`1` })
    .from(annotations)
    .innerJoin(categories, eq(annotations.categoryId, categories.id))
    .where(and(eq(annotations.imageId, images.id), categoryCondition));
}

/**
 * Traduce las opciones de búsqueda a una condición SQL combinada.
 * Todo el filtrado ocurre en la base de datos, nunca en memoria.
 */
function buildImageFilters(options: FindImagesOptions): SQL | undefined {
  const conditions: (SQL | undefined)[] = [];

  const statuses = options.status
    ? Array.isArray(options.status)
      ? options.status
      : [options.status]
    : null;
  if (statuses && statuses.length > 0) {
    const firstStatus = statuses[0];
    conditions.push(
      statuses.length === 1 && firstStatus
        ? eq(images.status, firstStatus)
        : inArray(images.status, statuses),
    );
  }

  if (options.dateFrom) {
    conditions.push(gte(images.createdAt, options.dateFrom));
  }
  if (options.dateTo) {
    conditions.push(lte(images.createdAt, options.dateTo));
  }

  // SPEC-SEARCH-001: operadores AND / OR sobre nombres de clase.
  if (options.classSearch && options.classSearch.terms.length > 0) {
    const { terms, operator } = options.classSearch;
    if (operator === 'AND') {
      // Un EXISTS por término: la imagen debe tener las N clases.
      for (const term of terms) {
        conditions.push(exists(hasAnnotationMatching(like(categories.name, term))));
      }
    } else {
      // Basta con que tenga alguna de las clases.
      conditions.push(exists(hasAnnotationMatching(inArray(categories.name, terms))));
    }
  }

  // Filtro por ids de categoría concretos.
  if (options.categoryIds && options.categoryIds.length > 0) {
    conditions.push(exists(hasAnnotationMatching(inArray(categories.id, options.categoryIds))));
  }

  const defined = conditions.filter((c): c is SQL => c !== undefined);
  return defined.length > 0 ? and(...defined) : undefined;
}

/**
 * Busca imágenes en MariaDB con filtros combinables (status, clases con
 * operadores, categorías, rango de fechas), paginación y orden por fecha
 * descendente. El conteo total usa las mismas condiciones.
 */
export async function findImages(options: FindImagesOptions): Promise<FindImagesResult> {
  const where = buildImageFilters(options);
  const offset = (options.page - 1) * options.pageSize;

  const [data, totalRows] = await Promise.all([
    db
      .select()
      .from(images)
      .where(where)
      .orderBy(desc(images.createdAt))
      .limit(options.pageSize)
      .offset(offset),
    db.select({ value: count() }).from(images).where(where),
  ]);

  return {
    data,
    total: totalRows[0]?.value ?? 0,
  };
}

/**
 * Devuelve todas las imágenes del dataset sin paginar. Usado por COCO.
 */
export async function findAllImages(): Promise<Image[]> {
  return db.select().from(images).orderBy(images.id);
}

/**
 * Conteo de anotaciones por imagen, para enriquecer el listado de búsqueda.
 */
export async function countAnnotationsForImages(imageIds: number[]): Promise<Map<number, number>> {
  if (imageIds.length === 0) return new Map();

  const rows = await db
    .select({ imageId: annotations.imageId, value: count() })
    .from(annotations)
    .where(inArray(annotations.imageId, imageIds))
    .groupBy(annotations.imageId);

  return new Map(rows.map((row) => [row.imageId, Number(row.value)]));
}

/**
 * Categorías distintas presentes en las anotaciones de cada imagen.
 */
export async function findCategoriesForImages(
  imageIds: number[],
): Promise<Map<number, { id: number; name: string; color: string }[]>> {
  if (imageIds.length === 0) return new Map();

  const rows = await db
    .selectDistinct({
      imageId: annotations.imageId,
      id: categories.id,
      name: categories.name,
      color: categories.color,
    })
    .from(annotations)
    .innerJoin(categories, eq(annotations.categoryId, categories.id))
    .where(inArray(annotations.imageId, imageIds))
    .orderBy(categories.name);

  const result = new Map<number, { id: number; name: string; color: string }[]>();
  for (const row of rows) {
    const list = result.get(row.imageId) ?? [];
    list.push({ id: row.id, name: row.name, color: row.color });
    result.set(row.imageId, list);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Métricas del dashboard, calculadas en SQL (SPEC-DASH-001)
// ---------------------------------------------------------------------------

export async function countImagesByStatus(): Promise<
  { status: ImageStatusValue; count: number }[]
> {
  const rows = await db
    .select({ status: images.status, value: count() })
    .from(images)
    .groupBy(images.status);
  return rows.map((row) => ({ status: row.status, count: Number(row.value) }));
}

export async function countAllAnnotations(): Promise<number> {
  const rows = await db.select({ value: count() }).from(annotations);
  return Number(rows[0]?.value ?? 0);
}

export async function countAllCategories(): Promise<number> {
  const rows = await db.select({ value: count() }).from(categories);
  return Number(rows[0]?.value ?? 0);
}

export async function countAnnotationsByCategory(): Promise<
  { categoryId: number; name: string; color: string; count: number }[]
> {
  const rows = await db
    .select({
      categoryId: annotations.categoryId,
      name: categories.name,
      color: categories.color,
      value: count(),
    })
    .from(annotations)
    .innerJoin(categories, eq(annotations.categoryId, categories.id))
    .groupBy(annotations.categoryId, categories.name, categories.color)
    .orderBy(desc(count()));
  return rows.map((row) => ({
    categoryId: row.categoryId,
    name: row.name,
    color: row.color,
    count: Number(row.value),
  }));
}

export async function findRecentImages(limit: number): Promise<Image[]> {
  return db.select().from(images).orderBy(desc(images.createdAt)).limit(limit);
}

/**
 * Busca una imagen por id. Devuelve null si no existe.
 */
export async function findImageById(id: number): Promise<Image | null> {
  const rows = await db.select().from(images).where(eq(images.id, id)).limit(1);
  return rows[0] ?? null;
}

/**
 * Actualiza el status de una imagen.
 */
export async function updateImageStatus(id: number, status: ImageStatusValue): Promise<void> {
  await db.update(images).set({ status }).where(eq(images.id, id));
}

/**
 * Elimina el registro de una imagen. Sus anotaciones se borran en cascada
 * (ver `onDelete: 'cascade'` en el schema).
 */
export async function deleteImageRow(id: number): Promise<void> {
  await db.delete(images).where(eq(images.id, id));
}
