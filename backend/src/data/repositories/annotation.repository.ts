import { count, eq } from 'drizzle-orm';
import { db } from '../db/client.js';
import { type Annotation, annotations, categories, type NewAnnotation } from '../db/schema.js';

/**
 * Forma de anotación enriquecida con un resumen de su categoría, tal como
 * la espera el frontend (ver annotationSchema en el frontend).
 */
export interface AnnotationWithCategory {
  id: number;
  imageId: number;
  categoryId: number;
  bboxX: number;
  bboxY: number;
  bboxWidth: number;
  bboxHeight: number;
  area: number;
  iscrowd: boolean;
  category: {
    id: number;
    name: string;
    color: string;
  };
}

function selectAnnotationsWithCategory() {
  return db
    .select({
      id: annotations.id,
      imageId: annotations.imageId,
      categoryId: annotations.categoryId,
      bboxX: annotations.bboxX,
      bboxY: annotations.bboxY,
      bboxWidth: annotations.bboxWidth,
      bboxHeight: annotations.bboxHeight,
      area: annotations.area,
      iscrowd: annotations.isCrowd,
      category: {
        id: categories.id,
        name: categories.name,
        color: categories.color,
      },
    })
    .from(annotations)
    .innerJoin(categories, eq(annotations.categoryId, categories.id));
}

/**
 * Lista las anotaciones de una imagen, con su categoría incluida.
 */
export async function listAnnotationsByImage(imageId: number): Promise<AnnotationWithCategory[]> {
  return selectAnnotationsWithCategory().where(eq(annotations.imageId, imageId));
}

/**
 * Busca una anotación por id, con su categoría incluida.
 */
export async function findAnnotationById(id: number): Promise<AnnotationWithCategory | null> {
  const rows = await selectAnnotationsWithCategory().where(eq(annotations.id, id)).limit(1);
  return rows[0] ?? null;
}

/**
 * Crea una anotación y devuelve su id.
 */
export async function createAnnotationRow(annotation: NewAnnotation): Promise<number> {
  const result = await db.insert(annotations).values(annotation).$returningId();
  const created = result[0];

  if (!created) {
    throw new Error('No se pudo guardar la anotación en MariaDB.');
  }

  return created.id;
}

/**
 * Actualiza (parcialmente) una anotación existente.
 */
export async function updateAnnotationRow(
  id: number,
  changes: Partial<NewAnnotation>,
): Promise<void> {
  await db.update(annotations).set(changes).where(eq(annotations.id, id));
}

/**
 * Elimina una anotación.
 */
export async function deleteAnnotationRow(id: number): Promise<void> {
  await db.delete(annotations).where(eq(annotations.id, id));
}

/**
 * Cuenta las anotaciones de una imagen. Usado para la máquina de estados.
 */
export async function countAnnotationsByImage(imageId: number): Promise<number> {
  const rows = await db
    .select({ value: count() })
    .from(annotations)
    .where(eq(annotations.imageId, imageId));
  return Number(rows[0]?.value ?? 0);
}

/**
 * Devuelve todas las anotaciones del dataset (fila cruda). Usado por COCO.
 */
export async function findAllAnnotationRows(): Promise<Annotation[]> {
  return db.select().from(annotations).orderBy(annotations.id);
}
