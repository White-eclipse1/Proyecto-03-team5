import { z } from 'zod';

/**
 * SPEC-ANNOT-001 ÔÇö Validaci├│n de bounding boxes
 *
 * Reglas de negocio (ver specs/annotation.spec.md):
 *  1. Origen espacial: bboxX >= 0, bboxY >= 0
 *  2. Dimensiones m├¡nimas: bboxWidth > 0, bboxHeight > 0
 *  3. L├¡mites del canvas: bboxX + bboxWidth <= imageWidth,
 *                         bboxY + bboxHeight <= imageHeight
 *  4. category_id debe ser un entero positivo v├ílido
 *  5. El ├írea la calcula el backend: area = bboxWidth * bboxHeight
 */

/**
 * Esquema base de una bounding box sin validaci├│n de l├¡mites de imagen.
 * Se usa para validar la forma de los datos antes de consultar la BD.
 */
export const bboxBaseSchema = z.object({
  categoryId: z.number().int().positive({ message: 'El category_id debe ser un entero positivo.' }),
  bboxX: z.number().nonnegative({ message: 'bboxX debe ser >= 0.' }),
  bboxY: z.number().nonnegative({ message: 'bboxY debe ser >= 0.' }),
  bboxWidth: z.number().positive({ message: 'bboxWidth debe ser > 0.' }),
  bboxHeight: z.number().positive({ message: 'bboxHeight debe ser > 0.' }),
});

export type BboxBaseInput = z.infer<typeof bboxBaseSchema>;

/**
 * Valida que la caja no se salga de los l├¡mites de la imagen.
 * Recibe las dimensiones reales de la imagen (le├¡das de MariaDB).
 */
export function createBboxWithBoundsSchema(imageWidth: number, imageHeight: number) {
  return bboxBaseSchema
    .refine((d) => d.bboxX + d.bboxWidth <= imageWidth, {
      message: `La caja excede el ancho de la imagen (${imageWidth}px).`,
      path: ['bboxWidth'],
    })
    .refine((d) => d.bboxY + d.bboxHeight <= imageHeight, {
      message: `La caja excede el alto de la imagen (${imageHeight}px).`,
      path: ['bboxHeight'],
    });
}

export type BboxWithBoundsInput = BboxBaseInput;

/** Resultado de validaci├│n con mensaje de error opcional. */
export interface ValidationResult {
  success: boolean;
  error?: string;
}

/**
 * Valida el payload de una anotaci├│n sin conocer las dimensiones de imagen.
 * ├Ütil para pruebas unitarias de la forma del dato.
 */
export function validateBboxBase(input: unknown): ValidationResult {
  const result = bboxBaseSchema.safeParse(input);
  return {
    success: result.success,
    error: result.success ? undefined : result.error.issues[0]?.message,
  };
}

/**
 * Validaci├│n completa: forma + l├¡mites del canvas.
 * Debe llamarse desde el servicio, despu├®s de obtener la imagen de BD.
 */
export function validateBboxWithBounds(
  input: unknown,
  imageWidth: number,
  imageHeight: number,
): ValidationResult {
  const schema = createBboxWithBoundsSchema(imageWidth, imageHeight);
  const result = schema.safeParse(input);
  return {
    success: result.success,
    error: result.success ? undefined : result.error.issues[0]?.message,
  };
}

/**
 * SPEC-VALID-001 ÔÇö Esquemas de la frontera HTTP.
 *
 * Estos esquemas validan el dato externo completo (incluido `imageId`) antes
 * de que llegue a la capa de datos. Los tipos se infieren del esquema.
 */

/** Identificador de recurso: entero positivo. Acepta el string de un route param. */
export const idParamSchema = z.coerce
  .number({ message: 'El id debe ser un n├║mero.' })
  .int({ message: 'El id debe ser un entero.' })
  .positive({ message: 'El id debe ser mayor que cero.' });

/**
 * Body de `POST /images/:imageId/annotations`.
 *
 * La imagen destino viaja en la ruta, no en el body. `iscrowd` es opcional
 * y por defecto `false` (objeto individual, seg├║n el est├índar COCO).
 */
export const createAnnotationForImageSchema = bboxBaseSchema.extend({
  iscrowd: z.boolean().optional(),
});

export type CreateAnnotationForImageBody = z.infer<typeof createAnnotationForImageSchema>;

/**
 * Body de `PATCH /annotations/:annotationId`.
 *
 * Todos los campos son opcionales: la UI puede mover una caja sin cambiar su
 * clase, o reclasificarla sin moverla. Se exige al menos un campo.
 */
export const patchAnnotationSchema = createAnnotationForImageSchema
  .partial()
  .refine((data) => Object.keys(data).length > 0, {
    message: 'Debe enviarse al menos un campo para actualizar.',
  });

export type PatchAnnotationBody = z.infer<typeof patchAnnotationSchema>;

/**
 * Combina un patch parcial con los valores actuales de la anotación.
 *
 * Cada campo ausente en el patch conserva su valor existente — incluido
 * `categoryId`: la UI puede mover o redimensionar una caja sin
 * reclasificarla, así que un patch sin `categoryId` nunca debe alterarlo.
 */
export function mergeAnnotationPatch(
  existing: BboxBaseInput,
  patch: PatchAnnotationBody,
): BboxBaseInput {
  return {
    categoryId: patch.categoryId ?? existing.categoryId,
    bboxX: patch.bboxX ?? existing.bboxX,
    bboxY: patch.bboxY ?? existing.bboxY,
    bboxWidth: patch.bboxWidth ?? existing.bboxWidth,
    bboxHeight: patch.bboxHeight ?? existing.bboxHeight,
  };
}

/** Estados v├ílidos del ciclo de anotaci├│n de una imagen. */
export const imageStatusValues = ['pending', 'in_progress', 'completed'] as const;

/**
 * Body de `PATCH /images/:imageId/status`.
 *
 * Solo se permiten las transiciones que dispara la UI de anotaci├│n:
 * `in_progress` al empezar a anotar y `completed` al finalizar.
 */
export const imageStatusTransitionSchema = z.object({
  status: z.enum(['in_progress', 'completed'], {
    message: 'El status debe ser "in_progress" o "completed".',
  }),
});

export type ImageStatusTransitionBody = z.infer<typeof imageStatusTransitionSchema>;

/**
 * Normaliza un query param que puede venir como lista separada por comas,
 * por ejemplo `?status=pending,in_progress` o `?categories=1,2`.
 *
 * Express tambi├®n puede entregar un arreglo si el param se repite.
 * Devuelve `undefined` cuando no hay valores ├║tiles.
 */
function splitList(value: unknown): string[] | undefined {
  if (value === undefined || value === null) return undefined;

  const raw = Array.isArray(value) ? value.map(String) : String(value).split(',');
  const items = raw.map((item) => item.trim()).filter((item) => item.length > 0);

  return items.length > 0 ? items : undefined;
}

/**
 * Esquema para los query params de `GET /images/search`.
 *
 * Refleja exactamente lo que env├¡a el frontend: `q`, `categories`, `status`,
 * `dateFrom`, `dateTo`, `page` y `pageSize`.
 */
export const imageSearchSchema = z.object({
  /** SPEC-SEARCH-001: expresi├│n de clases, ej. "car AND person". */
  q: z.string().optional(),
  /** B├║squeda por nombre de archivo (LIKE). */
  filename: z.string().optional(),
  /** Uno o varios estados separados por coma. */
  status: z.preprocess(
    splitList,
    z
      .array(
        z.enum(imageStatusValues, {
          message: `Cada status debe ser uno de: ${imageStatusValues.join(', ')}.`,
        }),
      )
      .optional(),
  ),
  /** Uno o varios ids de categor├¡a separados por coma. */
  categories: z.preprocess(
    splitList,
    z
      .array(
        z.coerce
          .number({ message: 'Cada categor├¡a debe ser un n├║mero.' })
          .int({ message: 'Cada categor├¡a debe ser un entero.' })
          .positive({ message: 'Cada categor├¡a debe ser mayor que cero.' }),
      )
      .optional(),
  ),
  dateFrom: z.coerce.date({ message: 'dateFrom debe tener formato de fecha válido.' }).optional(),
  dateTo: z.coerce.date({ message: 'dateTo debe tener formato de fecha válido.' }).optional(),
  page: z.coerce.number().int().positive().default(1),
  pageSize: z.coerce.number().int().positive().max(100).default(24),
});

export type ImageSearchInput = z.infer<typeof imageSearchSchema>;
