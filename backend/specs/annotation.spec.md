# SPEC-ANNOT-001 ÔÇö Validaci├│n de bounding boxes

## Objetivo

Garantizar que toda bounding box creada o modificada en el portal de anotaci├│n
cumpla con las reglas espaciales y de integridad referencial antes de persistirse
en la base de datos.

## Reglas de negocio

1. El origen de la caja debe estar dentro de la imagen: `bboxX >= 0` y `bboxY >= 0`.
2. Las dimensiones deben ser positivas: `bboxWidth > 0` y `bboxHeight > 0`.
3. La caja no puede salirse del canvas: `bboxX + bboxWidth <= image.width`
   y `bboxY + bboxHeight <= image.height`.
4. El `categoryId` debe referirse a una categor├¡a existente en la base de datos.
5. El ├írea se calcula en backend: `area = bboxWidth ├ù bboxHeight`.
   El cliente nunca env├¡a ni el backend conf├¡a en un valor de ├írea externo.
6. Si la validaci├│n falla, la anotaci├│n no debe persistirse.
7. Ninguna caja puede quedar sin categor├¡a v├ílida.
8. `PATCH /annotations/:id` acepta un patch parcial: un campo ausente en el
   patch conserva su valor actual. En particular, mover o redimensionar una
   caja sin incluir `categoryId` en el body nunca debe cambiar su
   categor├¡a, y enviar solo `categoryId` nunca debe alterar su geometr├¡a.

## Flujo esperado

UI ÔåÆ Logic (validateBboxWithBounds + assertCategoryExists) ÔåÆ Data (createAnnotation)

## Transiciones de estado de imagen

- Al crear la primera bounding box en una imagen `pending` ÔåÆ `in_progress`.
- Al eliminar todas las cajas de una imagen `in_progress` ÔåÆ `pending`.
- El usuario debe pulsar "Finalizar" expl├¡citamente para pasar a `completed`.
- Una imagen `completed` no puede completarse de nuevo (idempotente).
