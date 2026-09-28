# SPEC-COCO-001 ÔÇö Exportaci├│n del dataset en formato COCO

## Objetivo

Generar un archivo JSON v├ílido en formato COCO (Common Objects in Context)
que contenga todas las im├ígenes, anotaciones y categor├¡as del dataset,
descargable como un ├║nico archivo para alimentar el pipeline de entrenamiento
de la siguiente fase.

## Estructura del JSON

El documento COCO tiene exactamente tres secciones de nivel superior:

```json
{
  "images": [ ... ],
  "annotations": [ ... ],
  "categories": [ ... ]
}
```

### images

Cada imagen se representa con:

| Campo       | Tipo   | Origen                          |
|-------------|--------|---------------------------------|
| id          | number | images.id                       |
| file_name   | string | images.filename                 |
| width       | number | images.width                    |
| height      | number | images.height                   |

### annotations

Cada bounding box se representa con:

| Campo        | Tipo     | Origen / Regla                                       |
|--------------|----------|------------------------------------------------------|
| id           | number   | annotations.id                                       |
| image_id     | number   | annotations.imageId (referencia a images.id)         |
| category_id  | number   | annotations.categoryId (referencia a categories.id)  |
| bbox         | number[] | `[bboxX, bboxY, bboxWidth, bboxHeight]` en p├¡xeles     |
| area         | number   | `bboxWidth ├ù bboxHeight`, coherente con el bbox        |
| iscrowd      | 0 \| 1   | derivado de annotations.isCrowd (boolean ÔåÆ 0/1)      |
| segmentation | number[] | arreglo vac├¡o `[]` (no se soportan pol├¡gonos)        |

### categories

Cada categor├¡a se representa con:

| Campo | Tipo   | Origen          |
|-------|--------|-----------------|
| id    | number | categories.id   |
| name  | string | categories.name |

## Reglas de negocio

1. Los `id` deben ser enteros consistentes entre las tres secciones:
   todo `annotations[].image_id` debe existir en `images[].id`, y todo
   `annotations[].category_id` debe existir en `categories[].id`.
2. El `bbox` debe estar en el orden exacto `[x, y, width, height]`
   en p├¡xeles absolutos de la imagen (nunca normalizado).
3. El `area` debe ser coherente: `area === bbox[2] ├ù bbox[3]`.
4. El campo `iscrowd` siempre debe estar presente, con valor `0` o `1`.
5. La exportaci├│n incluye todas las im├ígenes, incluso las que no tienen
   anotaciones (el pipeline de entrenamiento las necesita como negativos).
6. El resultado debe ser descargable como un ├║nico archivo `.json`.

## Flujo esperado

UI (GET /export/coco) ÔåÆ Logic (buildCocoDataset) ÔåÆ Data (repositorios)
