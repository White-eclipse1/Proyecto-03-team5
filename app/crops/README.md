# ML-01 — Clases dog/cat y extracción validada de crops COCO

El clasificador del Proyecto 3 no etiqueta imágenes completas: cada
bounding box COCO válida del release aprobado de P2 se convierte en una
muestra independiente. Una imagen con un perro y un gato produce dos
crops, uno por clase.

## Clases oficiales

| Clase | Índice del modelo | `category_id` en P2 v0.1.1 |
|-------|-------------------|----------------------------|
| `dog` | 0                 | 3                          |
| `cat` | 1                 | 4                          |

`crops/classes.py` (`CLASS_NAMES`, `CLASS_TO_INDEX`) es la única fuente de
verdad y está congelada. Los `category_id` no se fijan en código: se
resuelven por nombre contra las categorías del release. Un release que no
declare `dog` y `cat`, o que repita una de ellas con dos ids, se rechaza.

## Release de origen y procedencia

Los crops salen del release aprobado fijado en `crops/crops.yaml`
(parámetro DVC `dataset_version: v0.1.1`), nunca del dataset de trabajo
`local-dev`. Antes de escribir un solo crop, `dvc_crops_stage.py`
(`crops/release.py`):

1. Busca el release en `reports/versions.json` y carga su reporte de calidad
   congelado (`reports/releases/v0.1.1/quality.json`). Falla si el release no
   existe, si el reporte describe otra versión o si su status es `failed`.
2. Comprueba que las anotaciones en disco son las que congeló ese reporte:
   mismos `image_id` por clase (check `max_imbalance_ratio`) y mismo total de
   anotaciones (check `degenerate_boxes`).
3. Recalcula el md5 de directorio de `data/raw/images` y
   `data/raw/annotations` (mismo algoritmo que DVC 3) y exige que coincida con
   `data/raw/images.dvc` / `data/raw/annotations.dvc`. Si no coincide, corre
   `dvc pull -r prod data/raw/images.dvc data/raw/annotations.dvc`.

La versión, la ruta del reporte congelado, su status y las dos huellas DVC
(`md5`, `size`, `nfiles`) quedan en `reports/crops.json` → `source`, así se
puede demostrar de qué datos salió cada crop.

## Mínimo de imágenes originales por clase

`summary.images_per_class` cuenta `image_id` distintos con al menos un crop
aceptado por clase (una imagen con dos perros cuenta una vez), a diferencia
de `accepted_per_class`, que cuenta crops. El stage exige
`min_images_per_class: 300` (en `crops/crops.yaml`) para `dog` y `cat` y
falla sin escribir el reporte si alguna queda por debajo.

Resultado sobre v0.1.1:

| Clase | Crops | Imágenes originales |
|-------|-------|---------------------|
| `dog` | 325   | 300                 |
| `cat` | 343   | 301                 |

668 anotaciones, 668 aceptadas, 0 rechazadas. `dog` queda exactamente en el
mínimo: cualquier rechazo nuevo sobre una imagen de perro hace fallar el stage.

## Qué se excluye

Cada anotación se valida por separado (a diferencia de
`ingestion.models.CocoDataset`, que rechaza el dataset entero). Una
anotación excluida no produce crop ni aparece en el manifiesto, y queda
registrada en `rejections` con todos sus motivos:

| Motivo | Cuándo |
|--------|--------|
| `malformed_bbox` | bbox sin 4 valores numéricos finitos |
| `non_positive_width` / `non_positive_height` | `width <= 0` / `height <= 0` |
| `negative_x` / `negative_y` | origen fuera de la imagen |
| `exceeds_image_width` / `exceeds_image_height` | `x + width > W` / `y + height > H` |
| `out_of_bounds` | acompaña a cualquiera de los cuatro anteriores |
| `unknown_image_id` | `image_id` que no está en `images` |
| `unknown_category_id` | `category_id` que no está en `categories` |
| `unsupported_category` | categoría del release distinta de dog/cat (p. ej. `person`) |
| `missing_image` | el archivo de la imagen no existe |
| `unreadable_image` | el archivo no es una imagen válida |
| `image_size_mismatch` | los píxeles no miden lo que declara `images[]` |

Las cajas no se reparan ni se recortan contra el borde: un crop recortado
ya no correspondería a la bbox anotada.

## Salidas

- `data/crops/<clase>/img<image_id>-ann<annotation_id>.png`: salida DVC
  (cacheada, nunca en Git). PNG sin pérdida.
- `reports/crops.json`: contrato `CropReport` v1.0 (`crops/models.py`).
  - `crops`: manifiesto de aceptados. Cada registro conserva `crop_id`,
    `image_id`, `annotation_id`, `category_id`, `class_name`, `bbox`
    original `[x, y, width, height]`, `crop_box` en píxeles
    `[left, top, right, bottom]`, `source_file_name`, `crop_path` y
    `sha256`.
  - `rejections`: exclusiones con sus motivos.
  - `source`: release de origen y huellas DVC del dataset fuente.
  - `summary`: crops aceptados por clase, imágenes originales distintas
    por clase (`images_per_class`) y conteo por motivo de rechazo (una
    anotación con dos motivos cuenta en ambos).

`crop_box` usa `floor` en el origen y `ceil` en el extremo, así el crop
cubre la bbox completa aunque tenga decimales.

## Cómo correrlo

El stage `crops` de `dvc.yaml` depende de `reports/versions.json`, del
reporte congelado del release y de los parámetros de `crops/crops.yaml`; su
entrada en `dvc.lock` fija el md5 de los datos, del código y de las salidas.
Con el dataset descargado:

```bash
dvc pull -r prod data/raw/images.dvc data/raw/annotations.dvc
dvc repro crops
dvc status crops   # "Data and pipelines are up to date."
```

Verificación manual (DoD): genera una hoja con cada imagen original, su
bbox dibujada y el crop al lado, y revísala a ojo:

```bash
cd app
uv run python -m crops.preview --limit 24 --out ../reports/crops-preview.png
```

Tests: `uv run pytest tests/test_crops.py`.
