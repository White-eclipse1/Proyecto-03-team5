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

Clases excluidas: ninguna. El COCO de v0.1.1 solo declara `dog` (325
anotaciones, 300 imágenes) y `cat` (343 anotaciones, 301 imágenes), y ambas
cumplen el mínimo de 300 imágenes originales. Las clases se fijaron el
2026-09-28, antes de cualquier entrenamiento o evaluación, y no se cambian
según el resultado de prueba. Si un release futuro trae otra categoría (p. ej.
`person`), sus anotaciones se excluyen como `unsupported_category`.

`crops/classes.py` (`CLASS_NAMES`, `CLASS_TO_INDEX`) es la única fuente de
verdad y está congelada. Los `category_id` no se fijan en código: se
resuelven por nombre contra las categorías del release. Un release que no
declare `dog` y `cat`, o que repita una de ellas con dos ids, se rechaza.

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

## Release de origen (OPS-01)

El stage no lee `data/raw` a ciegas: `releases/selection.yaml` fija el
release de P2 (`v0.1.1`) y `P2ReleaseService.verify_release` (OPS-01) lo
valida contra `releases/p2_releases.json` y el registro oficial
`reports/versions.json` antes de borrar o generar crops:

- el release existe y su reporte de calidad congelado
  (`reports/releases/v0.1.1/quality.json`) es de esa versión y no está en
  `failed`;
- los hashes de `data/raw/images.dvc` y `data/raw/annotations.dvc` son los
  del release.

Las rutas de COCO e imágenes salen del release verificado, y
`reports/crops.json` → `provenance` guarda `release_version`,
`images_dvc_hash`, `annotations_dvc_hash` y `quality_report`. La entrada
`crops` de `dvc.lock` registra además el md5 de `data/raw/images` y
`data/raw/annotations` en disco al generar los crops, que coincide con esos
hashes (`test_committed_crop_report_is_traceable_to_the_dvc_release`).

## Mínimo de imágenes originales por clase

`summary.accepted_images_per_class` cuenta `image_id` distintos con al menos
un crop aceptado (una imagen con dos perros cuenta una vez);
`accepted_per_class` cuenta crops. El stage exige
`MIN_IMAGES_PER_CLASS = 300` (`crops/classes.py`) para `dog` y `cat` y falla
sin escribir el reporte si alguna clase queda por debajo.

Resultado sobre v0.1.1: 668 anotaciones, 668 aceptadas, 0 rechazadas.

| Clase | Crops | Imágenes originales |
|-------|-------|---------------------|
| `dog` | 325   | 300                 |
| `cat` | 343   | 301                 |

`dog` queda exactamente en el mínimo: cualquier rechazo nuevo sobre una
imagen de perro hace fallar el stage.

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
  - `provenance`: release de origen, hashes DVC y reporte de calidad.
  - `summary`: crops aceptados por clase, imágenes originales distintas
    por clase (`accepted_images_per_class`) y conteo por motivo de rechazo
    (una anotación con dos motivos cuenta en ambos).

`crop_box` usa `floor` en el origen y `ceil` en el extremo, así el crop
cubre la bbox completa aunque tenga decimales.

## Cómo correrlo

El stage `crops` de `dvc.yaml` depende del marcador del quality gate, de
la procedencia de OPS-01 y de `releases/selection.yaml`. Las dependencias
de `crops/` se listan archivo por archivo para que `__pycache__` no entre en
el hash de `dvc.lock`. Con el dataset descargado:

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

Tests: `uv run pytest tests/test_crops.py tests/test_crops_release_integration.py`.

Evidencia sobre el release real (verificación manual, Agent Test y
mutaciones): [`tests/evidence/ml-01-crops.md`](../tests/evidence/ml-01-crops.md).
