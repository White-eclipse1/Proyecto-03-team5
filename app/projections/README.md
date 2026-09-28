# PCA / t-SNE — P2-36 Fase 2B

El stage independiente `projections` lee COCO mediante `ingestion.loader` y
procesa exactamente una imagen por `image_id`, ordenadas por ID. Anotaciones
repetidas no duplican puntos; etiquetas múltiples se deduplican y una imagen
sin anotaciones conserva `category_ids: []`. No consulta MariaDB/MinIO ni interpreta
IDs COCO como IDs del portal.

## Descriptor y algoritmos

**RGB reducido 16×16**, no un embedding semántico ni una representación aprendida:
Pillow aplica `ImageOps.exif_transpose`, convierte a RGB, redimensiona con LANCZOS
y aplana 16×16×3 píxeles a 768 features float64 normalizadas dividiendo por 255.
Se extraen una vez y se comparten entre ambos métodos.

`projections.yaml` es la configuración versionada, validada antes del cálculo.
PCA utiliza 2 componentes, `svd_solver=full`, `whiten=false`, sin StandardScaler.
Publica `explained_variance_ratio` de ambos componentes. Se rechazan menos de
3 muestras y entradas sin dos direcciones independientes de variación.

t-SNE utiliza 2 componentes, `random_state=42`, `init=pca`,
`learning_rate=auto`, `max_iter=1000`, `perplexity=30`, `method=barnes_hut`.
Se verificó la firma de PCA/TSNE de scikit-learn **1.9.1** del lockfile, con
NumPy **2.5.3**, Python 3.12. El resto de parámetros son los defaults de esa
versión fijada. Perplexity debe ser positiva y menor que N; no se ajusta
silenciosamente. La seed es independiente de splits.seed y no es criptográfica.
La repetibilidad se comprueba con tolerancia en el mismo entorno; no se promete
igualdad bit a bit entre BLAS, versiones o arquitecturas.

Estas coordenadas describen estructura visual gruesa (colores/disposición).
La cercanía no demuestra igualdad semántica, clase correcta ni ausencia de
leakage. La configuración actual conserva todas las imágenes COCO.

## Identidad y contrato

`presentation/projections_contracts.py` define `ProjectionsReport` v1.0 y el
frontend lo valida con `pipeline/projectionSchemas.ts`. Es independiente de
QualityReport/SplitsReport/VersionsReport. Campos:

- `schema_version`, `dataset_version`, `dataset_fingerprint`, `total_images`.
- `categories`: IDs/nombres COCO únicos.
- `features`: descriptor y parámetros efectivos.
- `pca`: `parameters`, `explained_variance_ratio`, `points`.
- `tsne`: `parameters`, `points`.
- Cada punto: `image_id`, `file_name`, `category_ids`, `x`, `y`.

Ambos métodos deben cubrir las mismas imágenes/filenames/etiquetas, sin IDs
repetidos, con coordenadas finitas y conteos completos. Filenames son relativos,
sin traversal ni rutas absolutas; no se construyen URLs a partir de ellos.

Fingerprint: SHA-256 del JSON UTF-8 canónico de
`{"coco": <COCO validado completo>, "image_sha256": [{"image_id": ..., "sha256": ...}]}`.
Las tres colecciones COCO se ordenan por ID; los digests SHA-256 de los bytes
codificados originales de imagen también se ordenan por ID. JSON usa claves
ordenadas, separadores `(',', ':')`, Unicode sin escape ASCII y prohíbe NaN.
Incluye los valores por defecto del modelo COCO validado; preserva el orden
interno de bbox/segmentación. No incluye directorio absoluto, mtime ni timestamps.
Cambiar la compresión original cambia su identidad aunque los píxeles se parezcan.

`dataset_version` del stage DVC viene exclusivamente de `projections.yaml`
y está registrado como parámetro DVC. El wrapper fija las rutas de input/output
de `dvc.yaml` mediante argumentos explícitos de Settings, ignorando overrides
DATASET_DIR/REPORTS_DIR/DATASET_VERSION sin alterar el entorno. El entry point
general fuera de DVC mantiene `storage.Settings` y DATASET_VERSION (default
`local-dev`). Nunca se infiere la versión del catálogo ni del número de imágenes.
El fingerprint identifica la entrada concreta.

## Generación y publicación

Desde la raíz, con `uv` y DVC disponibles:

```sh
uv sync --project app --locked
# Solo este stage: no ejecuta quality_gate ni crea releases.
dvc repro --single-item projections
```

Alternativamente, desde `app/`: `uv run python dvc_projections_stage.py`.
En la imagen Docker: `python -m presentation.projections`, usando las rutas
DATASET_DIR/REPORTS_DIR existentes. El wrapper DVC utiliza valores locales explícitos de Settings; sus placeholders
de infraestructura no se utilizan ni modifican variables de entorno.
No arranques el CMD por defecto de `app` para generar solo proyecciones, pues
ese CMD ejecuta el quality gate.

El stage publica `reports/projections.json` como output DVC `cache: false`,
no como métrica de calidad. No lee QualityReport ni exige status=passed.
`dvc.lock` registra código, configuración, dependencias y output del stage;
no hace falta subir el JSON al remote DVC. No se infiere un release nuevo.

La presentación valida antes de escribir un temporal en el mismo directorio y
hace replace atómico. Un error de imagen ausente/corrupta (identificado por ID),
COCO inválido, parámetros o cálculo impide publicar un reporte parcial y
preserva el reporte anterior. Si falla un recálculo, ese reporte anterior puede
seguir visible: comprobar su fingerprint antes de interpretarlo como actualizado.

La publicación `/reports/` existente sirve el JSON. `/pipeline/projections`
usa el fetch/ReportBoundary existente y Recharts: selector PCA/t-SNE, leyenda,
metadata y tooltip informativo. Multietiqueta y sin etiqueta tienen grupos
propios; cada imagen sigue siendo un único punto. No hay controles nuevos en
Settings ni enlaces COCO → anotación MariaDB.

## Pruebas

Desde `app/`: `uv run pytest tests/test_projections.py -q`.
Desde `frontend/`: `npm test -- tests/projections.test.tsx`.
Los tests Python generan imágenes sintéticas en memoria/tmp y calculan ambos
métodos realmente; el fixture JSON frontend existe exclusivamente en tests.

Los IDs del contrato de proyecciones son enteros matemáticos entre 0 y
9007199254740991, incluidos IDs de categorías y sus referencias. Se acepta 1.0
como 1, sin aceptar strings ni booleanos como enteros; ingestion no se modifica.
`whiten` acepta solo booleanos reales (true/false), nunca 0/1. La configuración
versionada continúa usando false; ningún parámetro de cálculo efectivo cambia.
