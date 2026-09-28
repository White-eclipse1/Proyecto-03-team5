# P2-12 — Contratos JSON v1.0

`contracts.py` define los contratos públicos con Pydantic v2. Los ejemplos
manuales y ficticios están en `examples/quality.json`, `examples/splits.json`
y `examples/versions.json`. No son resultados de un pipeline ejecutado ni releases
publicadas. No contienen credenciales, URLs de servicios ni rutas del equipo.

Los campos son obligatorios salvo `details`, que puede omitirse y toma `{}`.
Los modelos rechazan campos adicionales, conversiones de strings a números/bools,
números no finitos y valores fuera de los rangos definidos. `details` es la única
extensión abierta: sus claves dependen del check y sus valores deben ser JSON.
No se deben colocar objetos Python, secretos ni rutas locales en ese campo.

## Campos comunes

| Campo | Tipo y significado |
|---|---|
| `schema_version` | Literal `"1.0"`, versión del contrato, no del dataset. |
| `dataset_version` | Identificador opaco, no vacío, con letras ASCII, dígitos, puntos, guiones o guiones bajos; comienza con letra o dígito. Une los documentos de un mismo dataset. No implica un commit Git o hash DVC. |

La forma v1.0 queda congelada: cambiar campos, tipos o semántica requiere revisar
la versión del contrato y sus consumidores. Los schemas JSON se pueden obtener
con `QualityReport.model_json_schema()`, `SplitsReport.model_json_schema()` y
`VersionsReport.model_json_schema()` sin generar archivos ni acceder a servicios.
Las comprobaciones entre campos de los validadores Pydantic complementan el JSON
Schema: un validador JSON Schema genérico no ejecuta esos validadores Python.

## quality.json — QualityReport

| Campo | Uso |
|---|---|
| `schema_version`, `dataset_version` | Identifican el contrato y el dataset evaluado. |
| `status` | `passed`, `warning` o `failed`, resultado global entregado por la compuerta. |
| `checks` | Lista no vacía de checks ejecutados, con nombres únicos. No implica que todos los checks posibles hayan sido ejecutados. |
| `checks[].check_name` | Identificador estable del check; no es un texto traducido de UI. |
| `checks[].passed` | Booleano con el resultado ya calculado por el analizador/compuerta. |
| `checks[].metric_value` | Número finito; su unidad y significado dependen del check. |
| `checks[].details` | Objeto JSON con evidencia adicional; la UI puede mostrarlo sin asumir claves fijas. |
| `checks[].action` | `warn` o `fail`, efecto de un check no aprobado, suministrado por la capa de políticas. |

`QualityCheck` extiende `AnalyzerResult` sin modificarlo. Los checks existentes
se pueden adaptar con `QualityCheck.model_validate({**result.model_dump(),
"action": action})`; obtener `action` y calcular `status` corresponde a políticas,
no a presentación. `passed` significa que no hay incumplimientos; `warning`,
que hay advertencias pero no bloqueos; `failed`, que hay un bloqueo. El contrato
no calcula ni impone esa agregación para no implementar la compuerta de otro ticket.

## P2-30 — Valor y criterio por check (compatible con v1.0)

El gate conserva `metric_value`, `passed`, `action` y el `status` global.
Publica en `details.criterion` el campo evaluado (`metric: "metric_value"`),
el `threshold` y el `operator`. Es una extensión documentada de `details`,
no un campo nuevo de `QualityCheck` ni una nueva versión del contrato.
El esquema genérico permite estas claves; las pruebas del gate garantizan
su presencia y significado para los siete checks reales.

| Check | Valor | Criterio de cumplimiento |
|---|---|---|
| `min_images_per_class` | Mínimo de imágenes distintas por categoría | `>=` umbral mínimo |
| `max_imbalance_ratio` | Mayoría/minoría | Ratio definido **y** `<=` umbral máximo |
| `max_small_object_ratio` | Proporción de cajas pequeñas | `<=` proporción máxima |
| `degenerate_boxes` | Cantidad de anotaciones inválidas | `<=` cantidad permitida |
| `duplicate_similarity_threshold` | Cantidad de pares detectados | `== 0` pares |
| `spatial_bias` | Menor desviación estándar de centros normalizados X/Y | `>=` dispersión mínima |
| `cross_split_leakage` | Pares casi duplicados entre splits | `<=` umbral permitido |

Para desbalance, `criterion.requires_ratio_defined=true` exige además
`details.ratio_defined=true`. Con una categoría vacía, el valor cero es un
marcador finito, el ratio está indefinido y `passed=false`: comparar únicamente
el cero con el umbral produciría una interpretación incorrecta.

Para objetos pequeños, `details.small_box_detection` publica `width_px`,
`height_px`, `operator: "<"` y `combination: "and"`: ambos lados deben estar
bajo sus límites. Estos tamaños no son el umbral de proporción del criterio.

Para pHash, `details.similarity_threshold` es el umbral de **detección**,
con `similarity_operator: ">="` y fórmula
`similarity_formula: "1 - hamming_distance / hash_bits"`. Cada par conserva
su similitud y distancia. `criterion.threshold=0` describe la ausencia de
pares requerida por el analizador existente: nunca se compara una cantidad
con la similitud 0.94. No se evalúan splits.

`build_quality_report()` recibe una única instancia `QualityPolicy` y construye
con ella las configuraciones de los analizadores y los criterios publicados.
No vuelve a cargar YAML. Se retiró su argumento interno `policy_path`; los
llamadores cargan la política antes de invocarlo. Los algoritmos no cambian.

`passed=true` indica cumplimiento; si es falso, `action=warn` advierte y
`action=fail` bloquea. El estado global sigue siendo `failed` si existe un
bloqueo, `warning` si solo hay advertencias y `passed` si todo cumple.

El ejemplo manual histórico conserva seis checks con datos ficticios:
400 imágenes mínimas por clase, ratio 1, proporción pequeña 0.45, cero cajas
inválidas, un par similar y dispersión 0.20. Su estado global es `warning`.
No incluye `cross_split_leakage`; sigue siendo un reporte v1.0 válido. Desde P2-53
el gate calcula siete checks, incluido leakage de las asignaciones actuales.

## splits.json — SplitsReport

| Campo | Uso |
|---|---|
| `schema_version`, `dataset_version` | Identifican contrato y dataset. |
| `total_images` | Entero positivo de imágenes incluidas en la partición. |
| `splits` | Objeto con exactamente `train`, `validation` y `test`. |
| `splits.<nombre>.image_count` | Entero no negativo; una imagen se cuenta una sola vez en el resumen. |
| `splits.<nombre>.ratio` | Proporción real de imágenes, entre 0 y 1; no porcentaje ni proporción solicitada. |

Los conteos deben sumar `total_images`; cada proporción debe coincidir con
`image_count / total_images` con tolerancia absoluta de `1e-6`. Se admiten splits
vacíos con ambos valores en cero, pero no una partición de cero imágenes. El
contrato solo valida el resumen: no genera particiones ni verifica estratificación,
solapamientos reales o leakage. La capa `splits` puede usar `val` internamente,
pero el nombre público congelado es `validation`.

## versions.json — VersionsReport

| Campo | Uso |
|---|---|
| `schema_version` | Versión del catálogo. |
| `releases` | Lista de releases disponibles; puede estar vacía. No tiene orden semántico ni designa automáticamente una versión como la más reciente. |
| `releases[].dataset_version` | Identificador único dentro del catálogo. |
| `releases[].quality_file` | Referencia relativa a un archivo llamado `quality.json`. |
| `releases[].splits_file` | Referencia relativa a un archivo llamado `splits.json`. |

Las referencias se resuelven respecto del directorio que contiene `versions.json`.
Permiten subdirectorios de nombres ASCII alfanuméricos con `.`, `_` y `-`; prohíben
rutas absolutas, segmentos vacíos, `.`/`..`, URLs y separadores Windows. Para más
releases pueden usarse referencias como `demo-v2/quality.json`. El modelo valida
su forma, sin abrir los archivos: el productor del catálogo debe garantizar su
existencia y que `dataset_version` coincida en ambos reportes.

El ejemplo cataloga `demo-v1.0.0` y referencia los otros dos JSON de la misma
carpeta. No duplica el estado de calidad o los conteos en el catálogo para evitar
datos divergentes. No contiene URIs S3, ETags, firmas, semver obligatorio ni una
integración inventada con el frontend.

## Pruebas y límites

Desde `app/`, con las dependencias de desarrollo declaradas instaladas:

```bash
pytest
ruff check .
ruff format --check .
```

Las pruebas usan `unittest.TestCase`, que pytest descubre, y `model_validate`
de Pydantic v2. También pueden ejecutarse sin pytest con
`python -m unittest tests.test_json_contracts -v` (requiere Pydantic v2).
Verifican los tres ejemplos, referencias reales entre ellos, round trip y rechazos
de documentos inválidos. No importan el entrypoint `presentation.main` ni ejecutan
conexiones del código preexistente.

P2-11 deberá aportar resultados y un resumen compatibles; este ticket no implementa
sus algoritmos. P2-14 deberá acordar con estos contratos la identificación de releases
y ubicación de reportes. No se presupone su API: ambos tickets no están implementados
en esta copia. Cualquier necesidad nueva debe discutirse antes de cambiar v1.0.

## P2-22/23/24 — La compuerta de calidad end-to-end (`gate.py`)

`gate.py` es el primer código que efectivamente corre el pipeline contra el
dataset real: `ingestion/loader.py` junta los `annotations-lote-*.json` de
`data/raw/annotations/` en un `CocoDataset` validado, corre los checks de
calidad (`imbalance`, `small_objects`, `invalid_boxes`,
`duplicates`, `spatial_bias`, `cross_split_leakage`), arma un `QualityReport` real (no el ejemplo)
y lo escribe en `REPORTS_DIR/quality.json`.

- `min_images_per_class` no es su propio analizador (vive dentro de
  `imbalance.py`, ver `details.classes_below_minimum`); el gate lo deriva
  como check independiente porque `quality.yaml` le da una severidad
  distinta (`fail`) a la de `max_imbalance_ratio` (`warn`).
- `analyze_duplicates()` usa el check_name interno `duplicate_images`
  (ver `analyzers/duplicates.py` y su test); el gate lo renombra a
  `duplicate_similarity_threshold` en el reporte — el nombre de la política
  que en verdad evalúa — sin tocar el analizador ya mergeado.
- `spatial_bias` (P2-29) mide dispersión del centro normalizado de las
  cajas; el ticket original no definía el algoritmo, se confirmó con Andy
  (ver `analyzers/README.md`). `quality.yaml` gana una clave nueva
  (`min_spatial_dispersion`) que no existía para ningún analizador previo.
- `cross_split_leakage` cuenta pares pHash únicos repartidos entre los splits
  calculados en la misma evaluación. Usa threshold/action de su política.
  Siete checks se publican sin cambiar el contrato v1.0.
- `main()` devuelve `1` si algún check con `action: fail` no pasó — pensado
  para encadenarse como dependencia dura de la siguiente etapa
  (`python -m presentation.gate; echo "exit=$?"`).
- `presentation/main.py` corre el gate una vez al arrancar el contenedor
  `app` y sigue vivo aunque falle (loggea y continúa) — el contenedor es un
  servicio de larga duración, el gate es un paso de pipeline con su propio
  exit code; no se confunden.

`DATASET_DIR`/`REPORTS_DIR` (ver `storage/settings.py`) apuntan a los
volúmenes montados en `docker-compose.yml`, no a rutas calculadas desde
`__file__`: el Dockerfile aplana `app/` a `/app`, así que una ruta relativa
al código (`../../data/raw`) dejaría de tener sentido ahí. El frontend lee
`REPORTS_DIR` como estático servido por nginx en `/reports/*.json`
(volumen compartido `reports_data`, sin backend HTTP nuevo) — ver
`frontend/docker/nginx.conf` y `frontend/src/pipeline/dataSource.ts`.

Con el dataset real actual, la compuerta **bloquea de verdad**: `person`/
`car` están sembrados como categorías pero nunca se han anotado (0
imágenes cada una), así que `min_images_per_class` (umbral 300, severidad
`fail`) nunca pasa mientras existan categorías declaradas sin imágenes.
Esto no es un bug del gate — es el reflejo honesto de que el dataset
todavía no cumple la meta M3 del curso.

```bash
uv run pytest tests/test_loader.py tests/test_gate.py
DATABASE_URL=... MINIO_ENDPOINT=... MINIO_PORT=... MINIO_ACCESS_KEY=... \
MINIO_SECRET_KEY=... MINIO_BUCKET=... DATASET_DIR=../data/raw REPORTS_DIR=/tmp/reports \
uv run python -m presentation.gate
```

## P2-31 — "Warn no bloquea pero queda registrado"

Ticket de verificación, no de implementación nueva: el comportamiento que
pide (un `warn` no detiene el proceso, queda registrado en `quality.json`,
el estado final lo refleja, y sigue diferenciado de un `fail` real) ya
salió como consecuencia necesaria de cómo `build_quality_report()` calcula
`status` arriba — nadie lo notó como su propio criterio hasta este ticket.
No se cambió ninguna línea de lógica; se agregaron los dos tests que
faltaban para probarlo explícitamente (el resto de los tests del gate solo
cubrían el camino "todo pasa" o "algo `fail` no pasa"):

- `test_warn_action_check_failing_sets_warning_status_and_does_not_block`:
  un dataset con `max_imbalance_ratio` (acción `warn`) fallando y todo lo
  demás (`action: fail`) pasando produce `status="warning"`, no `"failed"`,
  y el check sigue apareciendo en `checks` con su `action` correcta.
- `test_warn_only_report_does_not_block_main_exit_code`: con `run()`
  mockeado (mismo patrón que `test_main_returns_nonzero_exit_code_when_failed`,
  sin tocar `policies/quality.yaml` real ni repetir I/O), `main()` devuelve
  `0` cuando `status="warning"`.

## P2-34 — Copilot MVP: servidor MCP de solo lectura (`mcp_server.py`)

`mcp_server.py` usa el SDK oficial `mcp` (`modelcontextprotocol/python-sdk`,
`MCPServer`/`ToolAnnotations` — la API se renombró de `FastMCP` a
`MCPServer` en la versión 2.x instalada aquí). Expone 5 herramientas de
solo lectura para que el futuro agente del Copilot consulte el dataset y
sus resultados de calidad, sin poder modificar nada:

| Herramienta | Qué devuelve |
|---|---|
| `get_dataset_summary` | Total de imágenes/anotaciones y conteo por categoría, calculado desde `data/raw/annotations/` real (vía `ingestion.loader`, igual que el gate), más su `dataset_version`; `{"available": false, ...}` si el dataset no está en este entorno (falta `dvc pull`). |
| `get_quality_report` | Sin argumentos, el `quality.json` vigente que escribió `gate.py` (copia de trabajo, `dataset_version` suele ser `local-dev`). Con `dataset_version` (p. ej. `v0.1.0`), el reporte congelado de ese release. `{"available": false, "reason": ...}` si no existe. |
| `get_check_result` | Un check específico por nombre del `quality.json` vigente, con el `dataset_version` de ese reporte; `{"available": false, ...}` si no existe ese check o el reporte no existe. |
| `get_splits_report` | Los splits de un release cortado (P2-45): el de mayor versión semántica `vX.Y.Z`, o el indicado con `dataset_version` (el catálogo no tiene orden propio; si ninguna versión es semver no se adivina y se pide indicarla). `data.dataset_version` dice de cuál es. |
| `get_versions_report` | El catálogo `versions.json` de releases cortados. |

Los splits no tienen una copia "vigente": solo existen dentro de un release
(`reports/releases/<version>/splits.json`), por eso `get_splits_report` se
resuelve a través del catálogo. Todas devuelven `available: false` en vez de
inventar datos o fallar cuando el reporte, el catálogo o la versión pedida no
existen — mismo principio que ya usa el frontend
(`frontend/src/pipeline/dataSource.ts`).

**Solo lectura, verificado en dos niveles:**
1. Cada herramienta declara `ToolAnnotations(read_only_hint=True,
   destructive_hint=False, ...)` — metadato del protocolo MCP mismo, no
   solo un comentario.
2. Ningún handler importa `storage.db` ni `storage.object_store` (el único
   tier autorizado a tocar MariaDB/MinIO); `tests/test_mcp_server.py`
   verifica ambas cosas, más un grep del código fuente contra los verbos
   SQL de escritura y `put_object`.

Desde `app/`:

```bash
uv run pytest tests/test_mcp_server.py
DATABASE_URL=... MINIO_ENDPOINT=... MINIO_PORT=... MINIO_ACCESS_KEY=... \
MINIO_SECRET_KEY=... MINIO_BUCKET=... DATASET_DIR=../data/raw REPORTS_DIR=../.reports \
uv run python -m presentation.mcp_server   # stdio, para un cliente MCP local
```

El agente que consume este servidor es el Copilot de P2-52 (`app/copilot/`,
ver su README): lo usa en proceso desde su propio servicio de `docker-compose.yml`.
Este servidor sigue pudiendo invocarse a mano vía stdio, como arriba.

### P2-53: evaluación compartida y leakage

`evaluate_dataset()` carga COCO y bytes una vez, ejecuta pHash una vez y construye
un `SplitResult` con la misma política de similitud y una única `SplitsConfig`.
Devuelve `(QualityReport, SplitResult)`. `build_quality_report()` conserva la API
de solo reporte; `cut_release()` consume ambos resultados de esa evaluación,
sin volver a detectar duplicados ni generar otra asignación. Un release histórico
puede conservar calidad failed, pero `cut_release()` rechaza nuevos releases con
calidad failed.

Leakage cuenta aristas pHash directas, no todas las combinaciones de una componente
transitiva: A-B/B-A cuenta una vez. `details.image_pairs` contiene solo pares
cruzados con IDs, splits, distancia y similitud; `total_pairs_evaluated` incluye
también pares internos. Incluye `images_per_split`, `splits_config`, threshold
pHash y `criterion: metric_value <= threshold`. La prevención SHA/filename/pHash
del splitter se conserva; la nueva métrica audita las asignaciones resultantes.
No certifica asignaciones externas ni históricas que no se persistieron.

Asignaciones incompletas, duplicadas o con referencias desconocidas son errores;
una contaminación válida es un check incumplido. Menos de tres componentes
independientes impide generar tres splits no vacíos y aborta explícitamente la
evaluación (por ejemplo, threshold pHash=0 une todas las imágenes). No se publica
un cero inventado. Settings mantiene sus seis controles; leakage no es editable.

DVC separa `quality_report` de `quality_gate` e incorpora código y parámetros de
splits (train/val/test/seed). El stage `split` depende del marcador que solo se
escribe cuando la compuerta pasa; una evaluación failed hace fallar DVC y no
ejecuta el downstream. No se persisten assignments en este stage.
