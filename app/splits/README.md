# P2-32 — Splits multietiqueta reproducibles

`stratified.split_dataset()` es puro: recibe COCO, `SplitsConfig`, contenido
codificado de las imágenes en memoria y las relaciones de duplicados ya
calculadas. No lee YAML, archivos, entorno, MinIO ni red; tampoco escribe o
modifica datos. `presentation.splits.build_splits_report()` adapta el resultado
al `SplitsReport` v1.0 existente, sin escribir `splits.json`.

## Uso y límites de integración

El llamador carga una sola configuración con `load_splits_config()` y una política
con `load_duplicate_config()`. Una vez cargados COCO e imágenes, puede invocar:

```python
from analyzers.duplicates import analyze_duplicates
from policies.duplicates import load_duplicate_config
from presentation.splits import build_splits_report
from splits.models import load_splits_config
from splits.stratified import split_dataset

# coco: documento validable como CocoDataset.
# image_contents: dict[image_id, bytes], completo para las mismas imágenes COCO.
duplicates = analyze_duplicates(image_contents, load_duplicate_config())
result = split_dataset(
    coco,
    load_splits_config(),
    image_contents=image_contents,
    duplicate_pairs=duplicates.details["image_pairs"],
)
report = build_splits_report(result, dataset_version="demo-v1")
json_text = report.model_dump_json(indent=2)  # Solo memoria; no escribe archivos.
```

No se conecta automáticamente a `presentation.main` ni al gate: así no se mezcla
P2-30, no se vuelve a ejecutar pHash si el llamador ya dispone del resultado y no
se introducen accesos a servicios. Un futuro orquestador puede publicar `report`
en `REPORTS_DIR`; las asignaciones permanecen en memoria. No se define un formato
persistente de entrenamiento, ni se implementa DVC. No escribir derivados dentro
de `data/raw/`. Los directorios locales de reportes no están globalmente ignorados:
antes de persistir artefactos hay que acordar su ubicación/ignore o seguimiento DVC.

## Unidad y leakage

Una imagen completa es indivisible, incluyendo todas sus anotaciones. Cada
imagen aporta una presencia por categoría, aunque tenga varias cajas de esa
categoría; puede aportar a varias categorías. Las categorías declaradas vacías
se conservan con conteos cero, y las imágenes sin anotaciones sí se asignan.

Se construye una partición en componentes conexas usando:

- Igualdad SHA-256 de los bytes suministrados (copias exactas con IDs/nombres distintos).
- Igualdad de `file_name` (alias de la misma referencia de archivo): solo se
  acepta si coincide el digest del contenido. Mismo nombre con bytes diferentes
  es una entrada contradictoria y produce un error explícito.
- Todos los pares suministrados por el detector pHash existente. Sus campos
  `image_id_a` e `image_id_b` se consumen directamente; el splitter no recalcula
  pHash ni interpreta su umbral o score.

A–B y B–C agrupan A, B y C, incluso sin un par A–C. Estos grupos nunca se rompen.
`image_contents` debe cubrir exactamente los IDs y contener bytes no vacíos.
No se decodifican imágenes aquí: el llamador/detector valida su contenido visual.
La garantía visual depende de recibir todos los pares detectados sobre esas
mismas imágenes y con la política acordada (actualmente similitud mínima 0.94).
Pasar una lista vacía significa declarar que no hay pares conocidos, no certificar
que las imágenes no son visualmente similares. pHash puede tener falsos positivos
o negativos; tampoco se infieren parentescos de pacientes, escenas o sesiones.

`verify_assignment(assignments, image_ids=..., duplicate_groups=...)` verifica
los tres nombres internos, no vacíos, cobertura exacta, ausencia de IDs repetidos
y permanencia de cada grupo en un único split. Requiere la lista completa de IDs
y grupos de origen: no deben reconstruirse a partir de una asignación sospechosa.
El splitter lo ejecuta antes de retornar. El adaptador también comprueba las
invariantes del resultado recibido antes de construir el resumen.

## Configuración y reproducibilidad

Se reutiliza `splits.yaml`: `train=0.70`, `val=0.15`, `test=0.15`, `seed=42`.
`SplitsConfig` admite ratios estrictamente entre cero y uno cuya suma difiera de
uno como máximo en `1e-6`. Se normaliza esa pequeña tolerancia antes de calcular
objetivos. No se añade otra configuración. La semilla es un entero estricto;
se admiten negativos, que `random.Random` maneja de forma determinista.

IDs, categorías y componentes se ordenan canónicamente. Un `Random(seed)` local
solo resuelve empates, incluidos restos de redondeo iguales; no cambia el RNG
global. Este RNG se usa exclusivamente para desempates reproducibles del dataset,
no para seguridad, secretos, tokens ni criptografía. La misma entrada, relaciones, bytes, configuración e implementación
producen exactamente el mismo resultado, incluso si se reordenan las listas COCO,
el mapping de bytes o los pares. No se promete idéntica asignación si cambia
el dataset, los pares detectados, la política de pHash o el algoritmo. Una semilla
diferente puede cambiar empates, pero no tiene obligación de cambiar el resultado.

## Asignación y medida de estratificación

Para cada clase `c`, `N_c` cuenta imágenes distintas y el objetivo en el split `s`
es `r_s * N_c`. El tamaño objetivo global se redondea por mayores restos para que
los tres tamaños enteros sumen exactamente `N`.

1. Se prioriza el grupo con la clase con menos presencias pendientes; dentro de
   ese criterio se prioriza el grupo más grande. Las imágenes sin etiquetas van
   después de los grupos etiquetados.
2. Se elige el destino que menos incrementa el error cuadrático normalizado:
   `sum_s ((n_s - target_size_s)^2 / N + sum_c (n_sc - r_s*N_c)^2 / N_c)`.
   Las categorías vacías no aportan divisiones por cero.
3. Se reservan grupos para los splits aún vacíos cuando es necesario.
4. Se mejoran desviaciones evitables moviendo grupos completos cuando el error
   disminuye estrictamente (más de `1e-12`), sin vaciar el origen. Esto termina
   en un mínimo local de movimientos individuales, no necesariamente el óptimo global.

Prioridad: cero leakage, cobertura/exclusividad, reproducibilidad y luego
aproximación de proporciones. Los objetivos y conteos reales quedan disponibles
para auditar las desviaciones. También puede calcularse la prevalencia
`class_counts[s][c] / len(assignments[s])` y compararla con `N_c / N`.

En escenarios simples factibles, los tests comprueban el reparto exacto o una
desviación de hasta una imagen por clase debido al redondeo. Esa cota no es una
garantía universal: multietiquetas, clases escasas y grupos grandes pueden impedirla.

## Resultado en memoria y contrato público

`SplitResult` realiza copias defensivas al construirse: los mappings exteriores
y anidados son de solo lectura (`MappingProxyType`), y los IDs/grupos se
convierten en tuplas. Ni mutar el resultado ni modificar los diccionarios/listas
originales de construcción puede alterar esas estructuras. No es un formato
de persistencia; el resumen público se serializa mediante `SplitsReport`.

`SplitResult` conserva:

- `assignments`: `train/val/test -> tuple[image_id, ...]`, ordenadas.
- `groups`: tuplas de IDs de los componentes, incluyendo imágenes aisladas.
- `target_sizes`: objetivos enteros por split.
- `class_counts`: conteos reales por split y categoría, incluidas categorías vacías.
- `target_class_counts`: objetivos fraccionarios por split y categoría.

Para 20 imágenes independientes, 10 de cada una de dos clases, y ratios
0.6/0.2/0.2, los tests obtienen 12/4/4 imágenes y 6/2/2 imágenes por clase.

El resumen conserva `schema_version="1.0"`, `dataset_version`, `total_images` y
`splits.train/validation/test`, cada uno con `image_count` y **ratio realizado**.
No publica IDs ni altera el contrato. El `val` interno se convierte en
`validation` solamente en presentación.

## Casos imposibles y validación

COCO se valida con los modelos estrictos existentes: IDs duplicados, referencias
huérfanas y estructuras inválidas producen errores, sin reparación ni renumeración.
Pares desconocidos o autorreferencias también se rechazan. Con menos de tres
componentes se devuelve un error explícito: no existen tres splits no vacíos.
Con tres grupos y ratios 0.98/0.01/0.01 se obtienen tres splits no vacíos aunque
los tamaños redondeados sean 3/0/0. Una clase de una sola imagen no puede aparecer
en los tres splits. Un grupo de ocho imágenes entre diez puede forzar 8/1/1:
se conserva el grupo y se expone la desviación, sin fingir un reparto exacto.

Desde `app/`, con el entorno existente:

```bash
.venv/bin/python -m pytest tests/test_stratified_splits.py tests/test_splits_models.py tests/test_json_contracts.py
.venv/bin/ruff check splits/stratified.py presentation/splits.py tests/test_stratified_splits.py
.venv/bin/ruff format --check splits/stratified.py presentation/splits.py tests/test_stratified_splits.py
```

Las pruebas son sintéticas, deterministas y no dependen del dataset local ni
persisten derivados. Incluyen una recomprensión JPEG en memoria y el detector
pHash existente, además de asignaciones deliberadamente contaminadas.
