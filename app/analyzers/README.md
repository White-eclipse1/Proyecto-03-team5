# P2-13 — Objetos pequeños

`analyze_small_objects(coco, config)` recibe un COCO ya cargado y devuelve el
`AnalyzerResult` común. No hace I/O ni modifica el documento de entrada.
La configuración se obtiene por separado:

```python
from analyzers.small_objects import analyze_small_objects
from policies.small_objects import load_small_object_config

config = load_small_object_config()  # Lee policies/quality.yaml; requiere PyYAML.
result = analyze_small_objects(coco, config)  # coco ya cargado por el llamador.
```

En `max_small_object_ratio`, `width_px` y `height_px` valen 32 por defecto en el
YAML. No hay valores de respaldo hardcodeados en Python. `threshold: 0.40`
sigue siendo la proporción máxima permitida, no un tamaño. `action: warn`
permanece intacto; este analizador no decide bloquear un release.

Para resolver la ambigüedad de “32×32”, una caja es pequeña solo si **ambas**
dimensiones son estrictamente menores que sus límites. Una caja 20×50 no es
pequeña con límites 32×32; tampoco una 32×32. Se usan ancho y alto de `bbox`
COCO `[x, y, width, height]`, no `area` (que puede describir una segmentación).
No se normalizan los tamaños por la resolución de imagen.

Se cuentan todas las anotaciones, incluidas las marcadas `iscrowd`. El COCO
debe tener `annotations` y `categories` válidos. Se rechazan cajas degeneradas
o no finitas y categorías desconocidas; no se reparan datos ni se implementa
el analizador de cajas inválidas. No se comprueban los límites de imagen.

Salida:

- `check_name`: `max_small_object_ratio`.
- `passed`: la proporción es menor o igual al `threshold` suministrado.
- `metric_value`: proporción entre 0 y 1 (multiplicar por 100 para porcentaje).
- `details.small_objects`, `details.total_objects`: conteos de cajas.
- `details.most_affected_class`: `category_id`, `category_name`, `small_objects`,
  `total_objects` y `ratio` dentro de esa clase; `null` si no hay ofensoras.
  Se elige por mayor **conteo** de cajas pequeñas, con desempate por menor ID.
- `details.offending_samples`: lista de `annotation_id`, `image_id`,
  `category_id`, `bbox`, en orden de entrada. Cada muestra es una anotación,
  no una imagen única; permite localizar varias ofensoras en la misma imagen.

Con cero anotaciones: proporción 0, `passed=true`, conteos 0, clase `null`
y lista vacía. Esto no certifica que un dataset vacío tenga calidad suficiente;
solo indica que este check no encontró objetos pequeños.

La salida conserva `AnalyzerResult` sin cambios y no agrega campos de la
compuerta ni integra contratos de presentación ausentes en esta rama.

Desde `app/`, con las dependencias declaradas disponibles:

```bash
python -m pytest tests/test_small_objects.py
python -m pytest
ruff check .
ruff format --check .
```

## P2-19 — Desbalance de clases

`analyze_imbalance(coco, config)` en `analyzers/imbalance.py` recibe el COCO
ya cargado y una `ImbalanceConfig`, sin I/O ni mutación de la entrada.
`policies.imbalance.load_imbalance_config()` reutiliza `load_quality_policy()`:
extrae `min_images_per_class.threshold` y `max_imbalance_ratio.threshold` del
YAML existente, sin duplicar el parser ni tener defaults numéricos en el analizador.

```python
from analyzers.imbalance import analyze_imbalance
from policies.imbalance import load_imbalance_config

result = analyze_imbalance(coco, load_imbalance_config())
```

Se consideran todas las categorías declaradas. Cada `image_id` cuenta una sola
vez por categoría, aunque tenga varias cajas; una imagen multiclase cuenta una
vez en cada clase. Se incluyen anotaciones `iscrowd`. Imágenes sin anotaciones
no se asignan artificialmente a una clase. Se rechazan referencias desconocidas,
IDs de imagen/categoría duplicados y ausencia de categorías con `ValueError`.
El resto de la validación estructural sigue correspondiendo a ingesta.

La salida conserva `AnalyzerResult`:

- `check_name`: `max_imbalance_ratio`.
- `metric_value`: máximo/mínimo de imágenes distintas por categoría, si el
  denominador es positivo.
- `passed`: ratio definido y menor o igual a `max_imbalance_ratio.threshold`.
  No combina la acción `warn` de esta regla con el `fail` del mínimo por clase;
  las clases bajo mínimo se reportan, sin implementar la compuerta.
- `details.images_per_category`: lista ordenada por ID, con `category_id`,
  `category_name`, `image_count` e `image_ids` distintos y ordenados.
- `details.majority_class` y `minority_class`: entradas con esos mismos campos;
  en empate se elige el menor ID (ambas pueden señalar la misma clase).
- `details.classes_below_minimum`: entradas con conteo estrictamente menor al
  mínimo. Igualar el mínimo es suficiente.
- `details.min_images_per_class` y `max_imbalance_ratio`: parámetros usados.
- `details.empty_category_ids`: categorías sin imágenes.
- `details.ratio_defined`: indica si `metric_value` es un ratio calculado.

**Categorías vacías:** dividir entre cero no produce un ratio finito. Para no
cambiar el contrato ni emitir infinito/NaN, `metric_value=0.0` es exclusivamente
un marcador cuando `ratio_defined=false`; `passed=false` en ese caso. No debe
mostrarse como “ratio cero” ni interpretarse como balance: el consumidor debe
mostrar “no calculable: categoría sin imágenes”. Las categorías vacías permanecen
en los conteos y en `empty_category_ids`, incluso si el mínimo configurado es cero.
Esto también aplica cuando todas las categorías están vacías.

Verificación manual del fixture: cat aparece en imágenes 1, 2, 3 y 4; dog en
1 y 2. Varias cajas de cat en imagen 1 no cambian el conteo: ratio 4/2 = 2.
Con mínimo 3, solo dog está bajo mínimo; con límite de ratio 2, el check pasa.

Desde `app/`: `python -m pytest tests/test_imbalance.py`.

## P2-20 — Duplicados perceptuales

`analyze_duplicates(images, config)` en `analyzers/duplicates.py` recibe un
mapping de IDs enteros no negativos a bytes codificados de imágenes. El llamador
resuelve COCO/archivos y carga los bytes: el analizador no conoce nombres,
rutas, MinIO ni splits. No modifica las entradas. Los IDs deben ser únicos;
un mapping no puede representar dos imágenes diferentes con el mismo ID.

`policies.duplicates.load_duplicate_config()` reutiliza `load_quality_policy()`
y extrae `duplicate_similarity_threshold.threshold`. Se valida entre 0 y 1,
sin valor predeterminado en Python. El YAML mantiene 0.94 como **similitud mínima**,
no distancia ni proporción de duplicados permitida.

Se usa `imagehash.phash`, con hash_size=8 (64 bits), una vez por imagen. La
distancia es Hamming y `similarity = 1 - hamming_distance / hash_bits`.
Se incluye un par cuando `similarity >= threshold`. Cada par aparece una sola
vez, con ID menor primero y orden lexicográfico por IDs, independientemente
del orden de entrada. La comparación es O(n²); no se incorpora un índice externo.

Salida `AnalyzerResult`:

- `check_name="duplicate_images"`: detección global, distinta de leakage entre splits.
- `metric_value`: cantidad de pares encontrados, no cantidad de imágenes únicas.
- `passed`: `true` si no se detectaron pares al umbral indicado; `false` si hay
  candidatos a duplicados. Es un resumen de detección, no decide bloquear un
  release ni introduce un umbral de calidad adicional en el YAML.
- `details.image_pairs`: lista de objetos con `image_id_a`, `image_id_b`,
  `hamming_distance` (entero) y `similarity` (número entre 0 y 1).
- `details.hash_bits`: 64.
- `details.similarity_threshold`: umbral usado.
- `details.total_images`: cantidad de imágenes examinadas.

`duplicate_similarity_threshold` sigue siendo un parámetro de detección;
su `action` no se aplica automáticamente como regla de bloqueo. La futura
compuerta deberá decidir cómo consumir `duplicate_images`. No se implementa
`cross_split_leakage`. Cero o una imagen producen cero pares; no certifican
la calidad global del dataset. Bytes corruptos/entradas inválidas producen
`ValueError` identificando el problema en vez de omitir imágenes silenciosamente.

pHash usa estructura de luminancia: los pares son candidatos, no prueba de
igualdad de archivos. No garantiza detectar cualquier recorte, giro o cambio
de contenido. No se normaliza orientación EXIF fuera de lo que hace la librería.

Desde `app/`: `python -m pytest tests/test_duplicates.py`. Las pruebas cubren
copias, JPEG recomprimido con otro nombre, reescalado, controles negativos,
distancia, similitud, umbral inclusivo, orden y no mutación. Una prueba adicional
usa `data/raw/images/cat.0.jpg` y genera JPEG de calidad 65 **solo en memoria**;
se omite explícitamente si ese archivo DVC no está materializado. Ninguna prueba
escribe variantes en el dataset; los archivos sintéticos usan `tmp_path`.

## P2-21 — Cajas inválidas

`analyze_invalid_boxes(coco, config)` en `analyzers/invalid_boxes.py` inspecciona
el documento crudo **antes** de `CocoDataset.model_validate()` y de objetos
pequeños. Es una auditoría previa: no reemplaza ni relaja la validación estricta
de ingesta. El llamador debe conservar este diagnóstico y resolver los datos
inválidos antes de enviarlos a analizadores que los rechazan; un `passed=true`
por un threshold permisivo no garantiza que ingesta vaya a aceptar el documento.
No se agrega orquestación ni se modifica `small_objects`.

`policies.invalid_boxes.load_invalid_box_config()` reutiliza `load_quality_policy()`
y obtiene `degenerate_boxes.threshold`, sin default en Python. El analizador no
lee archivos, imágenes físicas ni YAML, y no modifica la entrada.

Se usan `width` y `height` de `coco.images`. Contrastar esas dimensiones con
los píxeles reales corresponde al llamador. Se rechazan dimensiones de imagen
no positivas/no finitas e IDs de imagen duplicados mediante `ValueError`.

Salida:

- `check_name="degenerate_boxes"`.
- `metric_value`: cantidad de anotaciones inválidas, cada una contada una vez.
- `passed`: cantidad inválida menor o igual al threshold configurado.
- `details`: `total_annotations`, `invalid_annotations`, `offending_samples`.
- Cada muestra incluye `annotation_id`, `image_id`, copia de `bbox`, `reasons`,
  `reported_area`, `calculated_area`, `image_width` e `image_height`.

Códigos de `reasons`, en orden de evaluación:

1. `non_positive_width`: width <= 0.
2. `non_positive_height`: height <= 0.
3. `negative_x`: x < 0.
4. `negative_y`: y < 0.
5. `unknown_image_id`: referencia ausente; dimensiones del reporte quedan null,
   pero se siguen evaluando tamaño, origen y área.
6. `exceeds_image_width`: x + width > ancho de imagen.
7. `exceeds_image_height`: y + height > alto de imagen.
8. `non_finite_calculated_area`: overflow del producto; área calculada null.
9. `area_mismatch`: área reportada no coincide con width * height.

No hay cortocircuito entre comprobaciones independientes: se reportan todos
los motivos evaluables. Las muestras conservan el orden de anotaciones de
entrada; repetir el análisis de la misma entrada produce el mismo resultado.
Terminar exactamente en un borde es válido; no se usa tolerancia geométrica.

Para el área se usa `math.isclose` con tolerancia relativa `1e-9` y absoluta
`1e-6` píxeles cuadrados: admite residuos normales de punto flotante y compara
relativamente áreas grandes. La diferencia admisible es el máximo entre ambas
tolerancias (la relativa multiplicada por la mayor magnitud de las áreas).
No se recalcula desde segmentaciones ni se corrige el área reportada.

Se esperan IDs/campos COCO presentes, bbox con cuatro números finitos y área
numérica finita. Bbox malformadas o área no numérica/no finita se rechazan con
`ValueError`; no se convierten silenciosamente en cajas válidas. Los tamaños
degenerados sí se reportan, incluyendo su producto matemático aunque sea negativo.

Desde `app/`: `python -m pytest tests/test_invalid_boxes.py`. El caso obligatorio
inyecta en un mismo documento bbox `[0, 0, -10, 10]` y `[95, 0, 10, 10]` para
una imagen de 100×80: ambas se reportan, y las validaciones estrictas existentes
siguen rechazando la primera.

## P2-29 — Sesgo espacial

`analyze_spatial_bias(coco, config)` en `analyzers/spatial_bias.py` recibe el
COCO ya cargado y una `SpatialBiasConfig`, sin I/O ni mutación de la entrada.
El ticket original no definía el algoritmo (a diferencia de sus hermanos
P2-19/20/21); se confirmó con Andy la definición: centro normalizado de cada
bbox (0-1 en cada eje, relativo a su imagen) y desviación estándar poblacional
por eje. Baja dispersión ⇒ los anotadores centran sistemáticamente el objeto.

```python
from analyzers.spatial_bias import analyze_spatial_bias
from policies.spatial_bias import load_spatial_bias_config

result = analyze_spatial_bias(coco, load_spatial_bias_config())
```

`policies.spatial_bias.load_spatial_bias_config()` reutiliza
`load_quality_policy()` y extrae `min_spatial_dispersion.threshold` (clave
nueva en `quality.yaml`; no existía ninguna que mapeara a este check, a
diferencia de los otros 4 analizadores). Una distribución uniforme en [0, 1]
tiene desviación ≈0.289; el YAML fija el umbral en 0.15, bien por debajo de
eso, para no exigir uniformidad perfecta.

Salida:

- `check_name`: `spatial_bias`.
- `metric_value`: la **menor** de las desviaciones estándar de los dos ejes
  (`std_center_x`, `std_center_y`) — el eje con menos dispersión es el que
  marca el sesgo; no se promedian ambos ejes.
- `passed`: `metric_value >= min_spatial_dispersion.threshold`. A diferencia
  del resto de los checks (que exigen un máximo), este exige un **mínimo**
  de dispersión — mismo patrón de nombre que `min_images_per_class`.
- `details.total_boxes`, `mean_center_x`, `mean_center_y`, `std_center_x`,
  `std_center_y`.

Con 0 o 1 caja la dispersión es 0.0 por definición (no hay variación posible),
lo que falla cualquier umbral positivo — no es un caso de error, es la
respuesta correcta ("no hay suficiente evidencia de dispersión"). No se usa
`area` ni tamaño de caja, solo la posición del centro; no se corrige por
orientación de la imagen ni se pondera por categoría.

Corrido contra el dataset real (`data/raw/`): `metric_value≈0.088`,
`mean_center_x≈0.50`, `mean_center_y≈0.51` — el dataset (fotos de stock
recortadas) sí centra sistemáticamente al animal; el check falla
correctamente con `action: warn` (no bloquea, pero queda registrado).

Desde `app/`: `python -m pytest tests/test_spatial_bias.py`.

## P2-53 — cross_split_leakage

`analyze_cross_split_leakage(assignments, duplicate_pairs, config, image_ids=...)`
es puro: recibe assignments y pares del detector existente, sin recalcular hashes
ni generar splits. Cuenta pares únicos entre splits distintos; el universo COCO
permite rechazar omisiones/IDs extra incluso cuando no aparecen en ningún par.
`LeakageConfig` recibe el límite de pares y el threshold de detección de la misma
QualityPolicy. `passed` usa `metric_value <= threshold`. Se conserva pHash de
64 bits y se rechazan metadatos contradictorios de pares. Los resultados son
canónicos por ID y contienen splits/IDs/distancia/similitud de los pares cruzados.
El sesgo espacial P2-29 conserva su algoritmo; P2-53 amplía sus pruebas de integración.
