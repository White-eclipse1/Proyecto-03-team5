# Evidencia ML-01: crops COCO dog/cat sobre el release v0.1.1

Ejecución: 2026-09-29. Base de código: `03ee491` (rama `feat/ml-01-coco-crop-extraction`).
Release de origen: P2 `v0.1.1` (quality gate `warning`, no `failed`), `data/raw/images`
`951150dd4fb053f4665089fcb37a1c87.dir` (600 imágenes) y `data/raw/annotations`
`c7cb86ae7ece94ef7b853620e464a4d7.dir`.

## Correspondencia con la evaluación

- Rúbrica 1.2: cada crop conserva `image_id`, `annotation_id`, `category_id`, clase y bbox;
  rechaza cajas degeneradas, fuera de imagen e imágenes faltantes; clases con ≥ 300
  imágenes originales; verificación manual de crops y etiquetas contra el COCO original.
- Issue #3 (ML-01): Agent Test (bbox degenerada sin crop ni entrada en el manifiesto) y
  DoD "Manual crop verification completed".
- Rúbrica 7.1: mutaciones de reglas críticas detectadas por la suite.

## Reproducir desde la raíz

Requisitos: `dvc pull -r prod data/raw/images.dvc data/raw/annotations.dvc` y
`dvc repro crops` (o `dvc pull -r prod crops`), más `app/.venv` de `app/uv.lock`.

```bash
cd app
RUN_CROPS_EVIDENCE=1 .venv/bin/python -m pytest -q -s tests/test_crops_evidence.py
```

Para regenerar la hoja visual en esta carpeta:

```bash
RUN_CROPS_EVIDENCE=1 CROPS_EVIDENCE_SHEET="$PWD/tests/evidence/ml-01-crops-verification.jpg" \
  .venv/bin/python -m pytest -q tests/test_crops_evidence.py -k sheet
```

Sin `RUN_CROPS_EVIDENCE=1` los cuatro tests se omiten (el CI no descarga el dataset en el
job de pytest). El job `Quality gate (PROD dataset …)` del CI sí corre `dvc repro crops`
sobre PROD: `Crops aceptados=668 rechazados=0`.

## 1. Resultado sobre el release real

```text
{"total_annotations":668,"accepted":668,"rejected":0,
 "accepted_per_class":{"dog":325,"cat":343},
 "accepted_images_per_class":{"dog":300,"cat":301},"rejected_per_reason":{}}
```

Las categorías del COCO de v0.1.1 son solo `dog` (id 3, 325 anotaciones) y `cat` (id 4,
343 anotaciones): no se excluyó ninguna clase. Ambas cumplen `MIN_IMAGES_PER_CLASS = 300`
imágenes originales distintas; `dog` queda exactamente en el mínimo.

## 2. Crops y etiquetas contra el COCO original

`test_sampled_crops_match_the_original_coco` toma 40 crops de `reports/crops.json`
(todas las imágenes con dog y cat, más una muestra aleatoria con semilla 42) y, para cada
uno, compara contra la anotación COCO original: `image_id`, `category_id`, nombre de la
clase, bbox exacta, `file_name` y los píxeles del PNG guardado contra el recorte de la
imagen original en `crop_box`.

```text
Crops verificados contra COCO e imagen original: 40
Imágenes con dog y cat (un crop por objeto): [40]
```

## 3. Verificación manual (visual)

[`ml-01-crops-verification.jpg`](ml-01-crops-verification.jpg): 12 filas de la misma
muestra, con la imagen original y la bbox COCO en rojo a la izquierda, y el crop guardado
a la derecha. Revisión a ojo el 2026-09-29: los 12 crops corresponden a su bbox y la
etiqueta coincide con el animal. La imagen 40 (un perro y un gato) produce dos crops
independientes, `img40-ann65 · dog` e `img40-ann66 · cat`, no una etiqueta para la imagen
completa.

## 4. Agent Test: cajas inválidas inyectadas en el COCO real

`test_injected_invalid_boxes_on_the_real_release_are_excluded` agrega cuatro anotaciones
al COCO real (en memoria; la salida va a un directorio temporal) y extrae de nuevo:

```text
ann 669: sin crop ni manifiesto; motivos=['non_positive_width']
ann 670: sin crop ni manifiesto; motivos=['non_positive_height']
ann 671: sin crop ni manifiesto; motivos=['exceeds_image_width', 'out_of_bounds']
ann 672: sin crop ni manifiesto; motivos=['missing_image']
```

Ninguna produce PNG ni aparece en `crops`; las cuatro quedan en `rejections` con su
motivo, y los 668 crops válidos siguen aceptados.

## 5. Mutaciones de reglas críticas

En una copia aislada de `app/` (no en el repositorio), se rompió una regla a la vez y se
corrió `pytest tests/test_crops.py tests/test_crops_release_integration.py` (44 tests):

| Mutación | Resultado |
|----------|-----------|
| `width <= 0` → `width < 0` (acepta ancho cero) | 2 failed |
| `height <= 0` → `height < 0` | 1 failed |
| No revisar `x + width > W` | 1 failed |
| Imagen faltante devuelve píxeles en blanco | 2 failed |
| `ceil` → `floor` en el extremo del crop (recorta la bbox) | 1 failed |
| `annotation_id` guarda el `image_id` | 7 failed |
| Invertir el orden de `CLASS_NAMES` | 3 failed |
| `MIN_IMAGES_PER_CLASS = 1` | 2 failed |
| El stage no llama `assert_min_images_per_class` | 1 failed |

Las nueve mutaciones hacen fallar la suite. La copia se eliminó después.
