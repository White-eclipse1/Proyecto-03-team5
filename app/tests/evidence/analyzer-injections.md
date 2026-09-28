# Evidencia de inyecciones: analizadores y quality gate

Ejecución: 2026-09-18. Base de código: `746818a` (incluye `e01a5b9`).
Solo se agregan tests y esta evidencia; no se cambian analizadores ni configuración normal.

## Correspondencia con la evaluación

- Rúbrica 3.4: ancho negativo y caja fuera del límite de imagen.
- Rúbrica 3.3: copia JPEG recomprimida de una imagen real y detección pHash.
- Rúbrica 4.1–4.2: mínimo imposible, reporte failed, exit distinto de cero y bloqueo
  de downstream. Esta prueba figura en las acciones del profesor, aunque corresponde al gate.

Las funcionalidades ya existían en `analyzers/invalid_boxes.py`, `analyzers/duplicates.py`,
`presentation/gate.py` y `dvc_gate_stage.py`. Había tests unitarios de geometría,
recompresión y gate, pero faltaba esta ejecución conjunta documentada sobre datos reales
con bloqueo DVC verificado. No existe `params.yaml`: las políticas están en
`policies/quality.yaml`, y DVC referencia parámetros de `splits/splits.yaml` y
`projections/projections.yaml`. El reporte versionado quality.json es v1.0, warning,
con siete checks; no se utiliza como sustituto del recálculo.

## Reproducir desde la raíz

Requisitos: dataset materializado siguiendo README, `app/.venv` con dependencias de
`app/uv.lock`, y `.venv-dvc` con DVC del README. No se instalan dependencias ni se
contactan remotes dentro de estas pruebas. Se usa Python 3.12.

```bash
cd app
RUN_EVALUATION_INJECTIONS=1 .venv/bin/python -m pytest -q -s tests/test_evaluation_injections.py
```

Para guardar una nueva salida fuera del repositorio, conservando el exit code (bash/zsh):

```bash
set -o pipefail
RUN_EVALUATION_INJECTIONS=1 .venv/bin/python -m pytest -q -s tests/test_evaluation_injections.py 2>&1 | tee /tmp/analyzer-injections.txt
```

Se pueden ejecutar por separado añadiendo `-k invalid_boxes`, `-k recompressed_copy`
o `-k impossible_threshold`. Si DVC está en otra ruta, suministrar
`EVALUATION_DVC=/ruta/absoluta/al/dvc`. Sin `RUN_EVALUATION_INJECTIONS=1` los tres tests
se omiten deliberadamente: una omisión NO acredita estas pruebas. Con la variable,
la falta de datos o DVC produce un fallo explícito, no un falso PASS.

## Alcance y protección del dataset

Las cajas cambian únicamente en una copia en memoria del COCO crudo, antes de la
validación estricta de ingestion. La imagen recomprimida se escribe en tmp_path,
con ID nuevo y nombre `recompressed.jpg`; no se registra ni reemplaza en data/raw.
La prueba compara distancia Hamming y similitud mediante una llamada independiente
a ImageHash; no implementa pHash manualmente.

El dataset LOCAL utilizado contiene 600 imágenes, 668 anotaciones y categorías
person/car/dog/cat. La política original ya da failed porque person y car no tienen
anotaciones. Esto NO se corrige ni se oculta: `source_baseline` lo registra.
Para aislar el efecto del umbral, se construye una copia temporal con las mismas
600 imágenes y 668 anotaciones, declarando solamente las categorías anotadas dog/cat.
Esa selección define el fixture controlado; no es una reparación del dataset real.
Dog tiene 300 imágenes distintas y cat 301, calculadas con sets de image_id.

Sobre ese fixture, la política normal (mínimo 300/fail) produce warning. Se cambia
solo la copia temporal del mínimo a N+1=601; la severidad permanece fail. Produce
failed, metric_value=300 y las dos clases bajo el mínimo. Los umbrales normales
nunca se escriben ni necesitan restauración.

Se copian las definiciones reales quality_gate y split de dvc.yaml con sus deps/outs.
Se ejecuta ese subgrafo en un DVC `init --no-scm` temporal, sin remotes. Única adaptación
del comando: `uv run python` se sustituye por el intérprete de pytest ya instalado;
los scripts ejecutados son copias sin cambios de los scripts reales.
Esto verifica esas dos etapas, no toda la DAG ni una promoción a AWS.

Primero `dvc repro split` termina con 0 y genera splits.json (control positivo).
Se retira únicamente ese output temporal para que su presencia anterior no esconda
una ejecución indebida. Tras escribir el reporte failed, el gate CLI retorna 1 y
`dvc repro split` retorna 255: no aparece ejecución de split ni su reporte ni marcador.
El reporte fallido se serializa y valida nuevamente con QualityReport v1.0.

El fixture verifica SHA-256 antes/después de todos los archivos originales de data/raw
y quality.yaml. No se escriben reportes de runtime, cachés ni copias del dataset en Git.
pytest puede conservar tmp_path fuera del repositorio conforme a su retención habitual.

## Resultados

| Prueba | Antes | Inyección | Obtenido |
|---|---|---|---|
| Cajas | 0 inválidas / 668 | Anotación 1: width=-1; anotación 2: x=image_width | 2 inválidas; non_positive_width + area_mismatch, y exceeds_image_width |
| Recompresión | 0 pares (una imagen) | cat.0.jpg, ID 3, JPEG RGB calidad 65, ID temporal 665 | 1 par; Hamming=0; similitud=1; umbral=0.94; bytes distintos |
| Umbral | Fixture warning, mínimo 300 | Mínimo temporal 601/fail | failed; gate=1; DVC=255; split no ejecutado |

Resultado final: **3 passed in 14.07s**.
Tests existentes: **77 passed in 1.51s** (invalid_boxes, duplicates, gate,
dvc_pipeline y evaluation_gaps). Ruff lint/formato del archivo añadido pasan.
La primera ejecución señaló correctamente el baseline local ya fallido. Durante la
construcción del harness se corrigió además una expectativa del nombre del script
(`dvc_gate_stage.py`, no `dvc_quality_gate_stage.py`); no se alteró código productivo.

## Salida capturada

Se sustituye únicamente el prefijo de ruta local por `<REPO>`; no se copian variables
de entorno ni credenciales. Las sugerencias impresas por DVC no fueron ejecutadas.

```text
{"after": {"check_name": "degenerate_boxes", "details": {"invalid_annotations": 2, "offending_samples": [{"annotation_id": 1, "bbox": [3.15234375, 0.359375, -1.0, 380.80859375], "calculated_area": -380.80859375, "image_height": 414, "image_id": 8, "image_width": 500, "reasons": ["non_positive_width", "area_mismatch"], "reported_area": 179982.63668823242}, {"annotation_id": 2, "bbox": [312.0, 15.2734375, 230.03515625, 380.7265625], "calculated_area": 87580.49429321289, "image_height": 396, "image_id": 7, "image_width": 312, "reasons": ["exceeds_image_width"], "reported_area": 87580.49429321289}], "total_annotations": 668}, "metric_value": 2.0, "passed": false}, "before": {"check_name": "degenerate_boxes", "details": {"invalid_annotations": 0, "offending_samples": [], "total_annotations": 668}, "metric_value": 0.0, "passed": true}, "expected_invalid": 2, "probe": "boxes"}
.{"after": {"check_name": "duplicate_images", "details": {"hash_bits": 64, "image_pairs": [{"hamming_distance": 0, "image_id_a": 3, "image_id_b": 665, "similarity": 1.0}], "similarity_threshold": 0.94, "total_images": 2}, "metric_value": 1.0, "passed": false}, "before": {"check_name": "duplicate_images", "details": {"hash_bits": 64, "image_pairs": [], "similarity_threshold": 0.94, "total_images": 1}, "metric_value": 0.0, "passed": true}, "original_sha256": "f4471041d9383442389e6fb4c1b05c67876272747c8095d36b43ce81a053122c", "probe": "recompression", "quality": 65, "recompressed_sha256": "54d42064e05f21ee9053b2fed11e12e44e7d9ebbb588b80de39a2157c4150cc4", "source": "cat.0.jpg"}
.{"excluded_fixture_categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}], "probe": "source_baseline", "status": "failed"}
{"command": "'<REPO>/.venv-dvc/bin/dvc' init --no-scm", "exit_code": 0, "probe": "process", "stderr": "", "stdout": "Initialized DVC repository.\n\nWhat's next?\n------------\n- Check out the documentation: <https://dvc.org/doc>\n- Get help and share ideas: <https://dvc.org/chat>\n- Star us on GitHub: <https://github.com/treeverse/dvc>\n"}
{"command": "'<REPO>/.venv-dvc/bin/dvc' repro split", "exit_code": 0, "probe": "process", "stderr": "2026-09-18 22:45:14,873 INFO Compuerta de calidad OK (status=warning).\n", "stdout": "Running stage 'quality_gate':\n> '<REPO>/app/.venv/bin/python' dvc_gate_stage.py\nGenerating lock file 'dvc.lock'\nUpdating lock file 'dvc.lock'\n\nVerifying data sources in stage: 'data/raw/annotations.dvc'\n\nVerifying data sources in stage: 'data/raw/images.dvc'\n\nRunning stage 'split':\n> '<REPO>/app/.venv/bin/python' dvc_split_stage.py\nUpdating lock file 'dvc.lock'\nUse `dvc push` to send your updates to remote storage.\n"}
{"after_status": "failed", "before_status": "warning", "check": {"action": "fail", "check_name": "min_images_per_class", "details": {"classes_below_minimum": [{"category_id": 3, "category_name": "dog", "image_count": 300, "image_ids": [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664]}, {"category_id": 4, "category_name": "cat", "image_count": 301, "image_ids": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 40, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634]}], "criterion": {"metric": "metric_value", "operator": ">=", "threshold": 601.0}}, "metric_value": 300.0, "passed": false}, "original_threshold": 300.0, "probe": "threshold"}
{"command": "'<REPO>/app/.venv/bin/python' dvc_gate_stage.py", "exit_code": 1, "probe": "process", "stderr": "2026-09-18 22:45:20,950 ERROR Compuerta de calidad BLOQUEADA: el reporte tiene status=failed.\n", "stdout": ""}
{"command": "'<REPO>/.venv-dvc/bin/dvc' repro split", "exit_code": 255, "probe": "process", "stderr": "2026-09-18 22:45:21,964 ERROR Compuerta de calidad BLOQUEADA: el reporte tiene status=failed.\nERROR: failed to reproduce 'quality_gate': failed to run: '<REPO>/app/.venv/bin/python' dvc_gate_stage.py, exited with 1\n", "stdout": "Running stage 'quality_gate':\n> '<REPO>/app/.venv/bin/python' dvc_gate_stage.py\n"}
{"marker_exists": false, "probe": "downstream", "split_executed": false, "split_report_exists": false}
.
3 passed in 14.07s
```
