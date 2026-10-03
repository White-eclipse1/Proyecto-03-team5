# OPS-10 — Auditoría final con el prompt de evaluación del Proyecto 3

Evaluación interna del release candidate `eb9966a` (White-eclipse1/Proyecto-03-team5), 2026-10-03; los commits posteriores solo agregan pruebas, `tests/conftest.py` y esta evidencia, sin cambiar el código que se ejecuta. Se siguió el prompt del profesor como evaluador: clon limpio, solo el README, cifras recalculadas desde datos primarios y pruebas destructivas en copias aisladas. Equipo: team5.

Capturas del portal en el clon limpio: [Models](ops-10-models.jpg) · [Evaluation](ops-10-evaluation.jpg). Reporte HTML (Fase 4): [`ops-10-reporte-team5.html`](ops-10-reporte-team5.html).

## A. Requisitos mínimos

| Estado | Requisito | Evidencia |
|---|---|---|
| CUMPLE | M1 — El clon arranca siguiendo el README | Clon limpio (worktree en `eb9966a`) siguiendo solo «Proyecto 3 completo en un clon limpio»: 4 `dvc pull`, credenciales exportadas, `GIT_COMMIT`, `docker compose up -d --build --wait` → 9 servicios `Healthy`, exit 0; `tracking.snapshot restore` → 12 runs OK. Portal `/` (Tablero) y `/ml/training` cargan; un job real `job-771b5a16…` llegó a `succeeded` (APP-10). |
| CUMPLE | M2 — Consume un release P2 aprobado y versionado | `v0.1.1` (gate `warning`, aprobado; `v0.1.0` `failed` → 409). Hashes DVC de `data/raw/{images,annotations}.dvc` (`951150dd…`, `c7cb86ae…`) = `provenance` del manifiesto = tags del run candidato. `dvc status`: «Data and pipelines are up to date». |
| CUMPLE | M3 — Train, validación y prueba aislados | Intersecciones en el manifiesto real: `crop_id`, `source_image_id` y `duplicate_group` = 0 en los tres pares. Transforms aleatorios solo en train (`RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter`); validation/test vacíos. Candidato elegido con `best_val_loss` en validation y congelado antes del test. |
| CUMPLE | M4 — Modelo recargable para inferencia | `aws s3api get-object` de `models/dog-cat-resnet18/1.0.0/checkpoint/best.pt` → SHA-256 `84d6c88b…0d52` (= registrado); cargado en un proceso nuevo con su class map y preprocesamiento: `dog/img218-ann249` → dog 1.0. |

**Compuerta:** `NO ACTIVADA`

## B. Tabla de calificación

| Puntos obtenidos | Requisito | Lo que falta |
|---|---|---|
| 4 / 4 | 1.1 Traspaso real desde Proyecto 2 | — (Training lista `v0.1.0` (gate `failed`) y `v0.1.1` (`warning`); cambiar a `v0.1.0` muestra otro reporte y split P2 y el POST de un job responde 409 `training_blocked`; `v0.1.1` trae procedencia DVC, referencia de calidad y manifiesto.) |
| 5 / 5 | 1.2 Recortes COCO y selección de clases previa | — (`reports/crops.json`: `image_id`, `annotation_id`, categoría, bbox y `crop_box` por recorte; 0 rechazos en `v0.1.1`; dog 300 y cat 301 imágenes originales. 6/6 recortes muestreados idénticos píxel a píxel a su caja del COCO.) |
| 5 / 5 | 1.3 Manifiesto 70/20/10 reproducible y sin fuga | — (469/128/71 recortes (0.702/0.192/0.106); dog/cat 221/248, 61/67, 43/28; 420/120/60 originales. Dos generaciones con seed 42: mismos IDs y mismo `manifest_hash` `sha256:178b28bd…`. Grupos indivisibles: intersecciones 0.) |
| **14 / 14** | **Subtotal Integración y datos de entrenamiento** |  |
| 5 / 5 | 2.1 Clasificador entrenado por el equipo | — (ResNet18 `IMAGENET1K_V1` (torchvision), `layer4` + cabeza entrenables, class map dog=0/cat=1 (`test_trainable_policy_freezes_the_rest`, `test_optimizer_steps_change_trainable_weights_and_keep_frozen_ones`).) |
| 4 / 4 | 2.2 Minibatches y parámetros configurables | — (Un paso de optimizador por minibatch (`test_optimizer_steps_once_per_minibatch`); 8 peticiones inválidas por API → 422/409 sin crear jobs (APP-09) y validación del formulario (vitest, 35 tests).) |
| 4 / 4 | 2.3 Semillas y aumentación controlada | — (Cada run registra `seed_split`, `seed_dataloader`, `seed_augmentation`, `seed_weight_init`, versiones y plataforma; mismo orden de muestras con la misma semilla (`test_two_short_runs_with_the_same_seed_see_the_same_sample_order`).) |
| 5 / 5 | 2.4 Curvas, early stopping y mejor checkpoint | — (Curvas `curves/` por run; `patience`/`min_delta`; secuencia inyectada fuerza la parada y restaura la mejor época (`test_injected_metrics_force_early_stopping_and_restore_the_best_epoch`).) |
| **18 / 18** | **Subtotal Modelo, entrenamiento y reproducibilidad** |  |
| 6 / 6 | 3.1 Diez experimentos válidos | — (12 runs `FINISHED` de `ml07-v1`; los 7 parámetros varían (3–4 valores cada uno); mismo manifest y clases. 10 válidas listadas abajo (r12 se excluye por coincidir en `best_val_loss` con r01).) |
| 5 / 5 | 3.2 Registro completo en MLflow | — (Por run (API de MLflow): parámetros, 4 semillas, `git_commit`, hashes DVC, `manifest_hash`, `classes`, historia por época de `val_loss` y artefactos `curves/` y `checkpoints/best.pt`: completo en 12/12.) |
| 3 / 3 | 3.3 Comparación y elección reproducible | — (`reports/candidates/ml08_candidate.json`: `r03-sgd` por `best_val_loss` (validation), congelado `2026-10-02T00:37:26Z` < evaluación `01:26:33Z`; único run con `candidate=true`.) |
| **14 / 14** | **Subtotal Experimentos y MLflow** |  |
| 5 / 5 | 4.1 Protocolo de prueba congelado | — (Una sola evaluación del checkpoint congelado (sha256 verificado) sobre los 71 recortes de test; CSV y JSON por muestra con probabilidades; auditoría reproducible sin escribir.) |
| 4 / 4 | 4.2 Matriz y métricas correctas | — (Recalculado desde el CSV: matriz `[[43,0],[3,25]]` (suma 71), accuracy 0.9577464788732394, F1 macro 0.9548441806232775, P/R/support por clase; igual en MLflow 17/17, API 16/16 y portal 32/32 (ML-10).) |
| 6 / 6 | 4.3 Umbral de 85% en test | — (68/71 ≥ 0.85 con enteros (1360 ≥ 1207); IDs = test congelado; dog y cat cubiertas.) |
| 3 / 3 | 4.4 Interpretación de errores | — (Evaluation muestra los 3 errores y los aciertos con recorte, real, predicha y p; clase más confundida `cat → dog: 3`; baseline 43/71 = 0.6056; explica que el accuracy no oculta un recall bajo y nombra la clase más débil (cat 0.8929).) |
| **18 / 18** | **Subtotal Evaluación final y calidad del clasificador** |  |
| 4 / 4 | 5.1 Paquete de modelo completo | — (`data/models/dog-cat-resnet18/1.0.0/`: checkpoint, `package.json` (arquitectura, class map, preprocesamiento), `dependencies.json` (versiones + sha256 de `uv.lock`) y tarjeta con propósito, release P2, manifest hash, run, métricas de test, limitaciones y pesos preentrenados.) |
| 4 / 4 | 5.2 AWS S3 real y recuperable | — (`head-object`/`get-object` de pesos y tarjeta (1.0.0 y 0.9.0): SHA-256 local = registrado, `ChecksumSHA256` de S3 coincide, ETag ≠ SHA-256 y registrado aparte; `VersionId` = null (bucket sin versionado: «si existe»); inferencia con lo descargado en un proceso limpio.) |
| 2 / 2 | 5.3 Registro y versionado navegables | — (Models resuelve `1.0.0` (run `bb448230`, checkpoint `84d6c88b`, 128 px) y la versión anterior `0.9.0` (run `83e5a40a`, checkpoint `db64816a`, 224 px); Inference con cada una carga su artefacto: `img369-ann355` → cat 0.6321 vs 0.9946.) |
| **10 / 10** | **Subtotal Versión, tarjeta y publicación en S3** |  |
| 4 / 4 | 6.1 Training | — (Release aprobado, procedencia y split, formulario con los 7 parámetros, job fuera del request (202 + worker), estado/logs/checkpoint persistentes; rechaza gate fallido (409) y manifest ajeno.) |
| 3 / 3 | 6.2 Experiments | — (12 corridas reales de MLflow con curvas; un `log_metric` directo en MLflow cambia la UI (APP-09).) |
| 3 / 3 | 6.3 Evaluation | — (Candidato, release, métricas, matriz y ejemplos reales; sin candidato congelado no revela test (APP-05); exporta el CSV por muestra.) |
| 4 / 4 | 6.4 Models | — (Lista 1.0.0 y 0.9.0 con tarjeta, trazabilidad y descargas; «Publicado en S3» solo tras `head-object` + `ChecksumSHA256`; key inexistente → `inconsistent`; sin credenciales → «No verificable»; versión de dataset y de modelo separadas.) |
| 4 / 4 | 6.5 Inference | — (Acepta imagen nueva y recorte; rechaza PDF/SVG/WebP/vacío/truncado; usa la versión elegida y el preprocesamiento de evaluación; probabilidades suman 1; «Enviar a cola» crea un registro real verificado contra el registro.) |
| **18 / 18** | **Subtotal Portal integrado de extremo a extremo** |  |
| 4 / 4 | 7.1 Pruebas que detectan fallas reales y TDD | — (Ciclos Red → Green en los commits de P3. En copia aislada: matriz traspuesta → falla; predicción cambiada en el CSV → 3 tests fallan; duplicados sin unir → falla (test agregado en esta auditoría).) |
| 2 / 2 | 7.2 Prueba de integración del recorrido | — (`tests/test_app10_portal_smoke.py`: release → job → run → candidato → versión → inferencia → cola con IDs trazables (1 passed en el clon limpio); OPS-09 E2E en CI.) |
| 2 / 2 | 7.3 CI, lint y secretos | — (CI: Ruff check/format, pytest, Biome, tsc, tests y build de backend y frontend, Gitleaks sobre HEAD e historial, sin `continue-on-error`; `.gitignore` excluye `.env`, `*.pt`, datos; 0 claves con forma de access key en todo el historial.) |
| **8 / 8** | **Subtotal Pruebas, CI y disciplina de repositorio** |  |

## C. Trazabilidad de extremo a extremo

| Release DVC y hash | Manifiesto 70/20/10 y hash | Run MLflow elegido | Checkpoint | Versión de modelo | S3 bucket/key y hash | Predicción de prueba |
|---|---|---|---|---|---|---|
| `v0.1.1` · images `951150dd…dir` · annotations `c7cb86ae…dir` | `p3-v1` · `sha256:178b28bd…cb2b2` | `ml07-v1-r03-sgd` · `bb448230424146349a969253d30db43b` | `runs:/bb448230…/checkpoints/best.pt` · sha256 `84d6c88b…0d52` | `dog-cat-resnet18 1.0.0` (anterior: `0.9.0`) | `s3://mlops-p2-dvc-cache-280764207006/models/dog-cat-resnet18/1.0.0/checkpoint/best.pt` · SHA-256 `84d6c88b…0d52` | `img107-ann135` → dog 0.99995 (`inf-50a6a0f5…`) · cola: imagen registrada |

## D. Contraste de métricas

| Métrica | Reportado por el equipo | Verificado por el evaluador | ¿Coincide? |
|---|---|---|---|
| Corridas válidas de MLflow | 12 | 12 FINISHED (10 listadas como válidas) | Sí |
| Train / val / test (recortes y originales) | 469/128/71 · 420/120/60 | 469/128/71 · 420/120/60 | Sí |
| Accuracy top-1 en test | 0.9577 | 68/71 = 0.9577464788732394 | Sí |
| F1 macro en test | 0.9548 | 0.9548441806232775 | Sí |
| Total de la matriz de confusión | 71 | 71 | Sí |

Diez corridas válidas (ordenadas por `best_val_loss` en validation):

| Run ID | Entrada | Parámetros que cambian | Mejor val_loss | Estado |
|---|---|---|---|---|
| bb448230… | r03-sgd | sgd, lr 0.01 | 0.0406 | FINISHED (candidato) |
| 83e5a40a… | r07-img224 | image 224 | 0.0574 | FINISHED (versión 0.9.0) |
| e6921aea… | r06-lr3e-4-ep15 | lr 3e-4, épocas 15 | 0.0719 | FINISHED |
| 1d571224… | r02-adamw | adamw | 0.0802 | FINISHED |
| f0cd9825… | r01-base | base (adam, 32, 10, 1e-3, 128, [128], 0.2) | 0.0814 | FINISHED |
| 06fc1940… | r11-adamw-img160-head256-drop0.5 | adamw, lr 3e-4, ép 15, 160, [256], 0.5 | 0.0850 | FINISHED |
| 306817e8… | r04-batch16 | batch 16 | 0.1315 | FINISHED |
| 137244d2… | r10-no-hidden-drop0 | sin capas ocultas, dropout 0 | 0.1516 | FINISHED |
| 21db27c7… | r09-head256-64-drop0.3 | [256,64], dropout 0.3 | 0.1583 | FINISHED |
| f6b548b4… | r05-batch64 | batch 64 | 0.1646 | FINISHED |

## E. Resultado final

```text
Suma de secciones:      100 / 100
Compuerta aplicada:     no
CALIFICACIÓN FINAL:     100
Meta de 85% en test:    alcanzada
Escala:                 Excelente (90-100)
```

## F. Comentario para el equipo

- Aciertos: (1) la cadena release → manifest → run → checkpoint → versión → S3 → inferencia es real y verificable con hashes en cada paso; (2) la evaluación final es reproducible desde el CSV y coincide al bit con MLflow, API y portal; (3) el portal integra las cinco pantallas con datos reales y se prueba de forma adversarial.
- Correcciones hechas en esta auditoría: `dvc status` de un clon limpio marcaba el pipeline cambiado (marcador sin versionar y `__pycache__` en los hashes); solo existía una versión del modelo (se publicó `0.9.0`, otro run real de la matriz, sin métricas de test); la suite no detectaba duplicados separados entre particiones ni una predicción alterada en los artefactos versionados, y llenaba el disco con ~17 GB de temporales (tests de regresión y `conftest.py`).
- Riesgos residuales: el bucket de S3 no tiene versionado (VersionId null; habilitarlo protegería contra sobrescrituras externas); Models necesita credenciales de AWS en `ml-api` para mostrar «Publicado» (sin ellas, «No verificable»); vulnerabilidades npm moderadas en dependencias de producción (minio, react-router) cuya corrección exige un cambio mayor.

## G. Anexo de verificación

```text
$ git rev-parse --short HEAD          # clon limpio
eb9966a
$ dvc pull -r prod data/raw/images.dvc data/raw/annotations.dvc   → 612 files fetched
$ dvc pull -r prod crops                                         → 669 files fetched
$ dvc pull -r prod data/models.dvc                               → 8 files fetched
$ dvc pull -r prod data/mlflow-snapshot.dvc                      → 44 files fetched
$ dvc status                                                     → Data and pipelines are up to date.
$ docker compose -p ops10rc up -d --build --wait                 → 9 servicios Healthy, exit 0
$ uv run python -m tracking.snapshot restore                     → 12/12 runs OK
$ pytest tests/test_app10_portal_smoke.py                        → 1 passed
$ pytest tests/test_app09_portal_adversarial.py (S3 real)        → 30 passed
$ python -m classification.quality_audit                         → MLflow 17/17 · API 16/16 · Portal 32/32 · recortes 7/7
$ GET /api/ml/models                                             → 1.0.0 (sha 84d6c88b, 128 px, published) · 0.9.0 (sha db64816a, 224 px, published, sin test)
$ POST /api/ml/inference img369-ann355                           → 1.0.0: cat 0.6321 · 0.9.0: cat 0.9946
$ POST /api/ml/training/jobs dataset_version=v0.1.0              → HTTP 409
$ aws s3api head-object / get-object (1.0.0 y 0.9.0, pesos y tarjeta) → SHA-256 = registrado; ETag ≠ SHA-256; VersionId null
$ python -m classification.publication infer <best.pt descargado> → proceso limpio: dog 1.0 (ambas versiones)
$ intersecciones M3 (crop_id, source_image_id, duplicate_group)   → 0, 0, 0
$ build_release_manifest ×2 (seed 42)                            → mismos IDs y mismo manifest_hash
$ 6 recortes vs COCO original                                    → 6/6 idénticos píxel a píxel
$ pytest (suite completa, estado final)                         → 1156 passed, 53 skipped (pico de temporales 301 MB)
$ vitest frontend / backend                                      → 513 / 121 passed; tsc, Biome y build OK
$ git log --all -p | grep (AKIA|ASIA)[A-Z0-9]{16}                → 0 coincidencias

Datos malos inyectados (copia aislada, restaurada):
- generador sin unir duplicados → `test_declared_duplicates_share_group_and_split` falla (antes la suite pasaba: se agregó el test)
- matriz traspuesta en ML-09 → la suite falla
- predicción cambiada en el CSV de test → 3 tests de integridad fallan (antes la suite pasaba: se agregaron)
- key inexistente en s3_publications.json → Models «inconsistent» (APP-09)

No verificado y por qué:
- GPU: no disponible; no aplica (las corridas ya están persistidas y el entrenamiento corre en CPU).
- Clics en el navegador: las pantallas se leyeron renderizadas en Chrome headless y por API; la procedencia que aparece al elegir el release en Training se cubre con los tests del formulario.
- Puertos: el stack se levantó con un proyecto de Compose aislado y puertos propios porque los de por defecto estaban ocupados; los pasos del README no cambian.

Hallazgo de entorno corregido: una corrida completa de pytest dejaba ~17 GB de temporales y pytest conserva las tres últimas; el disco se llenó y la suite falló con «No space left on device». `tests/conftest.py` vacía el `tmp_path` de cada test que pasa (pico 301 MB) sin reusar rutas.

Estado final: el stack aislado se desmontó (`down -v`), los worktrees se borraron y el repositorio quedó limpio.
```
