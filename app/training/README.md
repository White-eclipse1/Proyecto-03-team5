# Jobs de entrenamiento (APP-03)

El portal lanza entrenamientos que corren **fuera del request HTTP**:

```text
Training (navegador) ──POST /api/ml/training/jobs──▶ nginx ──▶ ml-api (training/server.py)
                                                                   │ valida y encola
                                                                   ▼
                                       MariaDB: ml_training_jobs, ml_training_job_logs
                                                                   ▲
                       worker de OPS-04 ── claim_next / start / report_progress / log / succeed / fail
```

- `queue.py`: la cola. La importan la API y el worker; es la única que escribe esas
  tablas.
- `server.py`: la API (servicio `ml-api` de `docker-compose.yml`, puerto interno
  8001, publicado por nginx en `/api/ml/`).
- `releases.py`: lee `reports/versions.json` y los archivos del release para validar
  cada POST en el servidor.

## API

| Método y ruta | Respuesta | Errores (`ErrorResponse`) |
|---|---|---|
| `GET /api/ml/training/jobs` | `TrainingJobsResponse`, más reciente primero | — |
| `POST /api/ml/training/jobs` | **202** + `TrainingJob` en `queued` y `Location` | 400 `invalid_json`, 422 `invalid_request`, 422 `release_not_found`, 409 `training_blocked` |
| `GET /api/ml/training/jobs/{job_id}` | `TrainingJob` | 404 `job_not_found` |
| `GET /api/ml/training/jobs/{job_id}/logs?after=<seq>` | `TrainingLogsResponse` con las líneas de `seq > after` | 400 `invalid_request`, 404 `job_not_found` |
| `GET /api/ml/runs` | `RunsResponse`: runs de entrenamiento de MLflow, más recientes primero (APP-04) | 503 `mlflow_unavailable` (reintentable), 503 `mlflow_not_configured` |
| `GET /api/ml/runs/{run_id}/curves` | `RunCurvesResponse`: historial por época de cada métrica (APP-04) | 400 `invalid_request`, 404 `run_not_found`, 503 como arriba |
| `GET /api/ml/evaluation` | `EvaluationOverview`: candidato congelado y su evaluación final de test (APP-05) | — (lo que no cuadra va en `problem`) |
| `GET /api/ml/evaluation/predictions.csv` | CSV de predicciones por recorte de ML-09, como descarga | 404 `predictions_not_available` |
| `GET /api/ml/crops/{crop_id}` | PNG del recorte `img<image_id>-ann<annotation_id>` | 400 `invalid_request`, 404 `crop_not_found`, 503 `crops_not_configured` |
| `GET /api/ml/models` | `ModelsResponse`: versiones del registro de OPS-06 con su paquete y su publicación en S3 (APP-06) | 503 `registry_unavailable` |
| `GET /api/ml/models/{version}` | `RegisteredModelVersion` de esa versión (APP-06) | 404 `model_not_found`, 503 como arriba |
| `GET /api/ml/models/{version}/files/{name}` | Archivo del paquete, como descarga (APP-06) | 404 `file_not_found` (no es del paquete), 404 `file_not_available` (falta `dvc pull`), 409 `file_not_servable` (sha256 distinto) |
| `POST /api/ml/inference` | `InferenceRequest` (recorte del portal) → `InferenceResponse` (APP-07) | 400 `invalid_json`, 422 `invalid_request`, 422 `crop_not_found`, 422 `crop_not_available`, errores del modelo (abajo) |
| `POST /api/ml/inference/upload` | multipart `model_name`, `model_version`, `file` → `InferenceResponse` (APP-07) | 413 `image_too_large`, 415 `unsupported_image_type`, 422 `invalid_image`, 422 `invalid_request`, errores del modelo |

El POST aplica `training_request_rejection` (ml_contracts.py): bloquea un Quality
Gate `failed`, un release sin `provenance.json` o `manifest.json` válidos, y un
`manifest_hash` que no sea el del release. Un release desconocido es **422 y no
404**, porque el portal interpreta un 404 del POST como "servicio no conectado".

## Runs de MLflow (APP-04)

`training/experiments.py` traduce la API de MLflow a los contratos. Lee MLflow **en
cada petición** (sin caché), así que un cambio en un run se ve en el portal la
siguiente vez que se consulta. `ml-api` usa `MLFLOW_TRACKING_URI` con 2 reintentos y
10 s de timeout: si MLflow no responde, el portal recibe un 503 en segundos.

### Qué debe registrar cada run de entrenamiento (ML-04 / OPS-04)

Los nombres viven en `app/tracking/run_schema.py`; importa las constantes en vez de
escribirlos a mano. **Un run sin `dataset_version` y `manifest_hash` válidos no
aparece en Experiments.**

| Qué | Cómo | Ejemplo |
|---|---|---|
| Release de P2 | tag `dataset_version` | `v0.1.1` |
| Manifiesto 70/20/10 | tag `manifest_hash` | `sha256:178b28…` |
| Commit del código | tag `mlflow.source.git.commit` (SHA completo; en Docker no hay `.git`, ponlo explícito) | `3f2a9c1e…` (40 hex) |
| Job que lo lanzó | tag `training_job_id` | `job-55a8…` |
| Hiperparámetros | `log_params` con los nombres de `TrainingParams` | `optimizer`, `batch_size`, `max_epochs`, `learning_rate`, `image_size`, `hidden_layers`, `dropout`, `seed`, `patience`, `min_delta` |
| Curvas | `log_metric(nombre, valor, step=época)`, desde la época 1 | `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy` |

Un valor no finito (NaN o ±inf) se sirve como `null`: el run sigue en la lista, la
celda dice "no finito" y en la curva esa época queda como un hueco.

Las columnas de métricas de validación de la tabla son **todas** las métricas `val_*`
de los runs (por ejemplo, `val_accuracy_top1` si la evaluación la registra), y se
pueden ordenar.

## Evaluation (APP-05)

`training/evaluation_view.py` lee lo que dejan ML-08 y ML-09 en `reports/` en cada
petición:

- `candidates/ml08_candidate.json`: el candidato elegido con validation y congelado.
- `evaluations/test/<evaluation_id>.json` (contrato `Evaluation`) y
  `<evaluation_id>.predictions.csv`.

| Estado | Cuándo | Qué ve el portal |
|---|---|---|
| `candidate_not_frozen` | No hay candidato, o el archivo no es válido | Ningún resultado de test, aunque exista uno en disco |
| `candidate_frozen` | Hay candidato pero no una evaluación de test que sea suya | El candidato; sin métricas de test |
| `evaluated` | Una sola evaluación `split: test` del mismo run, checkpoint, release y manifest, creada después de `frozen_at` | Métricas, matriz, ejemplos y CSV |

Si hay una evaluación que no cuadra (otro run o manifest, anterior al
congelamiento, de validation o más de una), se oculta y `problem` explica por qué.
El portal recalcula accuracy, F1 y métricas por clase desde la matriz, y compara
con 0.85 sin redondear.

Los ejemplos muestran el recorte de ML-01: `ml-api` busca `crop_id` en
`reports/crops.json` y sirve `CROPS_DIR/<crop_path>` (`./data/crops`, montado de
solo lectura). El cliente nunca manda una ruta. Sin `dvc pull` de `data/crops` la
pantalla funciona igual y cada ejemplo dice "Recorte no disponible".

## Models (APP-06)

`training/models_view.py` lee `reports/models/registry.json` (OPS-06) y
`reports/models/s3_publications.json` (OPS-07) en cada petición:

- **`servable`**: este servidor tiene cada archivo del paquete
  (`data/models/<modelo>/<versión>/`) con el sha256 registrado. Inference solo
  ofrece versiones `servable`.
- **Publicación:**
  - `published` solo si cada archivo del paquete está en S3 con el sha256
    registrado, el `ChecksumSHA256` de S3 y la descarga de verificación de OPS-07
    coinciden **y** S3 lo confirma al responder (ver "Estado real");
  - `not_published` si no hay registro de publicación;
  - `inconsistent` si algo no cuadra (otro run o checkpoint, un archivo que falta,
    un checksum distinto, el archivo ilegible, o un objeto que S3 no tiene o cuyo
    `ChecksumSHA256` no es el registrado). Explica el motivo y no muestra keys;
  - `unverifiable` ("No verificable") si S3 no se pudo consultar: sin credenciales de
    AWS, acceso denegado, token vencido o sin red. Explica el motivo y no muestra keys.
- **Descargas:** solo los archivos que el registro lista para esa versión, y solo si
  tienen su sha256.
- **Estado real:** `ml-api` hace `head_object` (con `ChecksumMode=ENABLED` y el
  `VersionId` registrado) de cada objeto de la publicación, en la región registrada
  (`training/s3_verification.py`). Las respuestas se guardan 60 s por
  (bucket, key, version_id). Sin credenciales responde `unverifiable`, nunca un 500.

### Credenciales de AWS para Models

Son opcionales y no se escriben en el repo ni en `.env`: `docker-compose.yml` pasa a
`ml-api` `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` y `AWS_SESSION_TOKEN` desde el
entorno de quien levanta el stack (vacías por defecto; boto3 las ignora y Models
muestra "No verificable"). Con el perfil SSO del equipo:

```bash
# Git Bash / Linux / macOS
eval "$(aws configure export-credentials --profile mlops-p2 --format env)"
docker compose up -d ml-api
```

```powershell
# PowerShell
aws configure export-credentials --profile mlops-p2 --format powershell | Invoke-Expression
docker compose up -d ml-api
```

Las credenciales temporales del SSO vencen: cuando vencen, Models vuelve a mostrar
"No verificable" (token vencido) hasta repetir los dos pasos. Basta con permiso
`s3:GetObject` sobre el prefijo `models/` del bucket (`s3:GetObjectVersion` si la
publicación registra `VersionId`); sin `s3:ListBucket`, S3 responde
403 en vez de 404 a una key inexistente y Models lo muestra como "No verificable".
Fuera de Docker (`uv run python -m training.server`) se usa la cadena por defecto de
AWS, por ejemplo `AWS_PROFILE=mlops-p2`.

## Inference (APP-07)

`training/inference.py` clasifica con el modelo real:

1. **Imagen.** Una subida se valida por su contenido: PNG o JPEG que Pillow pueda
   decodificar, hasta 10 MB y 40 millones de pixeles. El nombre del archivo y el tipo
   que declara el navegador no deciden nada. Un recorte del portal se busca en
   `reports/crops.json` por release, `image_id` y `annotation_id`, y su sha256 debe
   ser el registrado.
2. **Modelo.** `model_name` + `model_version` (SemVer, por ejemplo `dog-cat-resnet18`
   `1.0.0`) se resuelven en el **registro de OPS-06** (`reports/models/registry.json`)
   con `classification.registry`. El paquete está en `data/models/<modelo>/<versión>/`
   (`dvc pull data/models.dvc`); se verifica el sha256 de cada archivo contra el
   registro, y el `image_size` y las `hidden_layers` del checkpoint deben ser los
   registrados. Run, checkpoint y release salen del registro. Los últimos 4 modelos
   quedan en memoria.
3. **Predicción.** `preprocess_image` (el preprocesamiento de validation/test de
   ML-02) y `predict_proba` (modo eval, sin gradiente). La respuesta dice qué
   checkpoint, con qué sha256 y con qué `image_size` se clasificó.

| Error del modelo | Cuándo |
|---|---|
| 422 `model_not_found` | La versión o el nombre no están en el registro (422 y no 404: el portal lee un 404 como "no conectado") |
| 409 `model_not_servable` | La versión solo tiene metadata, un archivo del paquete no tiene el sha256 registrado o el checkpoint no coincide con la arquitectura registrada |
| 503 `model_package_missing` | Falta el paquete en el servidor: hay que correr `dvc pull data/models.dvc` |
| 503 `registry_unavailable` | `registry.json` no se puede leer |

`ml-api` lee `reports/models/registry.json` y monta `./data/models` de solo lectura;
`package_path` se resuelve contra la carpeta padre de `REPORTS_DIR` (`/app` en
Docker, la raíz del repo en local). `tests/test_inference_real_model.py` comprueba,
con el paquete real `1.0.0`, que la inferencia reproduce las probabilidades de ML-09
en los 71 recortes de test (se omite si no hay `data/models` ni `data/crops`).

### Enviar a la cola de anotación (APP-08)

"Enviar a cola de anotación" manda el resultado al backend de Node
(`POST /api/images/from-inference`). Desde APP-09, el backend comprueba antes de
guardar:

- que modelo, versión, run y checkpoint estén en `reports/models/registry.json`;
- que el recorte tenga el sha256 de `reports/crops.json`, o que la imagen subida tenga
  el sha256 de `sourceRef`;
- que la clase sea la de mayor probabilidad.

Si algo no cuadra responde 400 con el motivo (`backend/src/logic/inference-traceability.ts`).

## Interfaz para el worker (OPS-04)

```python
from presentation.ml_contracts import ContractError
from storage.db import get_engine
from training.queue import TrainingJobQueue

queue = TrainingJobQueue(get_engine())
queue.create_tables()  # idempotente

job = queue.claim_next("worker-1")  # None si no hay jobs en cola
if job is not None:
    try:
        # crear el run de MLflow (OPS-03) con job.params, job.dataset_version y job.manifest_hash
        queue.start(job.job_id, experiment_id=experiment_id, run_id=run_id)
        queue.log(job.job_id, "Run creado.")
        for epoch in range(1, job.params.max_epochs + 1):
            ...  # entrenar una época
            queue.report_progress(job.job_id, epoch=epoch, metrics={"val_accuracy": acc})
            queue.log(job.job_id, f"Época {epoch}/{job.params.max_epochs}")
        queue.succeed(job.job_id, checkpoint=f"runs:/{run_id}/checkpoints/best.pt")
    except Exception as exc:
        queue.fail(
            job.job_id, ContractError(code="training_error", message=str(exc), retryable=False)
        )
```

| Método | Desde | Hacia | Notas |
|---|---|---|---|
| `claim_next(worker_id)` | `queued` | `queued` (tomado) | `SELECT ... FOR UPDATE SKIP LOCKED`: dos workers nunca toman el mismo job |
| `start(job_id, experiment_id=, run_id=)` | `queued` tomado | `running` | Exige haberlo tomado con `claim_next` |
| `report_progress(job_id, epoch=, metrics=)` | `running` | `running` | `epoch <= params.max_epochs`; las métricas se guardan como `float`, y NaN/±inf como `null` (el job sigue) |
| `log(job_id, message, *, level="info")` | cualquiera | — | `level` solo por nombre (`level="warning"`): `info`, `warning` o `error`; `seq` consecutivo por job |
| `succeed(job_id, checkpoint=)` | `running` | `succeeded` | El checkpoint debe ser `runs:/<mismo run_id>/...` |
| `fail(job_id, ContractError)` | `queued` o `running` | `failed` | Puede fallar antes de crear el run |

Toda transición se valida contra `TrainingJob` antes de guardarse; si deja el job
fuera del contrato lanza `InvalidTransitionError` y la fila no cambia. Un `job_id`
inexistente lanza `JobNotFoundError`.

**Pendiente para OPS-04:** si un worker muere con un job tomado o en `running`, ese
job se queda así. Recuperarlo (por ejemplo, marcar `failed` los jobs con
`claimed_at` viejo) le toca al worker.

## Pruebas

```bash
uv run pytest tests/test_training_queue.py tests/test_training_api.py tests/test_ml_api_wiring.py
```

`test_experiment_runs.py` prueba el mapeo MLflow → contratos con entidades reales de
MLflow y un cliente falso. `test_experiments_mlflow_integration.py` corre contra un
MLflow real (con `MLFLOW_INTEGRATION=1` y el stack levantado), incluido el Agent Test
de APP-04: cambiar un tag y una métrica en MLflow y ver el cambio en la API.

`test_training_queue.py` corre sobre SQLite. Con
`TRAINING_QUEUE_DATABASE_URL=mysql+pymysql://root:<clave>@127.0.0.1:3306/image_repo`
(y `docker compose up -d --wait mariadb`) corre sobre MariaDB, incluida la prueba de
`SKIP LOCKED`; así lo hace el job de CI "Cola de entrenamiento en MariaDB (APP-03)".


## OPS-04 - training worker

El entrenamiento corre en un proceso independiente del API:

```text
ml-api -> MariaDB queue -> training-worker -> run_training() -> MLflow
```

El servicio `training-worker`:

- reclama jobs con `claim_next`;
- ejecuta `run_training`;
- persiste progreso y logs mediante `JobQueueHooks`;
- marca `succeeded` con el checkpoint de MLflow;
- marca `failed` si el trainer lanza una excepcion;
- al reiniciar recupera jobs no terminales reclamados por el mismo `TRAINING_WORKER_ID`.

Variables relevantes:

- `DATABASE_URL`: cola persistente en MariaDB.
- `MLFLOW_TRACKING_URI`: servidor MLflow persistente.
- `TRAINING_WORKER_ID`: identidad estable del worker entre reinicios.
- `GIT_COMMIT`: SHA Git de 40 caracteres del codigo que ejecuta el entrenamiento.

Para una ejecucion real desde Compose, define el commit actual antes de levantar el worker:

```powershell
$env:GIT_COMMIT = git rev-parse HEAD
```

Despues, el servicio puede arrancarse con:

```text
docker compose up --build training-worker
```

No se debe usar un SHA inventado: el valor queda registrado en MLflow como procedencia del entrenamiento.
