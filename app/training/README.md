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

Las columnas de métricas de validación de la tabla son **todas** las métricas `val_*`
de los runs (por ejemplo, `val_accuracy_top1` si la evaluación la registra), y se
pueden ordenar.

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
| `report_progress(job_id, epoch=, metrics=)` | `running` | `running` | `epoch <= params.max_epochs`; las métricas se guardan como `float` |
| `log(job_id, message, level="info")` | cualquiera | — | `level`: `info`, `warning` o `error`; `seq` consecutivo por job |
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
