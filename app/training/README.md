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

El POST aplica `training_request_rejection` (ml_contracts.py): bloquea un Quality
Gate `failed`, un release sin `provenance.json` o `manifest.json` válidos, y un
`manifest_hash` que no sea el del release. Un release desconocido es **422 y no
404**, porque el portal interpreta un 404 del POST como "servicio no conectado".

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

`test_training_queue.py` corre sobre SQLite. Con
`TRAINING_QUEUE_DATABASE_URL=mysql+pymysql://root:<clave>@127.0.0.1:3306/image_repo`
(y `docker compose up -d --wait mariadb`) corre sobre MariaDB, incluida la prueba de
`SKIP LOCKED`; así lo hace el job de CI "Cola de entrenamiento en MariaDB (APP-03)".
