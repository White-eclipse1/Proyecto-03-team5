# Evidencia APP-10: recorrido completo desde el portal con IDs reales

Ejecución: 2026-10-03.

- **Stack:** completo y aislado (`COMPOSE_PROJECT_NAME=app10check`), con credenciales temporales. El portal en `http://localhost:8080`.
- **Datos reales:**
  - recortes de ML-01 (`dvc pull crops`);
  - paquete `dog-cat-resnet18 1.0.0` de OPS-06 (`dvc pull data/models.dvc`);
  - snapshot de MLflow de ML-07 (`dvc pull data/mlflow-snapshot.dvc` y `uv run python -m tracking.snapshot restore --compose "docker compose -p app10check --env-file <ruta>"`). Así el run del modelo publicado existe en el MLflow del stack con su mismo ID.

La parte automatizable es [`tests/test_app10_portal_smoke.py`](../test_app10_portal_smoke.py). Hace el recorrido en orden, pasando por nginx igual que el navegador: lee los reportes estáticos que usa Training (`/reports/...`), luego las APIs de las pantallas (`/api/ml/...`) y la cola de anotación (`/api/images/...`). Comprueba que cada ID encaja con el siguiente. Sin stack se omite.

```bash
APP10_PORTAL_URL=http://localhost:8080 uv run pytest tests/test_app10_portal_smoke.py -v
# 1 passed
```

## Cadena de IDs (Agent Test)

Corrida del 2026-10-03 (`APP10_TRACE_OUT`):

| Paso | ID | Encaja con |
|---|---|---|
| Release P2 aprobado | `v0.1.1`, Quality Gate `warning` (no bloquea; `v0.1.0` está `failed`) | — |
| DVC | `annotations` `c7cb86ae7ece94ef7b853620e464a4d7.dir`, `images` `951150dd4fb053f4665089fcb37a1c87.dir` | Son los hashes de `provenance` del manifiesto |
| Manifest | `sha256:178b28bddb1bef5c05975fc7150710658627e31af08a1a12d94b98f9e99cb2b2` | Es el del job, el run, el candidato y la versión |
| Training job | `job-a960b7bade434cf8afb0bd13a883da09` → `succeeded`, persistido y listado | Su run `5ff1a617c5fb45029ba548e7c22e73bf` está en Experiments con `v0.1.1` y el mismo manifest |
| Candidato (ML-08) | run `bb448230424146349a969253d30db43b`, checkpoint `runs:/bb44…/checkpoints/best.pt`, sha256 `84d6c88b…0d52` | Aparece en Experiments (`ml07-v1-r03-sgd`). Evaluation: test 68/71 = 0.9577 |
| Model version | `dog-cat-resnet18 1.0.0`, publicada en S3 (4 objetos `models/dog-cat-resnet18/1.0.0/…`) y servible | Mismo run, checkpoint sha256, release y manifest que el candidato |
| Inference | `img107-ann135` → dog 0.99995 (`inf-b634c2a373444b36`) | Run y checkpoint sha256 de 1.0.0. Mismas probabilidades que guardó ML-09 para ese recorte |
| Cola de anotación | imagen `1`, `created: true` | La imagen guardada es idéntica byte a byte al recorte. La cola verificó la trazabilidad contra `registry.json` y `crops.json` (APP-09) |

## Recorrido en el portal

| Criterio (issue #52) | Pantalla | Resultado |
|---|---|---|
| Seleccionar release P2 aprobado | Training: se elige `v0.1.1` ([captura](app-10-1-training.jpg)) | ✅ |
| Visualizar procedencia | Training: hashes DVC de `annotations` e `images` y Quality Gate | ✅ |
| Visualizar split/manifiesto | Training: 420/120/60 imágenes (70/20/10) y el manifest hash | ✅ |
| Lanzar training corto | Prueba E2E: job de 1 época, el POST responde en el acto (el worker entrena aparte) | ✅ |
| Training job queda persistido | El mismo `job_id` se vuelve a leer y aparece en la lista | ✅ |
| MLflow run aparece en Experiments | El run del job y el del candidato `ml07-v1-r03-sgd` ([captura](app-10-2-experiments.jpg)) | ✅ |
| Candidate seleccionado es visible | Evaluation: run, checkpoint, sha256, release, manifest y `best_val_loss` ([captura](app-10-3-evaluation-candidate.jpg)) | ✅ |
| Evaluation muestra datos reales | 68 de 71, 0.9577, meta alcanzada; el modelo enlaza a `dog-cat-resnet18 v1.0.0` ([captura](app-10-4-evaluation-test.jpg)) | ✅ |
| Models muestra versión real | `1.0.0`, run `bb44…`, "Paquete disponible", "Publicado en S3" ([captura](app-10-5-models.jpg)) | ✅ |
| Versión publicada puede seleccionarse | "Usar en Inference" abre Inference con `1.0.0` | ✅ |
| Inference produce predicción real | `img369-ann355`: cat 63.2 %, checkpoint `84d6c88b…` ([captura](app-10-6-inference-queue.jpg)) | ✅ |
| Send to annotation queue crea registro real | "Enviado a la cola de anotación como imagen 2"; en Buscar aparece como Pendiente ([captura](app-10-7-annotation-queue.jpg)) | ✅ |
| IDs son trazables entre todas las pantallas | Tabla de arriba. Además, Evaluation enlaza la versión registrada del mismo checkpoint (ver abajo) | ✅ |
| No se utilizan mocks para completar el flujo | Cada respuesta se valida con los contratos; release, recortes, modelo, MLflow y S3 son los reales | ✅ |

## Cambio hecho durante APP-10

**Evaluation decía "Modelo: Sin registrar"** aunque la versión `1.0.0` está registrada. ML-09 evalúa antes de que OPS-06 registre la versión, así que el JSON de la evaluación trae `model_name: null` y la pantalla no tenía cómo enlazarla.

Ahora Evaluation muestra la versión registrada con el **mismo run y el mismo checkpoint sha256**, como enlace a Models. Si no hay ninguna, sigue diciendo "Sin registrar". Las pruebas en rojo están en `frontend/tests/ml-evaluation.test.tsx`; la corrección en `Evaluation.tsx`.
