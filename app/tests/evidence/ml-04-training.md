# Evidencia ML-04: entrenamiento real registrado en el servidor MLflow

Ejecución: 2026-10-01. Rama `feat/ml-04-training-loop-mlflow`. Servidor MLflow de
OPS-03 levantado con `docker compose up -d --wait mlflow` (MariaDB + MinIO),
expuesto en `http://127.0.0.1:5050` (`MLFLOW_PORT=5050`; macOS ocupa el 5000).
Datos: manifiesto real `reports/releases/v0.1.1/manifest.json` y `data/crops` de DVC.

## Correspondencia con la evaluación

- Rúbrica 2.2: minibatches, un `optimizer.step()` por batch, parámetros
  configurables que el entrenador usa.
- Rúbrica 3.2: parámetros efectivos, semilla, commit, release DVC, hash del
  manifiesto, clases, estado, métricas por época y checkpoint en MLflow.
- Issue #17 (ML-04): Agent Test (corrida corta recuperada por la API de MLflow),
  runs fallidos en `FAILED`, y DoD "Real training run completed" / "MLflow
  instrumentation verified".

## Reproducir

```bash
docker compose up -d --wait mlflow        # .env con MARIADB_*, MINIO_* y MLFLOW_PORT
cd app
RUN_TRAINING_EVIDENCE=1 MLFLOW_TRACKING_URI=http://127.0.0.1:5050 \
  .venv/bin/python -m pytest -q -s tests/test_classification_training_evidence.py
```

## 1. Corrida real (3 épocas)

`adam`, `batch_size=32`, `max_epochs=3`, `learning_rate=0.001`, `image_size=128`,
`hidden_layers=[128]`, `dropout=0.2`, `seed=42`, ResNet18 ImageNet con `layer4` y la
cabeza entrenables. 469 crops de train, 128 de validation.

```text
MLflow: http://127.0.0.1:5050
[hook] run iniciado experiment_id=1 run_id=a047b9e466bb45e08dcebbc282f91e3e
[log:info] Run a047b9e466bb45e08dcebbc282f91e3e: 469 crops de train y 128 de validation, 3 épocas, batch_size=32
[hook] progreso época 1: {'train_loss': 0.267563266743189, 'train_accuracy': 0.8614072494669509, 'val_loss': 0.44411205500364304, 'val_accuracy': 0.8828125}
[log:info] Época 1/3: train_loss=0.2676 train_acc=0.8614 val_loss=0.4441 val_acc=0.8828
[hook] progreso época 2: {'train_loss': 0.08000676326556945, 'train_accuracy': 0.9594882729211087, 'val_loss': 0.23068933561444283, 'val_accuracy': 0.9296875}
[log:info] Época 2/3: train_loss=0.0800 train_acc=0.9595 val_loss=0.2307 val_acc=0.9297
[hook] progreso época 3: {'train_loss': 0.05942700452197081, 'train_accuracy': 0.9786780383795309, 'val_loss': 0.17234518565237522, 'val_accuracy': 0.953125}
[log:info] Época 3/3: train_loss=0.0594 train_acc=0.9787 val_loss=0.1723 val_acc=0.9531
[log:info] Run a047b9e466bb45e08dcebbc282f91e3e FINISHED; checkpoint en checkpoints/last.pt
duración: 23.6s  pasos del optimizador: 45
checkpoint: runs:/a047b9e466bb45e08dcebbc282f91e3e/checkpoints/last.pt
```

45 pasos = 3 épocas × 15 batches (469 / 32 redondeado hacia arriba). La
`val_accuracy` es de validation: no es la métrica final de test.

## 2. Agent Test: el run recuperado por la API de MLflow (proceso nuevo)

```text
== Agent Test: run recuperado por la API de MLflow (proceso nuevo) ==
status: FINISHED | experimento: dogcat-classifier | run_name: ml04-real-adam-bs32-img128
params: {'architecture': 'resnet18', 'batch_size': '32', 'dropout': '0.2', 'hidden_layers': '128', 'image_size': '128', 'learning_rate': '0.001', 'max_epochs': '3', 'min_delta': '0.0', 'optimizer': 'adam', 'patience': '2', 'pretrained': 'True', 'seed': '42', 'train_drop_last': 'False', 'train_samples': '469', 'trainable': 'layer4', 'validation_samples': '128'}
tag mlflow.source.git.commit: f2ae3329dade0a4dfa18f64e48cedec327056d3a
tag dataset_version: v0.1.1
tag manifest_hash: sha256:178b28bddb1bef5c05975fc7150710658627e31af08a1a12d94b98f9e99cb2b2
tag dvc_images_hash: 951150dd4fb053f4665089fcb37a1c87.dir
tag dvc_annotations_hash: c7cb86ae7ece94ef7b853620e464a4d7.dir
tag crops_sha256: sha256:11cea337fd46877cdddbdc6281b21bd0d7de694ed66fb8cc582d5cab31216f86
tag classes: dog,cat
tag class_map: {"dog": 0, "cat": 1}
tag weights_origin: {"weights": "ResNet18_Weights.IMAGENET1K_V1", "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth", "dataset": "ImageNet-1K", "library": "torchvision"}
tag torch_version: 2.14.0
train_loss: [(1, 0.2676), (2, 0.08), (3, 0.0594)]
train_accuracy: [(1, 0.8614), (2, 0.9595), (3, 0.9787)]
val_loss: [(1, 0.4441), (2, 0.2307), (3, 0.1723)]
val_accuracy: [(1, 0.8828), (2, 0.9297), (3, 0.9531)]
artefactos: ['checkpoints/last.pt']
checkpoint descargado: last.pt 45.0 MB
metadata del checkpoint: {'run_id': 'a047b9e466bb45e08dcebbc282f91e3e', 'dataset_version': 'v0.1.1', 'manifest_hash': 'sha256:178b28bddb1bef5c05975fc7150710658627e31af08a1a12d94b98f9e99cb2b2', 'git_commit': 'f2ae3329dade0a4dfa18f64e48cedec327056d3a', 'epochs': 3}
inferencia img102-ann118 real=dog pred=dog P=(1.000,0.000)
inferencia img103-ann138 real=dog pred=dog P=(0.996,0.004)
inferencia img106-ann134 real=dog pred=cat P=(0.054,0.946)
inferencia img109-ann137 real=dog pred=dog P=(0.998,0.002)

```

El commit registrado (`f2ae332…`) es el commit de la implementación en esta rama.

## 3. Un fallo deja el run en FAILED (servidor real)

Se inyecta una excepción después de la época 1:

```text
[log:error] Run bc25707ea41f4e6d8aa6fb58e5f9b94a terminó en FAILED: fallo inyectado tras la época 1
excepción propagada al worker: fallo inyectado tras la época 1
status en el servidor: FAILED | tag error: RuntimeError: fallo inyectado tras la época 1 | train_loss steps: [1]
```

## 4. Integración con la cola de jobs de APP-03 (PR #38)

Sobre esta rama más `training/queue.py` y `presentation/ml_contracts.py` de
`feat/app-03-training-jobs` (copia temporal, sin subir), con SQLite y MLflow local:
`enqueue` → `claim_next` → `run_training(hooks=JobQueueHooks(queue, job_id))` →
`succeed(checkpoint=result.checkpoint_uri)`.

```text
job: status=succeeded run_id=283c1ec535b94d72a0a15f36b447eddf experiment_id=928195207101951047
  checkpoint=runs:/283c1ec535b94d72a0a15f36b447eddf/checkpoints/last.pt
  progress=epoch=3 max_epochs=3 metrics={'train_loss': 0.10020135343074799, 'train_accuracy': 1.0, 'val_loss': 0.44349631667137146, 'val_accuracy': 1.0} updated_at='2026-10-01T14:04:25Z'
  MLflow: status=FINISHED
  log #1 [info] Run 283c1ec535b94d72a0a15f36b447eddf: 8 crops de train y 4 de validation, 3 épocas, batch_size=4
  log #2 [info] Época 1/3: train_loss=0.6572 train_acc=0.5000 val_loss=0.6583 val_acc=1.0000
  log #3 [info] Época 2/3: train_loss=0.3372 train_acc=0.8750 val_loss=0.5174 val_acc=1.0000
  log #4 [info] Época 3/3: train_loss=0.1002 train_acc=1.0000 val_loss=0.4435 val_acc=1.0000
  log #5 [info] Run 283c1ec535b94d72a0a15f36b447eddf FINISHED; checkpoint en checkpoints/last.pt
job inválido (batch_size=1): status=failed run_id=None error=training_error
```

> Esa fue la primera versión. Tras la revisión del PR #39, `batch_size=1` ya no se
> rechaza: ver la sección 6.

Esta prueba encontró que `TrainingJobQueue.log` recibe `level` solo por nombre.
`JobQueueHooks` lo pasaba por posición; se corrigió (`774d2c4`) y la cola falsa de
los tests ahora usa la misma firma.

## 5. Corrida de evidencia reproducible (2 épocas)

`tests/test_classification_training_evidence.py`, mismo `seed=42`:

```text
train_loss: [(1, 0.2676), (2, 0.08)]
train_accuracy: [(1, 0.8614), (2, 0.9595)]
val_loss: [(1, 0.4441), (2, 0.2307)]
val_accuracy: [(1, 0.8828), (2, 0.9297)]
```

Las dos primeras épocas coinciden con las de la corrida de la sección 1: con la
misma semilla y el mismo entorno, el entrenamiento se repite.

## 6. `batch_size=1` (ajuste de la revisión del PR #39)

El formulario y el API aceptan `batch_size=1`, y la primera versión del loop
rechazaba ese job ya creado. Ahora se entrena con un paso por crop y todas las
BatchNorm con sus estadísticas guardadas (`batchnorm_statistics=frozen`).

Corrida real contra el servidor MLflow (469 crops de train, pesos ImageNet,
`sgd`, `image_size=64`, 1 época):

```text
batch_size=1 real: FINISHED en 8s, 469 pasos (469 crops), batchnorm_statistics=frozen, train_drop_last=False
  métricas: {'train_loss': 0.5887, 'train_accuracy': 0.71, 'val_loss': 0.3929, 'val_accuracy': 0.7969}
```

Por la cola real de APP-03 (`training/queue.py` del PR #38, SQLite, dataset
controlado), incluido el caso más difícil (`image_size=32`, layer4 en 1x1):

```text
job batch_size=1 image_size=32: succeeded | progreso época 2/2 | runs:/754c92a9252a4d138e031a0269805314/checkpoints/last.pt
job batch_size=1 image_size=64: succeeded | progreso época 2/2 | runs:/d9a0320bc9544e1cbb3731dac0ad91d6/checkpoints/last.pt
```
