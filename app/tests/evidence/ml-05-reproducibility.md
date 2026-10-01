# Evidencia ML-05: semillas, orden de muestras y augmentation reproducibles

Ejecución: 2026-10-01. Rama `feat/ml-05-seeds-reproducibility` (sobre ML-04).
Servidor MLflow de OPS-03 en `http://127.0.0.1:5050` (`docker compose up mlflow`),
manifiesto real `reports/releases/v0.1.1/manifest.json` y `data/crops` de DVC.

## Correspondencia con la evaluación

- Rúbrica 2.3: semillas de partición, shuffle/DataLoader, augmentation e
  inicialización de pesos, versiones de librerías y entorno; transforms
  aleatorios solo en train; corrida repetida con igual semilla y mismo orden;
  operaciones no deterministas documentadas.
- Issue #18 (ML-05): Agent Test (dos corridas cortas con la misma semilla y
  comparación del orden de muestras).

## Reproducir

```bash
docker compose up -d --wait mlflow
cd app
RUN_REPRODUCIBILITY_EVIDENCE=1 MLFLOW_TRACKING_URI=http://127.0.0.1:5050 \
  .venv/bin/python -m pytest -q -s tests/test_classification_reproducibility_evidence.py
```

## 1. Agent Test: dos corridas reales con la misma semilla (y una con otra)

`adam`, `batch_size=32`, `max_epochs=2`, `image_size=96`, `hidden_layers=[64]`,
`dropout=0.2`, pesos ImageNet; 469 crops de train y 128 de validation. El orden se
lee del artefacto `reproducibility/sample_order.json` descargado por la API de MLflow.

```text
a715a27f0c364c12a7b6f3d79cb84bad ml05-seed42-a: train_order_sha256=a4d41607466a2348… época 1 empieza ['img484-ann479', 'img43-ann69', 'img461-ann450']
  métricas: [{'train_loss': 0.3047362463687783, 'train_accuracy': 0.8464818763326226, 'val_loss': 0.7291560396552086, 'val_accuracy': 0.8515625}, {'train_loss': 0.12261074467270232, 'train_accuracy': 0.9488272921108742, 'val_loss': 0.4798906072974205, 'val_accuracy': 0.8671875}]
3ebe17d7b5f34f2fbab57ed07610ebf1 ml05-seed42-b: train_order_sha256=a4d41607466a2348… época 1 empieza ['img484-ann479', 'img43-ann69', 'img461-ann450']
  métricas: [{'train_loss': 0.3047362463687783, 'train_accuracy': 0.8464818763326226, 'val_loss': 0.7291560396552086, 'val_accuracy': 0.8515625}, {'train_loss': 0.12261074467270232, 'train_accuracy': 0.9488272921108742, 'val_loss': 0.4798906072974205, 'val_accuracy': 0.8671875}]
8a7d2c95ec7545638dbeb0e178a02059 ml05-seed7: train_order_sha256=54fc5f60aebf892d… época 1 empieza ['img270-ann311', 'img493-ann488', 'img620-ann616']
  métricas: [{'train_loss': 0.32353682711180337, 'train_accuracy': 0.8571428571428571, 'val_loss': 0.7175259590148926, 'val_accuracy': 0.8828125}, {'train_loss': 0.1230319537627481, 'train_accuracy': 0.9488272921108742, 'val_loss': 0.24270504340529442, 'val_accuracy': 0.921875}]
semillas: {'seed': '42', 'seed_augmentation': '42', 'seed_dataloader': '42', 'seed_split': '42', 'seed_weight_init': '42'}
entorno: {'python_version': '3.12.6', 'torch_version': '2.14.0', 'torchvision_version': '0.29.0', 'numpy_version': '2.5.3', 'platform': 'macOS-15.7.1-arm64-arm-64bit', 'torch_num_threads': '4'}
```

- `seed=42` dos veces: mismo `train_order_sha256`, mismo orden de los 469 crops
  en las dos épocas y métricas idénticas al último decimal.
- `seed=7`: otro orden y otras métricas.
- El orden cambia entre la época 1 y la 2 (se baraja cada época).

Las métricas difieren de la evidencia de ML-04 con la misma configuración porque
ML-05 cambió de dónde sale la aleatoriedad de la augmentation (semilla por
muestra en lugar del generador global); con ML-05, cada valor es reproducible.

## 2. Operaciones no deterministas medidas

Dataset controlado de 12 crops de train, `image_size=64`, 2 épocas
(`/tmp`, MLflow local):

```text
hilos 4 vs 1: orden igual=True | métricas idénticas=True
num_workers 0 vs 2: orden igual = True | tensores aumentados iguales = True
SIN augmentation_seed, num_workers 0 vs 2: tensores iguales = False
use_deterministic_algorithms(True) en CPU: corre sin error; métricas iguales al modo normal = True
```

La tercera línea es el motivo de la semilla por muestra: con el generador global
(comportamiento de ML-02), agregar workers al DataLoader cambiaba la
augmentation.

## 3. Protección de validation y test

`tests/test_classification_reproducibility.py`:

- `random_transform_names` solo encuentra `RandomResizedCrop`,
  `RandomHorizontalFlip` y `ColorJitter` en train; en validation, test e
  inferencia, nada.
- Validation y test dan el mismo tensor con o sin `augmentation_seed` y en
  cualquier época.
- `run_training` solo carga train y validation, y rechaza (sin crear run) una
  validation con transforms aleatorios.
