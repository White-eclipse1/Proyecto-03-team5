# ML-02 — Dataset y preprocesamiento del clasificador dog/cat

PyTorch 2.14 + torchvision 0.29 (CPU), fijados en `app/uv.lock`.

```python
from classification.dataset import build_dataloader, load_split

train = load_split(
    REPO / "reports/releases/v0.1.1/manifest.json",
    REPO / "reports/crops.json",
    crops_dir=REPO / "data/crops",
    split="train",
    params=params,  # TrainingParams: image_size, batch_size, seed...
)
loader = build_dataloader(train, batch_size=params.batch_size, seed=params.seed)
```

## Qué consume

El Dataset no lee una carpeta de imágenes: cruza una partición del manifiesto
P3 de OPS-02 (`reports/releases/v0.1.1/manifest.json`, contrato
`manifests.models.P3Manifest`) con `reports/crops.json` (ML-01) por `crop_id` y
abre solo `data/crops/<crop_path>` de esos crops.

Manifiesto real `p3-v1` (seed 42), 668 crops de 600 imágenes:

| Split | Crops | dog | cat |
|-------|-------|-----|-----|
| train | 469 | 221 | 248 |
| validation | 128 | 61 | 67 |
| test | 71 | 43 | 28 |

Antes de entregar una sola muestra se comprueba que:

- `manifest_hash` coincide con el contenido del archivo (fórmula de OPS-02), así
  que un manifiesto editado a mano se rechaza;
- no hay fuga de `crop_id`, `source_image_id` ni `duplicate_group` entre
  particiones (`manifests.validation.validate_no_leakage`);
- `provenance.crops_sha256` es el sha256 del `crops.json` que se usa
  (`load_split`);
- el manifiesto y los crops son del mismo release, cada `crop_id` existe, no se
  repite, y su `class` y `source_image_id` coinciden con `crops.json`;
- existe el PNG de cada crop y su sha256 es el que `crops.json` registró al
  extraerlo (`CropRecord.sha256`): un PNG alterado o sustituido se rechaza al
  construir el Dataset, antes de entrenar. `dataset[i]` vuelve a verificar el
  hash de los bytes que decodifica (los 668 PNG reales se verifican en ~0.2 s).

## Muestras

`dataset[i]` devuelve `image` (`3 x image_size x image_size`, float32
normalizado), `label` (`dog=0`, `cat=1`, de `crops.classes.CLASS_TO_INDEX`),
`crop_id`, `source_image_id` y `class_name`. `dataset.sample(i)` da los mismos
metadatos sin abrir la imagen.

## Preprocesamiento

| Uso | Transform | Aleatorio |
|-----|-----------|-----------|
| train | `RandomResizedCrop(scale=0.8–1)`, `RandomHorizontalFlip`, `ColorJitter(0.2)`, float32, normalización ImageNet | sí |
| validation, test | `Resize(image_size)`, float32, normalización ImageNet | no |
| inference | `preprocess_image` = el mismo `eval_transform` de validation/test, sobre RGB | no |

`image_size` sale de la configuración de entrenamiento (`TrainingParams.image_size`,
múltiplo de 32 entre 32 y 1024) vía `load_split(..., params=...)`.

## DataLoader

`build_dataloader(dataset, batch_size=..., seed=...)` baraja solo train, con un
`torch.Generator` sembrado: la misma semilla da el mismo orden. Validation y
test conservan el orden del manifiesto. La aumentación de train usa el RNG
global de torch: el entrenamiento (ML-03) debe llamar `torch.manual_seed(seed)`
para que también sea reproducible.

Tests: `uv run pytest tests/test_classification_dataset.py`.

## ML-03 — Modelo: ResNet18 preentrenada + cabeza configurable

`classification/model.py`. El modelo **no** se usa tal como viene de torchvision:
se reemplaza su capa final por una cabeza propia de 2 clases y se entrena
(fine-tuning) con los crops del manifiesto P3.

### Arquitectura

```text
imagen 3 x image_size x image_size (preprocesamiento de ML-02, normalización ImageNet)
  -> ResNet18 backbone: conv1 -> bn1 -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool
  -> 512 características (la fc original de 1000 clases ImageNet se reemplaza por Identity)
  -> cabeza: [Linear(in, h) -> ReLU -> Dropout(dropout)] por cada h en hidden_layers
  -> Linear(…, 2) -> logits [dog, cat]
```

Ejemplo con `hidden_layers=[256, 64]`, `dropout=0.3`:
`Linear(512,256) ReLU Dropout(0.3) Linear(256,64) ReLU Dropout(0.3) Linear(64,2)`.
Con `hidden_layers=[]` la cabeza es `Dropout(dropout) Linear(512,2)`.

| Parámetro | Origen | Validación |
|-----------|--------|------------|
| `image_size` | `TrainingParams.image_size` | múltiplo de 32 entre 32 y 1024; `forward` rechaza otro tamaño |
| `hidden_layers` | `TrainingParams.hidden_layers` | 0 a 5 capas de 1 a 4096 unidades |
| `dropout` | `TrainingParams.dropout` | `0 <= dropout < 1` |
| `pretrained` | `ModelConfig` (por defecto `True`) | — |
| `trainable` | `ModelConfig` (por defecto `layer4`) | `head`, `layer4` o `all` |

`ModelConfig.from_params(params)` toma los tres primeros de la configuración de
entrenamiento; una configuración inválida se rechaza antes de construir la red.

Clases: `CLASS_MAP = {"dog": 0, "cat": 1}` (`crops.classes.CLASS_TO_INDEX`). La
salida tiene exactamente 2 logits; `predict_proba` devuelve el softmax.

### Pesos iniciales

Parte de **pesos preentrenados**, no de cero:

- `torchvision.models.ResNet18_Weights.IMAGENET1K_V1` (torchvision 0.29),
  entrenados por PyTorch en ImageNet-1K (1000 clases).
- URL: `https://download.pytorch.org/models/resnet18-f37072fd.pth`. torchvision
  comprueba al descargar que el sha256 empiece por `f37072fd`.
- La capa `fc` de ImageNet no se usa; la cabeza de 2 clases empieza con pesos
  aleatorios (inicialización por defecto de `nn.Linear`).

Con `pretrained=False` la misma arquitectura arranca con pesos aleatorios, y el
checkpoint registra `weights_origin: null`.

### Capas entrenables y congeladas

| `trainable` | Entrenables | Congeladas | Parámetros entrenables / congelados (`hidden_layers=[256]`) |
|-------------|-------------|------------|-------------------------------------|
| `head` | cabeza | conv1, bn1, layer1–layer4 | 131,842 / 11,176,512 |
| `layer4` (por defecto) | layer4 + cabeza | conv1, bn1, layer1–layer3 | 8,525,570 / 2,782,784 |
| `all` | todo | — | 11,308,354 / 0 |

Los bloques congelados tienen `requires_grad=False` y sus BatchNorm permanecen
en modo eval incluso con `model.train()`, así sus estadísticas de ImageNet no
cambian. `model.trainable_summary()` da este desglose para registrarlo en
MLflow (ML-04).

### Checkpoint

`save_checkpoint(model, path, metadata=...)` guarda con `torch.save`: pesos
(`state_dict`), `architecture`, `config`, `class_map`, `preprocessing`
(`image_size`, resize cuadrado, media/desviación de ImageNet), `weights_origin`
y `metadata` libre (run ID, manifest_hash, etc.). `load_checkpoint(path)`
reconstruye la red **sin descargar** pesos, rechaza otra arquitectura u otro
`class_map`, carga con `weights_only=True` y deja el modelo en modo eval.

Tests: `uv run pytest tests/test_classification_model.py`. Evidencia con pesos y
crops reales: [`tests/evidence/ml-03-model.md`](../tests/evidence/ml-03-model.md).

## ML-04 — Loop de entrenamiento y MLflow

`classification/training.py` → `run_training(...)`, lo que ejecuta el worker de
OPS-04 por cada job de la cola de APP-03.

```python
from classification.training import DataPaths, JobQueueHooks, run_training
from tracking.client import tracking_client  # OPS-03, usa MLFLOW_TRACKING_URI

result = run_training(
    job.params,  # TrainingParams del job
    dataset_version=job.dataset_version,
    manifest_hash=job.manifest_hash,
    data=DataPaths.for_release(job.dataset_version),
    client=tracking_client(),
    hooks=JobQueueHooks(queue, job.job_id),  # start / report_progress / log de la cola
)
queue.succeed(job.job_id, checkpoint=result.checkpoint_uri)  # runs:/<run_id>/checkpoints/best.pt
```

Si `run_training` lanza una excepción, el run ya quedó `FAILED` en MLflow (o no
se creó, si el job no coincide con el manifiesto). El worker solo llama
`queue.fail(job.job_id, ContractError(code="training_error", ...))`.

### Qué hace

1. Carga train y validation con el Dataset de ML-02. Falla **antes** de crear el
   run si `dataset_version` o `manifest_hash` del job no son los del manifiesto.
2. `torch.manual_seed(seed)` y construye el modelo de ML-03 (`image_size`,
   `hidden_layers`, `dropout`) y el optimizador configurado sobre los parámetros
   entrenables: `adam` → `Adam`, `adamw` → `AdamW`, `sgd` → `SGD(momentum=0.9)`,
   con `learning_rate`.
3. Por época, un `optimizer.step()` por minibatch del DataLoader de train
   (`batch_size`), y luego la evaluación sin gradiente en validation.
   - `batch_size=1` (lo aceptan el formulario y el API) se entrena con un paso por
     crop y todas las BatchNorm usando sus estadísticas guardadas
     (`batchnorm_statistics=frozen`; sus pesos gamma/beta se siguen entrenando):
     las estadísticas de una sola imagen son ruido, y con `image_size=32` BatchNorm
     no puede normalizar un único valor por canal.
   - Con `batch_size > 1`, BatchNorm usa las estadísticas del batch
     (`batchnorm_statistics=batch`) y, si el último batch de train tendría una sola
     muestra, se descarta (`train_drop_last=True`) por la misma razón.
4. Early stopping y mejor checkpoint (ML-06, ver abajo): sube `checkpoints/best.pt`
   con los pesos de la mejor época y las curvas, y cierra el run `FINISHED`.

### Qué queda en MLflow (experimento `dogcat-classifier`)

| Tipo | Claves |
|------|--------|
| Parámetros efectivos (texto, como `ExperimentRun.params`) | `optimizer`, `batch_size`, `max_epochs`, `learning_rate`, `image_size`, `hidden_layers` (`"128,64"`), `dropout`, `seed`, `patience`, `min_delta`, `architecture`, `pretrained`, `trainable`, `train_samples`, `validation_samples`, `train_drop_last`, `batchnorm_statistics` |
| Métricas por época (`step` = época) | `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy` |
| Etiquetas de procedencia | `mlflow.source.git.commit`, `git_commit`, `dataset_version`, `manifest_hash`, `manifest_version`, `dvc_images_hash`, `dvc_annotations_hash`, `quality_report`, `crops_sha256`, `classes` (`dog,cat`), `class_map`, `weights_origin`, `trainable_summary`, `torch_version`, `torchvision_version` |
| Artefactos | `checkpoints/best.pt` (`save_checkpoint` de ML-03, con `run_id`, release, `manifest_hash`, commit, `best_epoch` y `stopped_epoch` en `metadata`), `curves/training_curves.png`, `curves/history.json`, `reproducibility/sample_order.json` |
| Fallo | estado `FAILED` (`KILLED` si se interrumpe) y etiqueta `error` |

### Requisitos del proceso que entrena (worker)

- `MLFLOW_TRACKING_URI`: en Compose, `http://mlflow:5000` (OPS-03).
- `GIT_COMMIT` con el SHA de 40 caracteres si el contenedor no tiene `.git`
  (por ejemplo, como argumento de build). Fuera de Docker se usa
  `git rev-parse HEAD`. Sin commit válido, el job falla antes de crear el run.
- `reports/releases/<version>/manifest.json`, `reports/crops.json` y `data/crops`
  (de DVC) en las rutas de `DataPaths.for_release`, o rutas explícitas en
  `DataPaths(...)`.
- Red para descargar los pesos ImageNet de torchvision la primera vez (~45 MB),
  o la caché de torch hub (`TORCH_HOME`) en un volumen.
- CPU: 3 épocas con `image_size=128` y `batch_size=32` tardan ~25 s en un Mac M.

Tests: `uv run pytest tests/test_classification_training.py` (dataset controlado
y MLflow local, sin red). Evidencia con el servidor real:
[`tests/evidence/ml-04-training.md`](../tests/evidence/ml-04-training.md).

## ML-05 — Semillas y augmentation controlada

### Semillas registradas en cada run (parámetros de MLflow)

| Parámetro | Valor | Qué controla |
|-----------|-------|--------------|
| `seed_split` | `seed` del manifiesto de OPS-02 (42 en `p3-v1`) | Asignación de crops a train/validation/test (70/20/10) |
| `seed_dataloader` | `TrainingParams.seed` | `torch.Generator` del DataLoader: orden de train en cada época |
| `seed_augmentation` | `TrainingParams.seed` | Augmentation de train: cada muestra usa la semilla de `sha256(seed:época:índice)` |
| `seed_weight_init` | `TrainingParams.seed` | `torch.manual_seed` antes de construir el modelo: pesos iniciales de la cabeza y secuencia de dropout |

Cada semilla alimenta su propio generador, así que comparten valor sin compartir
secuencia. La augmentation corre dentro de `torch.random.fork_rng`: no depende
del orden de lectura ni de los workers, y no consume el generador global que usa
el dropout.

El orden real en que entró cada crop de train, por época, queda en el artefacto
`reproducibility/sample_order.json` y en la etiqueta `train_order_sha256`: dos
runs vieron el mismo orden si esa etiqueta coincide.

Versiones y entorno (etiquetas, `environment_tags()`): `python_version`,
`torch_version`, `torchvision_version`, `numpy_version`, `pillow_version`,
`mlflow_version`, `platform`, `torch_num_threads`,
`torch_deterministic_algorithms` y `cuda_available`. Las dependencias exactas
están fijadas en `app/uv.lock`.

### Validation, test e inferencia

- Train: `RandomResizedCrop`, `RandomHorizontalFlip` y `ColorJitter`.
- Validation, test e inferencia: `eval_transform`, sin pasos aleatorios
  (`random_transform_names(...) == []`).
- `run_training` solo carga train y validation (nunca test). Si validation tuviera
  un transform aleatorio, se niega a entrenar antes de crear el run.

### Operaciones no deterministas conocidas

Comprobado el 2026-10-01 en macOS arm64, CPU, torch 2.14.0
([evidencia](../tests/evidence/ml-05-reproducibility.md)):

| Fuente | Estado en este proyecto |
|--------|-------------------------|
| Hilos de CPU (`torch_num_threads`) | 4 hilos y 1 hilo dieron métricas idénticas. Otro CPU o BLAS puede cambiar los últimos decimales: por eso se registra la plataforma |
| Workers del DataLoader | Con `seed_augmentation`, `num_workers=0` y `2` dan el mismo orden y los mismos tensores. Sin semilla por muestra (como en ML-02) la augmentation cambiaba con los workers |
| `torch.use_deterministic_algorithms(True)` | En CPU corre sin error y da las mismas métricas que el modo normal; no se activa por defecto |
| GPU / cuDNN | No se usa GPU (`cuda_available=False`). En GPU, algunas convoluciones y `scatter_add` no son deterministas: habría que activar `torch.backends.cudnn.deterministic = True`, `benchmark = False` y `use_deterministic_algorithms(True)` |
| Pesos ImageNet | Fijos: torchvision verifica el hash `f37072fd` al descargarlos |
| Librerías | `uv.lock` fija versiones exactas; otra versión de torch/torchvision/Pillow puede cambiar el redimensionado o los kernels |
| Tiempo y orden de logs | Los `timestamp` de MLflow y los `run_id` cambian en cada corrida; no afectan el entrenamiento |

Tests: `uv run pytest tests/test_classification_reproducibility.py`.

## ML-06 — Early stopping y mejor checkpoint

`classification/early_stopping.py` (lógica pura) y su uso en `run_training`.

| Elemento | Definición |
|----------|------------|
| Métrica vigilada (predeclarada) | `val_loss`, a minimizar (`MONITOR`, `MONITOR_MODE`) |
| Mejor época (`best_epoch`, pesos que se restauran) | La de **menor** `val_loss` finito, aunque haya bajado menos que `min_delta` |
| Reinicio de la paciencia | Solo si `val_loss < menor anterior - min_delta`. Un valor igual o una mejora menor que `min_delta` no la reinicia |
| Pérdidas no finitas | `NaN`, `+inf` y `-inf` nunca son la mejor época y cuentan como época sin mejora; si todas lo son, el run falla sin checkpoint |
| Parada | Tras `patience` épocas seguidas sin una mejora mayor que `min_delta`; esa época es `stopped_epoch` |
| `patience`, `min_delta` | Vienen de `TrainingParams` (formulario y API) |
| Pesos finales | Los de `best_epoch`, copiados en memoria cuando mejoró y **restaurados** al terminar, no los de la última época |

Se usa `val_loss` y no `val_accuracy` porque con 128 crops de validation la
accuracy se mueve en saltos de 1/128 y empata con facilidad. La pérdida es
continua.

### Qué queda en MLflow

| Tipo | Claves |
|------|--------|
| Parámetros | `early_stopping_monitor=val_loss`, `early_stopping_mode=min`, `patience`, `min_delta` |
| Métricas de resumen | `best_epoch`, `best_val_loss`, `epochs_completed` y, si hubo parada temprana, `stopped_epoch` |
| Etiqueta | `early_stopped` (`True` o `False`) |
| Artefactos | `checkpoints/best.pt` (pesos restaurados; `metadata` con `best_epoch`, `best_val_loss`, `stopped_epoch` y `epochs_completed`), `curves/training_curves.png` (loss y accuracy de train y validation por época, con líneas en `best_epoch` y en la parada), `curves/history.json` (los mismos valores que las métricas por época de MLflow) |

`TrainingResult` devuelve `best_epoch`, `stopped_epoch` (o `None`) y
`checkpoint_uri = runs:/<run_id>/checkpoints/best.pt`, que el worker pasa a
`queue.succeed`.

Tests: `uv run pytest tests/test_classification_early_stopping.py`. Las
secuencias artificiales de `val_loss` prueban la lógica, y una secuencia
inyectada en `run_training` comprueba que el checkpoint final son los pesos de
la mejor época. Evidencia con datos reales:
[`tests/evidence/ml-06-early-stopping.md`](../tests/evidence/ml-06-early-stopping.md).

## ML-07 — Matriz de experimentos y 10+ corridas en MLflow

`classification/ml07_matrix.yaml` define 12 corridas sobre el manifiesto `v0.1.1`.
Todas usan una `base` común y cambian uno o dos parámetros a la vez. Los siete
parámetros de la rúbrica toman al menos dos valores:

| Parámetro | Valores en la matriz |
|-----------|----------------------|
| `optimizer` | `adam`, `adamw`, `sgd` |
| `batch_size` | 16, 32, 64 |
| `max_epochs` | 6, 10, 15 |
| `learning_rate` | 0.0003, 0.001, 0.01 |
| `image_size` | 96, 128, 160, 224 |
| `hidden_layers` | `[]`, `[128]`, `[256]`, `[256, 64]` |
| `dropout` | 0.0, 0.2, 0.3, 0.5 |

`seed=42`, early stopping (`val_loss`, `patience=3`, `min_delta=0`), clases y
manifiesto son iguales en todas, así las corridas son comparables.
`load_matrix` rechaza una matriz donde algún parámetro no varíe, con
configuraciones duplicadas o con valores fuera de `TrainingParams`.

```bash
docker compose up -d --wait mlflow            # MLflow de OPS-03
cd app
uv run python -m classification.experiments run --matrix classification/ml07_matrix.yaml
uv run python -m classification.experiments report --matrix classification/ml07_matrix.yaml \
  --out ../reports/experiments/ml07_runs.json
```

- `run` entrena cada entrada con `run_training` y etiqueta el run con
  `experiment_matrix=ml07-v1` y `matrix_entry=<nombre>`. Si se interrumpe, al
  volver a correrlo salta las entradas que ya tienen un run `FINISHED`. Se niega
  a correr con cambios sin commit, porque el `git_commit` registrado debe ser el
  código que entrenó.
- `report` consulta solo la API de MLflow y comprueba sobre los runs `FINISHED`
  de la matriz:
  - al menos 10 corridas, los siete parámetros variando y sin configuraciones
    duplicadas;
  - mismo manifiesto, release y clases;
  - semilla, commit, release DVC, métricas por época y `checkpoints/best.pt`.

  Escribe la lista de run IDs y sus métricas de validation en
  `reports/experiments/ml07_runs.json` y termina con código ≠ 0 si falla algún
  criterio.

### Compartir las corridas: snapshot de MLflow

El MLflow de OPS-03 vive en los volúmenes Docker de cada máquina. Para que otro
clon (el de la evaluación o el de un compañero) vea **los mismos run IDs**,
`tracking/snapshot.py` crea un snapshot versionado con DVC en
`data/mlflow-snapshot`:

| Archivo | Contenido |
|---------|-----------|
| `mlflow.sql` | Volcado de la base `mlflow` de MariaDB (`mariadb-dump` dentro del contenedor; la contraseña nunca sale del contenedor) |
| `artifacts/<run_id>/...` | Artefactos de los runs `FINISHED` de la matriz: `checkpoints/best.pt`, `curves/*`, `reproducibility/sample_order.json` |
| `snapshot.json` | Tamaño y sha256 de cada artefacto |

```bash
# Crear o actualizar (con el MLflow que tiene las corridas):
uv run python -m tracking.snapshot create
cd .. && dvc add data/mlflow-snapshot && dvc push -r prod data/mlflow-snapshot.dvc

# Restaurar en otro clon (MLflow levantado):
dvc pull -r prod data/mlflow-snapshot.dvc
cd app && uv run python -m tracking.snapshot restore     # carga, reinicia MLflow, sube y verifica
uv run python -m tracking.snapshot verify --deep          # solo verificar
```

`restore` **reemplaza** la base `mlflow` del MariaDB local por la del snapshot.
Las corridas que solo existían en ese MLflow se pierden, así que úsalo en un
stack nuevo o respalda antes. `--compose` permite apuntar a otro proyecto de
Compose (por ejemplo, `--compose "docker compose -p otro"`).

Evidencia: [`tests/evidence/ml-07-experiments.md`](../tests/evidence/ml-07-experiments.md).

## ML-08 — Selección y congelamiento del candidato (solo validation)

`classification/selection.py` con la política predeclarada en
`classification/selection_policy.yaml`:

| Regla | Valor |
|-------|-------|
| Candidatos | Runs `FINISHED` de la matriz `ml07-v1` (ML-07) |
| Métrica de selección | **menor `best_val_loss`**, la de validation que vigila el early stopping (ML-06) y corresponde a `checkpoints/best.pt` |
| Desempates | Mayor `val_accuracy` en la mejor época; después, el nombre de la entrada |
| Test | No participa. La política rechaza cualquier métrica que no sea de validation, y la selección se niega si algún run de la matriz ya tiene una métrica de test |

```bash
uv run python -m classification.selection select   # ranking, sin congelar
uv run python -m classification.selection freeze   # elige y congela
uv run python -m classification.selection check    # congelamiento anterior a todo test
```

`freeze` escribe `reports/candidates/ml08_candidate.json` (contrato
`CandidateSelection`) con `run_id`, `checkpoint` (`runs:/<run_id>/checkpoints/best.pt`)
y su sha256, `dataset_version`, `manifest_hash`, el commit que entrenó y el que
seleccionó, `frozen_at` (UTC) y el ranking completo. Además marca el run en
MLflow con `candidate=true` y `candidate_frozen_at`. Son los datos que pueden
leer Experiments (APP-04), Evaluation (APP-05) y ML-09.

Reglas del congelamiento:

- Congelar otra vez el mismo run no cambia nada (conserva `frozen_at`).
- Cambiar a otro candidato exige `replace=True`. El run anterior queda
  `candidate=false`, sin `candidate_frozen_at` y con `candidate_replaced_by` y
  `candidate_replaced_at`: en MLflow siempre hay un solo run con `candidate=true`.
- Si existe **cualquier** evaluación de test (`reports/evaluations/**/*.json` con
  `"split": "test"`), no se puede congelar por primera vez ni cambiar el candidato.

ML-09 debe llamar `require_frozen_candidate()` antes de evaluar test y guardar
sus evaluaciones en `reports/evaluations/` con `created_at`.
`early_test_evaluations()` demuestra que todas son posteriores a `frozen_at`.

**Candidato congelado:** `r03-sgd` (`bb448230424146349a969253d30db43b`), `best_val_loss` =
0.0406, congelado en `2026-10-02T00:37:26.037627Z` sin ninguna evaluación de test
previa. Evidencia: [`tests/evidence/ml-08-candidate.md`](../tests/evidence/ml-08-candidate.md).

Tests: `uv run pytest tests/test_classification_selection.py`.

## ML-09 — Evaluación final sobre el test congelado

`classification/evaluation.py`. Evalúa **una vez** el candidato congelado de
ML-08 sobre el split `test`.

1. Exige `reports/candidates/ml08_candidate.json` y verifica que el checkpoint
   descargado de MLflow tenga el `checkpoint_sha256` congelado y que el
   manifiesto sea el del candidato.
2. Solo carga `test`, con el `image_size` del checkpoint. Se niega si el
   preprocesamiento tuviera pasos aleatorios. Modo eval, sin gradiente ni
   optimizador.
3. Escribe en `reports/evaluations/test/`:
   - `<id>.json`: contrato `Evaluation`, con la matriz (filas reales, columnas
     predichas), las métricas por clase y las predicciones;
   - `<id>.predictions.csv`: `crop_id`, `image_id`, `annotation_id`, clases,
     `p_dog`, `p_cat` y acierto.
4. Registra en el run del candidato `test_accuracy`, `test_f1_macro`,
   `test_correct`, `test_total` y `test_precision_*`, `test_recall_*`,
   `test_f1_*`, `test_support_*` por clase; los tags `test_accuracy_meets_target`
   (`aciertos / total` contra 0.85 **con enteros**: `aciertos·20 >= 17·total`, sin
   dividir ni redondear), `test_target_accuracy` y
   `test_evaluation_id`; y los dos archivos como artefactos.

```bash
uv run python -m classification.evaluation evaluate  # una sola vez
uv run python -m classification.evaluation audit     # re-infiere y compara cada muestra, sin escribir
uv run python -m classification.evaluation verify    # CSV vs MLflow (Agent Test)
```

**Si MLflow falla a mitad del registro.** `evaluate` escribe primero el JSON y el
CSV y después registra en MLflow. Si MLflow se cae o se corta la red en ese paso,
`evaluate` termina con un error que dice que la evaluación quedó en disco, y no se
puede repetir porque el test se evalúa una sola vez. Para completar MLflow **sin
volver a inferir**:

```bash
uv run python -m classification.evaluation register  # registra la evaluación que está en disco
uv run python -m classification.evaluation verify    # debe dar OK en todas las métricas
```

`register` exige que la evaluación sea del candidato congelado (run, checkpoint,
manifiesto y release) y que el CSV coincida con el JSON muestra por muestra. Si el
run ya tiene otra evaluación de test registrada, no la reemplaza. Repetirlo no cambia
las métricas ni los tags.

**Resultado:** `r03-sgd` obtiene accuracy = 68/71 = **0.9577** (≥ 0.85) y F1
macro = 0.9548 en test. El recall de `cat` es 0.893 (3 gatos predichos como
perro). Evidencia: [`tests/evidence/ml-09-test-evaluation.md`](../tests/evidence/ml-09-test-evaluation.md).

La auditoría compara el **registro completo de cada muestra**: en el JSON, por
`annotation_id`, los ids, la clase real, la clase predicha y las probabilidades; en
el CSV, la fila entera por `crop_id`. Además compara la cabecera, la matriz y las
métricas. Que cuadren la matriz y las métricas no basta: intercambiar las etiquetas
reales de dos muestras con la misma predicción deja ambas iguales.

Tests: `uv run pytest tests/test_classification_evaluation.py`.
