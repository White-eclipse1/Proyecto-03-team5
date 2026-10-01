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
- existe el PNG de cada crop.

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
