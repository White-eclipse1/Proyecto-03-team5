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
