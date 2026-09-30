# ML-02 — Dataset y preprocesamiento del clasificador dog/cat

PyTorch 2.14 + torchvision 0.29 (CPU), fijados en `app/uv.lock`.

## Qué consume

El Dataset no lee una carpeta de imágenes: cruza una partición del manifiesto
P3 70/20/10 (OPS-02) con `reports/crops.json` (ML-01) por `crop_id` y abre solo
`data/crops/<crop_path>` de esos crops.

Formato mínimo del manifiesto que espera `classification/manifest.py` (los demás
campos de OPS-02, como hash, versión o conteos, se aceptan y se ignoran):

```json
{
  "dataset_version": "v0.1.1",
  "records": [
    {
      "crop_id": "img40-ann65",
      "source_image_id": 40,
      "duplicate_group": "g40",
      "class": "dog",
      "split": "train"
    }
  ]
}
```

- `crop_id`: el de `reports/crops.json`, `img<image_id>-ann<annotation_id>`.
- `source_image_id`: el `image_id` COCO de la imagen original.
- `class`: `dog` o `cat`.
- `split`: `train`, `validation` o `test`.
- `duplicate_group`: texto o entero.

Al construirse, el Dataset falla si el manifiesto es de otro release, si un
`crop_id` no está entre los crops aceptados, si `class` o `source_image_id` no
coinciden con `crops.json`, o si falta el PNG del crop.

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
