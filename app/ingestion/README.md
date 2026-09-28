# P2-11 — Modelos Pydantic v2

`ingestion/models.py` valida el COCO crudo del Proyecto 1 antes de que
`analyzers`/`policies` lo usen. `CocoDataset.model_validate(coco)` es la
única puerta de entrada: si pasa, `analyze_small_objects` y el resto de
analizadores pueden seguir recibiendo el `dict` ya validado
(`dataset.model_dump()`) con la garantía de que `bbox` tiene 4 elementos y
que `image_id`/`category_id` existen.

Casos rechazados, cada uno con un error de Pydantic que nombra el campo (no
un `KeyError`/`IndexError`):

- `bbox` con una cantidad de elementos distinta de 4, o con `width`/`height`
  no positivos.
- `annotations[].category_id` que no aparece en `categories`.
- `annotations[].image_id` que no aparece en `images` (huérfana).
- Cualquier campo extra no declarado (`extra="forbid"`), tipos no exactos
  (`strict=True`) o `NaN`/`Infinity` (`allow_inf_nan=False`).
- `images[].id`, `annotations[].id` o `categories[].id` repetidos dentro
  del dataset (P2-22/23/24: el bug real que motivó esto — dos lotes
  anotados por separado reutilizaron los mismos ids — ver
  `ingestion/loader.py`).

`ingestion/loader.py` (P2-22/23/24) es quien junta los
`annotations-lote-*.json` de `data/raw/annotations/` en un solo
`CocoDataset`: como valida el resultado combinado, un choque de ids entre
dos lotes distintos se rechaza ahí mismo, no río abajo en los analizadores.

`policies/models.py` (`QualityPolicy`) y `storage/settings.py` (`Settings`)
siguen el mismo principio para `quality.yaml` y las variables de entorno:
`quality.yaml` entra a `Settings` como fuente adicional de
`pydantic-settings`, así que un umbral roto o una variable de entorno
faltante se rechazan igual, antes de llegar a la compuerta o a
`storage/db.py`/`storage/object_store.py`. `splits/models.py`
(`SplitsConfig`) valida `splits/splits.yaml` (proporciones que suman 1.0 y
la semilla) de forma independiente, con el mismo patrón de carga que
`policies/small_objects.py`.

Desde `app/`, con las dependencias declaradas disponibles:

```bash
uv run pytest tests/test_ingestion_models.py tests/test_policies_models.py \
    tests/test_splits_models.py tests/test_storage_settings.py
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
