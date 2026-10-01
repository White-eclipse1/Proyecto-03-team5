# Evidencia ML-03: ResNet18 preentrenada se entrena con crops reales

Ejecución: 2026-09-30. Rama `feat/ml-03-cnn-classifier` (sobre ML-02).
Pesos: `ResNet18_Weights.IMAGENET1K_V1`, descargados de
`https://download.pytorch.org/models/resnet18-f37072fd.pth` (sha256 local
`f37072fd47e89c5e…`). Datos: manifiesto P3 `reports/releases/v0.1.1/manifest.json`
y `data/crops` de DVC.

## Correspondencia con la evaluación

- Rúbrica 2.1: CNN documentada, salida de 2 clases, origen de los pesos
  preentrenados, capas entrenables, y pesos que cambian con un entrenamiento corto
  (inferencia antes y después).
- Issue #5 (ML-03): Agent Test (pesos antes y después de varios pasos del
  optimizador) y DoD "Model actually trains" / "Save/load works".

## Reproducir

Requiere `dvc pull -r prod crops` (o `dvc repro crops`) y red para descargar los
pesos la primera vez (~45 MB, se guardan en la caché de torch hub).

```bash
cd app
RUN_MODEL_EVIDENCE=1 .venv/bin/python -m pytest -q -s tests/test_classification_model_evidence.py
```

Sin `RUN_MODEL_EVIDENCE=1` se omite. Los 37 tests de
`tests/test_classification_model.py` cubren lo mismo con pesos aleatorios y
tensores sintéticos, y corren en CI sin red.

## Salida

Configuración: `image_size=128`, `hidden_layers=[128]`, `dropout=0.2`,
`trainable=layer4`, Adam `lr=0.001`, `batch_size=16`, seed 42; 5 pasos del
optimizador con batches reales de train del DataLoader de ML-02.

```text
{'trainable_blocks': ['layer4', 'head'], 'frozen_blocks': ['conv1', 'bn1', 'layer1', 'layer2', 'layer3'], 'trainable_params': 8459650, 'frozen_params': 2782784}
step 1: loss=0.8457 crops=['img484-ann479', 'img43-ann69']...
step 2: loss=0.3387 crops=['img244-ann277', 'img390-ann375']...
step 3: loss=0.8608 crops=['img648-ann652', 'img187-ann213']...
step 4: loss=0.3866 crops=['img6-ann3', 'img170-ann195']...
step 5: loss=0.0962 crops=['img620-ann616', 'img603-ann606']...
bloques con pesos cambiados: ['backbone.layer4', 'head.0', 'head.3']
tensores congelados que cambiaron: 0
validation crop img102-ann118 (dog): P(dog,cat) antes=[0.8203914165496826, 0.17960859835147858] después=[0.9970079064369202, 0.0029921659734100103]
máximo cambio de probabilidad en 8 crops de validation: 0.7095
checkpoint recargado: predicciones idénticas
```

- Los pasos del optimizador cambian solo los bloques entrenables (`layer4` y las
  dos `Linear` de la cabeza). Ningún peso ni estadística de BatchNorm de los
  bloques congelados cambia.
- La inferencia sobre los mismos 8 crops de validation cambia hasta 0.71 en
  probabilidad: el modelo realmente se actualizó.
- Esto **no** es una métrica de desempeño: 5 pasos no son un entrenamiento. El
  entrenamiento con early stopping y MLflow es ML-04.
