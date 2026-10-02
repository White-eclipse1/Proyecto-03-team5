# Evidencia ML-09: evaluación final sobre el test congelado

Ejecución: 2026-10-02. Rama `feat/ml-09-frozen-test-evaluation`, código del commit
`6528332`. Candidato congelado de ML-08: `r03-sgd` (`bb448230424146349a969253d30db43b`),
`frozen_at = 2026-10-02T00:37:26.037627Z`. Manifiesto P3 `v0.1.1` (`sha256:178b28bddb1bef5c05975fc7150710658627e31af08a1a12d94b98f9e99cb2b2`),
split `test` (71 crops).

## Correspondencia con la evaluación

- Rúbrica 4.1:
  - el checkpoint elegido se evalúa una vez sobre el 10% de test, sin entrenar,
    aumentar ni ajustar nada con esas etiquetas;
  - se guardan por muestra ID, clase real, clase predicha y probabilidades;
  - la evaluación se repite para auditoría con el mismo artefacto y manifiesto;
  - el orden temporal y los hashes quedan verificados.
- Rúbrica 4.2: matriz con filas reales y columnas predichas; accuracy, F1 macro, y
  precision/recall/support por clase. Recalculados de forma independiente desde
  las predicciones y comparados con MLflow.
- Rúbrica 4.3: `aciertos / total de crops de test` comparado con 0.85 sin redondear.
- Issue #28 (ML-09): Agent Test (recalcular accuracy y F1 desde el archivo de
  predicciones y compararlos con MLflow) y TDD (métricas con predicciones conocidas).

## Archivos de auditoría

- [`reports/evaluations/test/test-r03-sgd-20261002T012633Z.json`](../../../reports/evaluations/test/test-r03-sgd-20261002T012633Z.json):
  contrato `Evaluation` de APP-01. Valida que la matriz, las métricas por clase, la
  accuracy y el F1 macro sean consistentes entre sí y con las 71 predicciones.
- [`reports/evaluations/test/test-r03-sgd-20261002T012633Z.predictions.csv`](../../../reports/evaluations/test/test-r03-sgd-20261002T012633Z.predictions.csv):
  por crop, `crop_id`, `image_id`, `annotation_id`, clase real, clase predicha,
  `p_dog`, `p_cat` y acierto.
- Los dos archivos también están como artefactos del run en MLflow
  (`evaluation/test/`) y en el snapshot DVC `data/mlflow-snapshot`.

## 1. Evaluación final

`uv run python -m classification.evaluation evaluate`:

```text
Evaluación test-r03-sgd-20261002T012633Z de r03-sgd (bb448230424146349a969253d30db43b)
accuracy = 68/71 = 0.9577464788732394
¿>= 0.85 sin redondear?: True
f1_macro = 0.9548441806232775
confusion_matrix (filas=real, columnas=predicha ['dog', 'cat']):
  dog  [43, 0]
  cat  [3, 25]
  dog: precision=0.9348 recall=1.0000 f1=0.9663 support=43
  cat: precision=1.0000 recall=0.8929 f1=0.9434 support=28
```

- Solo se evaluó el checkpoint congelado: su sha256 coincidió con
  `checkpoint_sha256` del candidato antes de cargarlo.
- Modo eval, sin gradiente ni optimizador; el preprocesamiento de test no tiene
  pasos aleatorios. Los tests lo comprueban: `test_evaluation_never_trains` y
  `test_evaluation_refuses_random_preprocessing`.
- **Meta:** 68/71 = 0.9577464788732394 ≥ 0.85, comparado sin redondear.

## 2. Agent Test: métricas desde el CSV contra MLflow

`uv run python -m classification.evaluation verify`:

```text
test_accuracy            archivo=0.9577464788732394     mlflow=0.9577464788732394     OK
test_f1_macro            archivo=0.9548441806232775     mlflow=0.9548441806232775     OK
test_correct             archivo=68.0                   mlflow=68.0                   OK
test_total               archivo=71.0                   mlflow=71.0                   OK
test_precision_dog       archivo=0.9347826086956522     mlflow=0.9347826086956522     OK
test_recall_dog          archivo=1.0                    mlflow=1.0                    OK
test_f1_dog              archivo=0.9662921348314606     mlflow=0.9662921348314606     OK
test_support_dog         archivo=43.0                   mlflow=43.0                   OK
test_precision_cat       archivo=1.0                    mlflow=1.0                    OK
test_recall_cat          archivo=0.8928571428571429     mlflow=0.8928571428571429     OK
test_f1_cat              archivo=0.9433962264150945     mlflow=0.9433962264150945     OK
test_support_cat         archivo=28.0                   mlflow=28.0                   OK
```

## 3. Recálculo independiente con scikit-learn

```text
filas=71 | crop_ids == split test del manifiesto: True | ninguno de train/validation: True
accuracy sklearn: 0.9577464788732394 | f1_macro sklearn: 0.9548441806232775
confusion_matrix sklearn (labels=[dog,cat]): [[43, 0], [3, 25]] | suma: 71
por clase sklearn: {'dog': (np.float64(0.9348), np.float64(1.0), np.float64(0.9663), 43), 'cat': (np.float64(1.0), np.float64(0.8929), np.float64(0.9434), 28)}
coincide con el archivo Evaluation: True
baseline clase mayoritaria (dog) en el mismo test: 43/71 = 0.6056
errores: [('img18-ann16', 'cat', 'dog', 0.769), ('img27-ann42', 'cat', 'dog', 0.535), ('img366-ann351', 'cat', 'dog', 0.795)]
```

Los 71 `crop_id` son exactamente los del split `test` del manifiesto. El
baseline de clase mayoritaria (siempre `dog`) da 0.6056 en el mismo test.

**Lectura de errores:**
- Los 3 errores son gatos predichos como perro; ningún perro se clasificó como
  gato.
- El recall de `cat` (0.893) es el más bajo: el 95.8 % de accuracy no oculta un
  recall bajo, pero `cat` es la clase con más errores.

## 4. Auditoría: la evaluación se reproduce

`uv run python -m classification.evaluation audit` vuelve a inferir el test con el
mismo checkpoint y manifiesto, sin escribir nada:

```text
Auditoría: sin diferencias
```

Los archivos de `reports/evaluations/` no cambiaron (mismo sha). Una segunda
evaluación "final" se rechaza (`test_the_final_evaluation_happens_only_once`).

## 5. El test se consultó después del congelamiento, y el candidato quedó bloqueado

```text
Candidato r03-sgd (bb448230424146349a969253d30db43b) congelado en 2026-10-02T00:37:26.037627+00:00; evaluaciones de test: 1, anteriores al congelamiento: 0
```

- `created_at` de la evaluación: `2026-10-02T01:26:33.097538Z`; `frozen_at`: `2026-10-02T00:37:26.037627Z`.
- Volver a seleccionar ahora se rechaza, porque el candidato ya tiene métricas de
  test:

```text
raise ValueError(
ValueError: Runs de ml07-v1 con métricas de test (['r03-sgd']): el candidato debe elegirse antes de consultar el test
```

## 6. Clon limpio

Snapshot actualizado (DVC `prod`) y restaurado en un stack de Compose vacío: 12
runs OK, y `verify` sobre ese MLflow da las 12 métricas iguales al CSV.
