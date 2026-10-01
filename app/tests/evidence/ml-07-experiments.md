# Evidencia ML-07: matriz de 12 corridas reales en MLflow

Ejecución: 2026-10-01. Rama `feat/ml-07-experiment-matrix`, código del commit
`86a33f8` (el mismo en las 12 corridas). Servidor MLflow de OPS-03
(`docker compose up mlflow`), experimento `dogcat-classifier`. Manifiesto P3
`v0.1.1` (`sha256:178b28bddb1bef5c05975fc7150710658627e31af08a1a12d94b98f9e99cb2b2`), 469 crops de train y 128 de validation, pesos
ImageNet. El conjunto de test **no se usa** en ninguna corrida.

## Correspondencia con la evaluación

- Rúbrica 3.1: al menos 10 corridas terminadas con entrenamiento real y
  resultados distintos, misma versión de manifiesto y clases, y los siete
  parámetros con al menos dos valores; lista de run IDs.
- Rúbrica 3.2: por corrida, parámetros efectivos, semilla, commit, release DVC,
  hash del manifiesto, clases, estado, métricas por época y artefactos (curvas y
  checkpoint), consultados por la API de MLflow.
- Issue #26 (ML-07): Agent Test (enumerar por la API al menos diez run IDs
  `FINISHED` con parámetros distintos).

## Reproducir

```bash
docker compose up -d --wait mlflow
cd app
uv run python -m classification.experiments run --matrix classification/ml07_matrix.yaml
uv run python -m classification.experiments report --matrix classification/ml07_matrix.yaml \
  --out ../reports/experiments/ml07_runs.json
```

`run` salta las entradas que ya tienen un run `FINISHED`, y `report` termina con
código ≠ 0 si falla algún criterio. El reporte completo (parámetros, métricas,
commit, hashes DVC y checkpoint de cada run) está en
[`reports/experiments/ml07_runs.json`](../../../reports/experiments/ml07_runs.json).

## Run IDs válidos (Agent Test, leído de la API de MLflow)

Los 12 runs son `FINISHED`, con etiqueta `experiment_matrix=ml07-v1`. Las
métricas son de **validation** en la mejor época (pesos restaurados en
`checkpoints/best.pt`), no de test.

| Entrada | run_id | optimizer | batch | max_epochs | lr | image_size | hidden_layers | dropout | épocas | best_epoch | val_loss | val_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `r01-base` | `f0cd9825563e45ccbdfe9dfd652a06c8` | adam | 32 | 10 | 0.001 | 128 | `128` | 0.2 | 8 | 5 | 0.0814 | 0.9688 |
| `r02-adamw` | `1d571224e7fc46468b495cdcbd33e88f` | adamw | 32 | 10 | 0.001 | 128 | `128` | 0.2 | 8 | 5 | 0.0802 | 0.9766 |
| `r03-sgd` | `bb448230424146349a969253d30db43b` | sgd | 32 | 10 | 0.01 | 128 | `128` | 0.2 | 6 | 3 | 0.0406 | 0.9766 |
| `r04-batch16` | `306817e87ca144eb9b9d716816097181` | adam | 16 | 10 | 0.001 | 128 | `128` | 0.2 | 6 | 3 | 0.1315 | 0.9297 |
| `r05-batch64` | `f6b548b403714d4a984cf756198a9c59` | adam | 64 | 10 | 0.001 | 128 | `128` | 0.2 | 4 | 1 | 0.1646 | 0.9531 |
| `r06-lr3e-4-ep15` | `e6921aea1f3b409695b06736aa8f4f8f` | adam | 32 | 15 | 0.0003 | 128 | `128` | 0.2 | 4 | 1 | 0.0719 | 0.9688 |
| `r07-img224` | `83e5a40ad86942478269c49a66d1d04b` | adam | 32 | 10 | 0.001 | 224 | `128` | 0.2 | 4 | 1 | 0.0574 | 0.9766 |
| `r08-img96` | `d9c99b5a0e4e4b3c84dd4c49a96cd82c` | adam | 32 | 10 | 0.001 | 96 | `128` | 0.2 | 7 | 4 | 0.2962 | 0.8984 |
| `r09-head256-64-drop0.3` | `21db27c7ecd24ebc894bf2219b875148` | adam | 32 | 10 | 0.001 | 128 | `256,64` | 0.3 | 7 | 4 | 0.1583 | 0.9531 |
| `r10-no-hidden-drop0` | `137244d23bd34a24b1633bdd9f44f49b` | adam | 32 | 10 | 0.001 | 128 | `[]` | 0.0 | 7 | 4 | 0.1516 | 0.9375 |
| `r11-adamw-img160-head256-drop0.5` | `06fc194ab16242f7a8403da8cd87a849` | adamw | 32 | 15 | 0.0003 | 160 | `256` | 0.5 | 5 | 2 | 0.0850 | 0.9609 |
| `r12-ep6` | `adc1c0b4ec304e28b626e8dacb458c29` | adam | 32 | 6 | 0.001 | 128 | `128` | 0.2 | 6 | 5 | 0.0814 | 0.9688 |

## Criterios comprobados por `matrix_report`

| Criterio | Resultado |
|---|---|
| `at_least_min_runs` | ✅ |
| `at_least_min_distinct_results` | ✅ |
| `seven_params_vary` | ✅ |
| `no_duplicate_configs` | ✅ |
| `same_manifest_hash` | ✅ |
| `same_dataset_version` | ✅ |
| `same_classes_dog_cat` | ✅ |
| `all_record_seed` | ✅ |
| `all_record_git_commit` | ✅ |
| `all_record_dvc_release` | ✅ |
| `all_have_epoch_metrics` | ✅ |
| `all_have_checkpoint` | ✅ |
| `all_trained_on_real_split` | ✅ |

Valores por parámetro entre las corridas:

```json
{"optimizer": ["adam", "adamw", "sgd"], "batch_size": ["16", "32", "64"], "max_epochs": ["10", "15", "6"], "learning_rate": ["0.0003", "0.001", "0.01"], "image_size": ["128", "160", "224", "96"], "hidden_layers": ["", "128", "256", "256,64"], "dropout": ["0.0", "0.2", "0.3", "0.5"]}
```

## Resultados idénticos: `r01-base` y `r12-ep6`

`r12-ep6` solo cambia `max_epochs` de 10 a 6. `r01-base` encontró su mejor época
en la 5 y se detuvo en la 8. Con la misma semilla, el entrenamiento es
determinista (ML-05), así que `r12-ep6` repite las mismas 6 primeras épocas y
llega al **mismo** mejor checkpoint. No es una configuración duplicada (cambia
`max_epochs`), pero tampoco es un resultado distinto, y el reporte lo marca en
`identical_results`. Sin contarla quedan **11 corridas con
resultados distintos** (≥ 10), y `max_epochs` sigue tomando dos valores entre
ellas (10 y 15).

## Corridas de verificación archivadas

Antes de la matriz, el experimento tenía 16 corridas de verificación de ML-04 a
ML-06 (código anterior, incluidas parejas idénticas a propósito para probar la
reproducibilidad). Se archivaron con el borrado lógico de MLflow
(`MlflowClient.delete_run`, reversible con `restore_run`) para que no se
confundan con la matriz.
