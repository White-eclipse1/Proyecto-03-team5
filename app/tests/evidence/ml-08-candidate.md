# Evidencia ML-08: candidato elegido solo con validation y congelado antes del test

Ejecución: 2026-10-02. Rama `feat/ml-08-candidate-selection`, código de selección
del commit `e579fc2`. MLflow de OPS-03 con las 12 corridas
de ML-07 (matriz `ml07-v1`).

## Correspondencia con la evaluación

- Rúbrica 3.3: la selección usa una métrica de validation predeclarada, remite a un
  run ID y un checkpoint inequívocos, y ocurre antes de evaluar test.
- Rúbrica 4.1 (parte que corresponde a ML-08): el checkpoint elegido queda fijo
  antes de evaluar test, y no puede cambiarse después.
- Issue #27 (ML-08): Agent Test (timestamps: la selección es anterior a la primera
  evaluación de test) y TDD (no se puede cambiar el candidato si existe una
  evaluación de test).

## 1. Política predeclarada

`app/classification/selection_policy.yaml` (versionada antes de congelar):

- **Candidatos:** runs `FINISHED` de `ml07-v1`.
- **Métrica:** menor `best_val_loss`, la de validation que vigila el early
  stopping y corresponde a `checkpoints/best.pt`.
- **Desempates:** mayor `best_val_accuracy` y, después, el nombre de la entrada.
- `SelectionPolicy` rechaza cualquier métrica que no sea de validation.

## 2. El test seguía oculto al congelar

Antes de congelar, en el MLflow y en el repositorio:

```text
runs activos en MLflow: 12 | métricas con "test": [] | runs con tag candidate: []
reports/candidates y reports/evaluations: no existían
```

## 3. Selección y congelamiento

`uv run python -m classification.selection freeze`:

```text
1. r03-sgd                            bb448230424146349a969253d30db43b best_val_loss=0.0406 best_val_accuracy=0.9766 (época 3)
 2. r07-img224                         83e5a40ad86942478269c49a66d1d04b best_val_loss=0.0574 best_val_accuracy=0.9766 (época 1)
 3. r06-lr3e-4-ep15                    e6921aea1f3b409695b06736aa8f4f8f best_val_loss=0.0719 best_val_accuracy=0.9688 (época 1)
 4. r02-adamw                          1d571224e7fc46468b495cdcbd33e88f best_val_loss=0.0802 best_val_accuracy=0.9766 (época 5)
 5. r01-base                           f0cd9825563e45ccbdfe9dfd652a06c8 best_val_loss=0.0814 best_val_accuracy=0.9688 (época 5)
 6. r12-ep6                            adc1c0b4ec304e28b626e8dacb458c29 best_val_loss=0.0814 best_val_accuracy=0.9688 (época 5)
 7. r11-adamw-img160-head256-drop0.5   06fc194ab16242f7a8403da8cd87a849 best_val_loss=0.0850 best_val_accuracy=0.9609 (época 2)
 8. r04-batch16                        306817e87ca144eb9b9d716816097181 best_val_loss=0.1315 best_val_accuracy=0.9297 (época 3)
 9. r10-no-hidden-drop0                137244d23bd34a24b1633bdd9f44f49b best_val_loss=0.1516 best_val_accuracy=0.9375 (época 4)
10. r09-head256-64-drop0.3             21db27c7ecd24ebc894bf2219b875148 best_val_loss=0.1583 best_val_accuracy=0.9531 (época 4)
11. r05-batch64                        f6b548b403714d4a984cf756198a9c59 best_val_loss=0.1646 best_val_accuracy=0.9531 (época 1)
12. r08-img96                          d9c99b5a0e4e4b3c84dd4c49a96cd82c best_val_loss=0.2962 best_val_accuracy=0.8984 (época 4)
Candidato congelado: r03-sgd bb448230424146349a969253d30db43b en /Users/stephanieborregoarroyo/iCloud Drive (archivado)/Documents/MLOps/Proyecto-03-team5/reports/candidates/ml08_candidate.json
```

`r01-base` y `r12-ep6` empatan en `best_val_loss` y `best_val_accuracy` (ver la
evidencia de ML-07); el desempate por entrada los ordena de forma fija. No afecta
al candidato.

## 4. Candidato congelado (`reports/candidates/ml08_candidate.json`)

| Campo | Valor |
|-------|-------|
| Entrada | `r03-sgd` |
| run_id | `bb448230424146349a969253d30db43b` |
| Checkpoint | `runs:/bb448230424146349a969253d30db43b/checkpoints/best.pt` |
| sha256 del checkpoint | `84d6c88b4bd84c3e6f0229dd23f7f188ff85622bbb39e5c3988a97598fa90d52` |
| Release / manifest_hash | `v0.1.1` / `sha256:178b28bddb1bef5c05975fc7150710658627e31af08a1a12d94b98f9e99cb2b2` |
| Commit que entrenó | `86a33f8426226d0e3da9d75cf849765aa06e6396` |
| Commit que seleccionó | `e579fc2e6a00bfb9862a09d72f0d033b89b2b997` |
| Métrica | `best_val_loss` (min) = 0.040618, split `validation` |
| `frozen_at` (UTC) | `2026-10-02T00:37:26.037627Z` |

En MLflow, el run tiene `candidate=true` y `candidate_frozen_at=2026-10-02T00:37:26.037627Z`.

## 5. Agent Test: congelamiento anterior a todo test

`uv run python -m classification.selection check`:

```text
Candidato r03-sgd (bb448230424146349a969253d30db43b) congelado en 2026-10-02T00:37:26.037627+00:00; evaluaciones de test: 0, anteriores al congelamiento: 0
```

ML-09 debe llamar `require_frozen_candidate()` y guardar sus evaluaciones en
`reports/evaluations/` con `created_at`; `check` seguirá demostrando que todas
son posteriores a `frozen_at`.

## 6. El candidato no puede cambiar

Sobre el candidato real, sin modificarlo:

```text
re-congelar el mismo run conserva frozen_at: True
cambiar a r07-img224 sin replace -> rechazado: Ya hay un candidato congelado (r03-sgd, bb448230424146349a969253d30db43b); usa replace=True para cambiarlo antes de evaluar test
candidato sigue siendo: r03-sgd
```

Con una evaluación de test presente, ni siquiera `replace=True` permite cambiarlo
(`test_candidate_cannot_change_once_a_test_evaluation_exists`, requisito TDD del
issue), y no se puede congelar por primera vez si ya se consultó test
(`test_freezing_after_a_test_evaluation_is_refused`).

## 7. El candidato también llega a un clon limpio

El snapshot de MLflow (`data/mlflow-snapshot`, DVC `prod`) se actualizó con el tag
del candidato. Restaurado en un stack de Compose nuevo y vacío:

```text
clon limpio restaurado: runs con candidate=true -> [('bb448230424146349a969253d30db43b', 'r03-sgd', '2026-10-02T00:37:26.037627Z')]
coincide con reports/candidates/ml08_candidate.json: True
```
