# Evidencia OPS-06: paquete semántico y registro de modelos

Ejecución: 2026-10-02, rama `feat/ops-06-model-registry`. Corrige la revisión del
PR #58:
- se validaba la metadata, pero no que el paquete contuviera el checkpoint correcto;
- el `registry.json` estaba escrito a mano: no había paquete ni checkpoint
  verificable.

## 1. Integridad del checkpoint

Reproducción con el código anterior: `materialize_model_package` copió como
`best.pt` un archivo con sha256 `5940f4ee…` aunque el paquete registraba
`84d6c88b…`.

Ahora el comportamiento es este:
- verifica el sha256 **antes** de escribir nada y falla si no coincide
  (`test_materialize_rejects_a_checkpoint_whose_sha256_does_not_match`);
- verifica de nuevo la copia;
- `resolve_checkpoint` vuelve a verificar cada archivo del paquete al resolver una
  versión (`test_resolve_checkpoint_rejects_a_tampered_package`).

## 2. Flujo real: MLflow → paquete → registro

```text
$ uv run python -m classification.registry publish 1.0.0
dog-cat-resnet18 1.0.0 → data/models/dog-cat-resnet18/1.0.0
  checkpoint/best.pt: 84d6c88b4bd84c3e6f0229dd23f7f188ff85622bbb39e5c3988a97598fa90d52
  model-card.md: b3dd3e51a9e0adb70c1d94b48cbba8a08eff8fe105bfaebb5a352f5f5e7a8703
  dependencies.json: 71e0bddd72fbf056bf1d80d454cf046e634ec9ae81708d3ee58031785d732504

$ uv run python -m classification.registry verify 1.0.0
1.0.0 → run bb448230424146349a969253d30db43b → checkpoint 84d6c88b4bd8… → release v0.1.1 → sha256:178b28bddb1bef5c05975fc7150710658627e31af08a1a12d94b98f9e99cb2b2
Cadena verificada
```

- **De dónde sale el checkpoint:** se descargó del run `bb44823…` en MLflow, y su
  sha256 es el `checkpoint_sha256` congelado en ML-08.
- **Cómo se generó el registro:** `reports/models/registry.json` lo genera
  `publish`, no está escrito a mano. Incluye `package_path` y el sha256 de cada
  archivo del paquete. Ya no tiene el texto con la codificación rota del registro
  anterior.
- **Dónde vive el paquete:** `data/models` (45 MB) está versionado con DVC
  (`data/models.dvc`). `dvc push -r prod` subió 4 archivos y después `dvc status`
  dio "in sync".

## 3. Clon limpio

Lo simulé con un worktree nuevo, con solo el remote `prod` configurado:

```text
$ dvc pull -r prod data/models.dvc          → 5 files fetched
$ python -m classification.registry resolve 1.0.0
.../data/models/dog-cat-resnet18/1.0.0/checkpoint/best.pt
$ python -m classification.registry verify 1.0.0
Cadena verificada
```

Sin `dvc pull`, `resolve` falla con "Falta …/best.pt (¿dvc pull data/models.dvc?)".
No inventa un checkpoint.

**Inferencia en un proceso limpio con el checkpoint resuelto:**
- `cat/img369-ann355`: `{dog: 0.3679, cat: 0.6321}`;
- `dog/img218-ann249`: `{dog: 1.0, cat: 0.0}`.

Son las mismas probabilidades que la evaluación final de ML-09 y la auditoría de
ML-10: el paquete contiene el checkpoint evaluado.

## 4. Pruebas

`uv run pytest tests/test_model_registry.py`: 27 tests. Son los 10 originales y 17
nuevos.

Los nuevos son de punta a punta: entrenan un run corto en un MLflow local, lo
congelan como ML-08, evalúan el test como ML-09 y luego publican. Cubren:
- rechazo por sha256 distinto;
- el paquete contiene exactamente el artefacto del run;
- tarjeta con propósito, release, manifest hash, run, métricas y limitaciones
  reales;
- dependencias instaladas de verdad;
- rechazo si el checkpoint del run no es el congelado o si falta la evaluación
  final;
- una versión nueva mantiene recuperable la anterior;
- republicar la misma versión no cambia nada, y con otro contenido se rechaza;
- una entrada con solo metadata no resuelve a un checkpoint;
- paquete alterado;
- el Agent Test detecta un manifiesto de otro release, un run de otro dataset y un
  checkpoint cambiado en MLflow;
- el registro versionado apunta a un paquete verificable.

**Mutaciones** (worktree aislado): rompí 13 reglas a propósito, entre ellas:
- materializar sin verificar el sha256 (el bug reportado) o escribiendo antes de
  verificar;
- no verificar el sha256 del run;
- publicar sin la evaluación final o reescribir una versión;
- resolver sin verificar o aceptar solo metadata;
- `verify` sin los tags, sin el checkpoint de MLflow o sin el manifest hash;
- registrar sin sha256 de archivos, con dependencias fijas o con limitaciones
  genéricas.

Mueren las 13.
