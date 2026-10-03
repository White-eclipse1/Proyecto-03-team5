# Evidencia APP-09: verificación adversarial de las cinco pantallas del Proyecto 3

Ejecución: 2026-10-02.
- **Código:** rama `feat/app-09-adversarial-verification` sobre `main`, con APP-03 a APP-08 y OPS-04 a OPS-09.
- **Stack:** completo y aislado (`COMPOSE_PROJECT_NAME=app09check`), con credenciales temporales: nginx, `ml-api`, `training-worker`, MLflow, MariaDB y MinIO. El portal en `http://localhost:8080`.
- **Datos reales:**
  - release P2 `v0.1.1`, manifiesto `sha256:178b28bd…`;
  - recortes de ML-01 (`dvc pull crops`);
  - candidato `r03-sgd` de ML-08 y evaluación de test de ML-09;
  - paquete `dog-cat-resnet18 1.0.0` de OPS-06 (`dvc pull data/models.dvc`);
  - publicación en S3 de OPS-07.

Los casos automatizables están en
[`tests/test_app09_portal_adversarial.py`](../test_app09_portal_adversarial.py). La prueba pasa por nginx (`/api/ml/...`), igual que el navegador; sin stack se omite:

```bash
APP09_PORTAL_URL=http://localhost:8080 MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
APP09_AWS_PROFILE=mlops-p2 uv run pytest tests/test_app09_portal_adversarial.py -v
# 28 passed
```

El resto se revisó en el navegador con el mismo stack.

## Resultado por criterio (issue #51)

| Criterio | Cómo se probó | Resultado |
|---|---|---|
| Training rechaza parámetros inválidos | `batch_size 0`, `image_size 100`, `patience > max_epochs`, 6 capas, `learning_rate 0`, release inexistente: 422, y el número de jobs no cambia | ✅ |
| Training rechaza un release con gate fallido | `v0.1.0` (Quality Gate `failed`): 409 `training_blocked`, también un `manifest_hash` de otro manifiesto; sin jobs nuevos | ✅ |
| Refrescar durante el training conserva el job | El POST responde en < 5 s (el entrenamiento corre en el worker); el job se vuelve a leer por id con el mismo estado, logs y checkpoint `runs:/<run>/checkpoints/best.pt`. En el portal ya se había refrescado a mitad de un job (cierre de APP-03, #15) | ✅ |
| Experiments muestra runs reales | El run del job lanzado aparece en `/api/ml/runs` y en la pantalla (`nosy-conch-724`, `FINISHED`, `v0.1.1`) con su curva de `val_loss` | ✅ |
| Cambiar datos en MLflow cambia la UI | `log_metric(val_accuracy_top1, 0.4242)` directo en MLflow: la API y la columna de la pantalla muestran 0.4242 | ✅ |
| Evaluation no muestra el test antes del freeze | Pruebas de APP-05 (`test_evaluation_api.py`): sin candidato, con otra métrica de selección, otro run o manifest, evaluación anterior al freeze, más de una evaluación: ningún número de test | ✅ |
| La matriz de confusión suma el total correcto | Con el stack: matriz = predicciones = recortes `test` del manifiesto = filas del CSV = **71**. Pantalla: "68 de 71", "La matriz suma 71 = 71" | ✅ |
| Models no muestra published si no existe en S3 | Para cada objeto que Models marca publicado, `head-object` en el **S3 real** devuelve el mismo `ChecksumSHA256` (4/4). **Tras la revisión del PR #63:** `ml-api` ya no se fía solo de `s3_publications.json`; hace `head_object` de cada objeto antes de decir `published`. Una key inexistente, otro `VersionId` o un `ChecksumSHA256` distinto → `inconsistent`; sin credenciales de AWS → `unverifiable` ("No verificable"). Casos en `test_models_api.py` y `test_s3_verification.py`; pendiente repetirlo contra el stack (ver abajo) | ✅ (corregido) |
| Cambiar de model version cambia el artefacto | Inference responde con el `checkpoint_sha256` de la versión elegida (`84d6c88b…` para 1.0.0). Con dos versiones, cada una carga su checkpoint (`test_inference_api.py`, `test_models_api.py`) | ✅ |
| Inference rechaza archivos inválidos | PDF, texto con nombre `.png`, WebP y SVG: 415; vacío y PNG truncado: 422; versión inexistente: 422 `model_not_found`. Todos con mensaje útil | ✅ |
| Las probabilidades de inference son coherentes | Suman 1 y la clase es la de mayor probabilidad, también con imágenes de 1×1, grises, RGBA, 16 bits, paleta y CMYK. El recorte real `img369-ann355` da **cat 0.6321**, igual que ML-09; los 71 recortes de test reproducen ML-09 (`test_inference_real_model.py`) | ✅ |
| Send to annotation queue crea un registro real | Un resultado real de Inference crea la imagen en la cola: 201, la imagen guardada es idéntica byte a byte y un reenvío devuelve el mismo registro (200, sin duplicar). En el portal, "Enviar a cola de anotación" muestra "imagen 2" y el recorte aparece como **Pendiente** en Buscar. La cola rechaza trazabilidad falsa (ver hallazgos) | ✅ |
| No quedan mocks de producción | `frontend/src/ml` y `app/training` no importan el corpus de ejemplos ni tienen datos fijos | ✅ |
| No quedan placeholders funcionales | `test_every_endpoint_the_portal_calls_is_served_by_ml_api`: cada ruta de `ML_ENDPOINTS` existe en `ml-api` | ✅ |

## Bugs encontrados y corregidos

### nginx y las imágenes de más de 1 MB

**nginx cortaba las imágenes de más de 1 MB.** Inference acepta hasta 10 MB (la pantalla lo dice), pero `location /api/ml/` no fijaba `client_max_body_size`. nginx usaba su máximo por defecto de 1 MB y respondía a un PNG válido de 3.6 MB con un `413` en HTML, sin llegar a `ml-api`. El portal solo podía mostrar "El servidor respondió con estado 413". Las pruebas unitarias no lo veían porque no pasan por nginx.

- Regresión, en rojo antes de la corrección:
  - `test_nginx_accepts_the_uploads_that_inference_allows` (`test_ml_api_wiring.py`);
  - `test_inference_accepts_a_valid_image_of_several_megabytes` (contra el portal).
- Corrección: `client_max_body_size 11m;` en `location /api/ml/` (10 MB de imagen más el multipart). Con el frontend reconstruido, el PNG de 3.6 MB se clasifica (200).

## Correcciones de la revisión del PR #63

Un revisor encontró dos fallas que la verificación anterior no vio.

### 1. Models marcaba `published` un objeto que no existe en S3 (rúbrica 6.4)

Cambió en `reports/models/s3_publications.json` la key de `model-card.md` por una que da 404 en S3, y `GET /api/ml/models` siguió diciendo `published` con esa key. `training/models_view.py` solo comparaba la publicación con `registry.json`; la prueba de arriba con el S3 real solo revisaba los objetos que ya estaban bien.

- Regresión, en rojo antes de corregir (commit `51e53d4`):
  - `test_models_api.py`: key inexistente, objeto borrado, otro `VersionId`, `ChecksumSHA256` distinto o ausente → `inconsistent`; sin credenciales o sin verificador → `unverifiable`; todo confirmado → `published`;
  - `test_s3_verification.py`: el verificador real con `botocore.Stubber` (`ChecksumMode=ENABLED`, `VersionId`, 404 → `None`, 403/sin credenciales/sin red → `S3Unverifiable`, caché de 60 s);
  - contrato: `unverifiable` exige `problem` y no tiene bucket, región, fecha ni objetos (Pydantic, Zod y 3 casos nuevos en `invalid_cases.json`);
  - frontend: la pantalla muestra "No verificable" y el motivo, sin keys.
- Corrección (commit `609f5da`):
  - `training/s3_verification.py` (`BotoS3Verifier`): `head_object` de cada objeto, con caché por (bucket, key, version_id);
  - `create_app(s3_verifier=...)` es inyectable; `main()` usa boto3 con la cadena por defecto de AWS y la región de cada publicación;
  - el nuevo estado `unverifiable` está en el contrato, en la pantalla Models y en el README;
  - `docker-compose.yml` pasa a `ml-api` credenciales opcionales desde el entorno (vacías por defecto, sin secretos en el repo). Cómo habilitarlas: `app/training/README.md`, "Credenciales de AWS para Models".
- Mutaciones a mano, todas detectadas (10/10): no comparar el checksum, tratar el 404 como existente (en el verificador y en `models_view`), tratar sin credenciales o sin verificador como `published`, ignorar el `VersionId` (en `head_object` y en `models_view`), caché sin `VersionId`, no capturar `NoCredentialsError` (500) y omitir `ChecksumMode`.

### 2. nginx cortaba en `/api/` lo que Inference acepta

Con un PNG válido de exactamente 10 MiB, Inference respondía 200, pero "Enviar a cola de anotación" (`POST /api/images/from-inference`, backend de Node) recibía un `413` en HTML. El motivo: `location /api/` tenía `client_max_body_size 10m`, y ese límite incluye el multipart y `metadata`.

- Regresión, en rojo antes de corregir:
  - `test_nginx_accepts_the_uploads_that_inference_allows[/api/]`;
  - contra el portal, `test_annotation_queue_accepts_the_largest_image_that_inference_accepts`: PNG válido de 10 451 265 bytes, clasificado y luego enviado a la cola.
- Corrección: `client_max_body_size 11m;` también en `location /api/`. El backend sigue limitando el archivo a `MAX_UPLOAD_SIZE_BYTES` (10 MiB).

## Hallazgos en la cola de anotación (APP-08), corregidos

El caso normal funcionaba, pero el backend guardaba como "registro real" la trazabilidad que mandaba el navegador sin compararla con nada:

1. Aceptaba una versión del modelo inexistente (`9.9.9`), un `checkpointSha256` falso, otro modelo u otro run.
2. Aceptaba una `predictedClass` que no era la de mayor probabilidad.
3. En una imagen subida, no comparaba el sha256 de `sourceRef` con los bytes recibidos. Como la clave de idempotencia no incluye la imagen, otra imagen con los mismos metadatos devolvía el registro anterior.
4. Cualquier metadato inválido respondía **500**: el `ZodError` del esquema no se convertía en 400.

Pruebas en rojo antes de corregir:
- contra el portal: `test_annotation_queue_rejects_*` y `test_annotation_queue_checks_the_uploaded_image_against_its_sha256`;
- backend: `tests/inference-traceability.test.ts`, `tests/inference-queue.test.ts` y `tests/inference-queue-service.test.ts`;
- wiring de compose: `test_backend_verifies_annotation_queue_traceability_against_the_reports`.

Corrección en `backend/src/logic/inference-traceability.ts`, que se aplica antes de buscar o subir nada:
- modelo, versión, run y checkpoint contra `reports/models/registry.json`;
- un recorte contra su sha256 en `reports/crops.json` (mismo release);
- una imagen subida contra el sha256 de `sourceRef`;
- la clase predicha es la de mayor probabilidad;
- metadatos inválidos responden 400 con el motivo, y un reporte ilegible responde 503.

El servicio `backend` monta `./reports` de solo lectura (`REPORTS_DIR`). Las 6 mutaciones del verificador se detectan. Con la imagen atada a `sourceRef`, una misma clave implica la misma imagen.

## Otros intentos sin hallazgos

- **Path traversal por el portal** (`/api/ml/crops/..%2F..%2Fetc%2Fpasswd` y `/api/ml/models/1.0.0/files/..%2F…`): nginx resuelve los `..` antes de pasar la petición, así que ninguna llega a `ml-api`. La primera recibe el `index.html` del portal y la segunda el 404 del backend de Node ("Cannot GET /etc/passwd"); ninguna respuesta contiene el archivo. Además, `ml-api` rechaza por sí mismo el traversal codificado (`test_evaluation_api.py`, `test_models_api.py`).
- **Bodies inválidos:** JSON mal formado (400), `schema_version` 2.0, campos de más (por ejemplo `predicted_class` en la petición) o IDs negativos (422). Un recorte inexistente da 422 `crop_not_found`.
- **Descargas de Models:** solo los archivos que el registro lista para esa versión; `package.json` y las rutas fuera del paquete dan 404.

## Pendiente

- Repetir contra el stack, después de reconstruir `ml-api` y `frontend`:
  - Models con credenciales (`aws configure export-credentials`) → `published`;
  - Models sin credenciales → "No verificable";
  - el PNG de casi 10 MiB por `/api/images/from-inference`.
- Después sigue APP-10 (#52): el recorrido completo desde el portal.
