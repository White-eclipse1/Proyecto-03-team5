# Evidencia OPS-07: modelo publicado en AWS S3 y recuperado en un entorno limpio

Ejecución: 2026-10-02, rama `feat/ops-07-s3-model-publication`, perfil SSO
`mlops-p2` (rol `MLOpsDataAccess`), región `us-east-1`. Model version `1.0.0`:
run `bb448230424146349a969253d30db43b`, release `v0.1.1`.

## Bucket elegido

Los buckets `-artifacts-` de Terraform (P2-06) no están desplegados: el README de
`terraform/` los describe como código "sin despliegue". Aplicar `environments/prod`
crearía además VPC, EC2, RDS y otros buckets, y el rol del equipo no tiene esos
permisos.

El modelo se publicó en un bucket **real de AWS** del equipo,
`mlops-p2-dvc-cache-280764207006`, bajo el prefijo `models/`. Ese prefijo es
independiente del `files/` que usa DVC. El objetivo es la rúbrica 5.2: objeto real
en S3, no solo configuración ni MinIO.

## 1. Publicación

```text
$ uv run python -m classification.publication publish 1.0.0 --bucket mlops-p2-dvc-cache-280764207006 --profile mlops-p2
s3://…/models/dog-cat-resnet18/1.0.0/checkpoint/best.pt VersionId=None sha256=84d6c88b4bd84c3e6f0229dd23f7f188ff85622bbb39e5c3988a97598fa90d52 (S3 ChecksumSHA256=84d6c88b…0d52)
s3://…/models/dog-cat-resnet18/1.0.0/model-card.md     VersionId=None sha256=b3dd3e51a9e0adb70c1d94b48cbba8a08eff8fe105bfaebb5a352f5f5e7a8703 (S3 ChecksumSHA256=b3dd3e51…8703)
s3://…/models/dog-cat-resnet18/1.0.0/package.json      VersionId=None sha256=b584f4cb8cb19bade61a577a5d136500b68db6ffda8f3c0cedbb2a116d9592d0 (S3 ChecksumSHA256=b584f4cb…92d0)
s3://…/models/dog-cat-resnet18/1.0.0/dependencies.json VersionId=None sha256=71e0bddd72fbf056bf1d80d454cf046e634ec9ae81708d3ee58031785d732504 (S3 ChecksumSHA256=71e0bddd…2504)
```

El registro está en [`reports/models/s3_publications.json`](../../../reports/models/s3_publications.json).
El checkpoint tiene el mismo SHA-256 que el `checkpoint_sha256` congelado en ML-08 y
registrado en OPS-06.

## 2. Verificación independiente con la AWS CLI

```text
$ aws s3api head-object --bucket mlops-p2-dvc-cache-280764207006 --key models/dog-cat-resnet18/1.0.0/checkpoint/best.pt --checksum-mode ENABLED
ContentLength 45043841, ETag "5fb6da4eaa508973c7967a8f04562be4", VersionId null,
ChecksumSHA256 hNbIi0vYTD5vAindI/fxiP+FYiu7OeXDmIqXWY+pDVI=, Metadata.sha256 84d6c88b…0d52, SSE AES256
$ aws s3api get-object … best.pt && shasum -a 256 best.pt
84d6c88b4bd84c3e6f0229dd23f7f188ff85622bbb39e5c3988a97598fa90d52
$ … ChecksumSHA256 | base64 -d | xxd -p      → 84d6c88b4bd84c3e6f0229dd23f7f188ff85622bbb39e5c3988a97598fa90d52
```

La model card da lo mismo: `head-object` responde 1790 bytes, y `shasum` del archivo
descargado da `b3dd3e51…8703`, igual que el registrado.

- **SHA-256 local:** se calcula del archivo real, antes de subirlo y otra vez sobre
  la descarga.
- **Checksum de S3:** el `ChecksumSHA256` que calcula S3 coincide con el SHA-256
  local.
- **ETag:** `5fb6da4e…` es el MD5 de una subida simple. Se registra aparte y
  **no** se usa como SHA-256.
- **VersionId:** el bucket no tiene versionado, así que S3 no devuelve `VersionId`
  y queda registrado como `null` ("si existe").

## 3. Agent Test: recuperar e inferir en un proceso limpio

```text
$ uv run python -m classification.publication verify 1.0.0 --profile mlops-p2 --image ../data/crops/cat/img369-ann355.png
checkpoint/best.pt: head-object=OK VersionId=None sha256 descargado=84d6c88b…0d52
model-card.md: head-object=OK VersionId=None sha256 descargado=b3dd3e51…8703
package.json: head-object=OK VersionId=None sha256 descargado=b584f4cb…92d0
dependencies.json: head-object=OK VersionId=None sha256 descargado=71e0bddd…2504
proceso limpio (pid 60859): img369-ann355.png → cat {'dog': 0.36790385842323303, 'cat': 0.6320961713790894}
Publicación verificada
```

El checkpoint se cargó e infirió en **otro proceso**, que solo vio los archivos
descargados de S3.

**Más inferencias:**
- Con el `best.pt` que descargó la AWS CLI, en otro proceso nuevo,
  `dog/img218-ann249.png` dio `dog` con probabilidad 1.0.
- Ambas probabilidades coinciden con las de la evaluación final de ML-09 y la
  auditoría de ML-10: el modelo de S3 es el modelo evaluado.

## 4. Una versión anterior sigue recuperable

En AWS hoy solo existe la versión `1.0.0`. No se publicó una versión artificial solo
para la prueba. Cada model version tiene su propia ruta
(`models/dog-cat-resnet18/<versión>/`) y `publish` nunca sobrescribe una key con
otro contenido. Al publicar `1.1.0`, `1.0.0` sigue intacta y resoluble.

Las pruebas lo demuestran:
- `test_a_previous_model_version_remains_recoverable`: publica `1.0.0` y `1.1.0`,
  alguien sobrescribe la key de `1.0.0` y el `VersionId` registrado recupera el
  original;
- `test_a_different_object_already_at_the_key_is_never_overwritten`;
- `test_agent_test_detects_an_object_changed_in_s3`: sin versionado, `verify`
  detecta el cambio por SHA-256.

**Recomendación para el admin de AWS:** habilitar el versionado del bucket. Así el
`VersionId` protegería también contra sobrescrituras hechas fuera de este código.

## 5. Pruebas

`uv run pytest tests/test_model_publication.py`: 14 tests más 1 de integración.
- Los 14 usan un S3 en memoria con la semántica de boto3: versiones, `VersionId`,
  `ETag` y `ChecksumSHA256`.
- El de integración publica y recupera contra un S3 real. Pasó contra el MinIO local
  con versionado, y se limpia solo:
  `S3_INTEGRATION_ENDPOINT=http://127.0.0.1:9000 uv run pytest -k real_s3`.
- Otros casos cubiertos: rechazo de un paquete local alterado, de un
  `ChecksumSHA256` distinto y de una descarga distinta; publicación sin versionado;
  republicar sin subir nada; no sobrescribir; Agent Test completo con inferencia en
  otro pid; e inferencia solo con el checkpoint descargado.
- Credenciales: ningún archivo versionado contiene access keys (patrones
  `AKIA`/`ASIA` y `aws_secret_access_key`), y el módulo usa solo la cadena de
  credenciales por defecto.

**Mutaciones** (worktree aislado): rompí 10 reglas a propósito:
- publicar sin verificar el paquete local;
- sobrescribir una key;
- ignorar el `ChecksumSHA256` de S3;
- no hacer la descarga de prueba;
- no pedir el checksum;
- tomar el ETag como SHA-256;
- no registrar el `VersionId`;
- que `verify` ignore el `VersionId`;
- que `verify` no compare el SHA-256;
- inferir en el mismo proceso.

Mueren las 10.
