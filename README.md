# Portal de anotación de imágenes

Monolito para subir, anotar y exportar un dataset de detección de objetos.
Las imágenes se almacenan en MinIO; los metadatos y las anotaciones en MariaDB.

## Estructura

```text
backend/    API HTTP con Express (UI → Logic → Data)
frontend/   Interfaz React + Vite (upload, anotación, dashboard, búsqueda)
```

Cada carpeta es un paquete npm independiente con su propio `package.json`.

## Arquitectura

```text
frontend  →  backend
                ├── src/ui      Endpoints HTTP
                ├── src/logic   Reglas de negocio y validación con Zod
                └── src/data    Drizzle (MariaDB) y MinIO
                                    ├── MariaDB  metadatos y anotaciones
                                    └── MinIO    archivos binarios
```

La capa UI nunca accede a MariaDB ni a MinIO: solo invoca a `logic`. La capa
`logic` es la única que puede importar de `data`.

Todo dato que entra por HTTP se valida con Zod antes de llegar a la capa de
datos, y los tipos se infieren del esquema con `z.infer`. La capa Logic lanza
errores tipados que la UI mapea a códigos HTTP:

| Error             | HTTP | Cuándo                                       |
|-------------------|------|----------------------------------------------|
| `ValidationError` | 400  | Dato mal formado o regla de negocio violada  |
| `NotFoundError`   | 404  | El recurso no existe en la base de datos     |

## Contribuir

Convención de ramas, commits y PR, qué valida el CI y cómo correrlo en tu máquina:
[CONTRIBUTING.md](CONTRIBUTING.md).

## Onboarding de desarrollo

Sigue esta sección de arriba hacia abajo en un clon nuevo. El proyecto tiene
dos entornos Python separados: `app/.venv` para el pipeline y `.venv-dvc`
para DVC. No los mezcles.

### 1. Herramientas necesarias

Obligatorias para el trabajo habitual:

- Git.
- Python 3.12.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) para el
  entorno Python de `app/`.
- Docker Desktop y Docker Compose si vas a levantar los servicios locales.
- Node.js y npm si vas a desarrollar o probar `backend/` y `frontend/`.
- AWS CLI v2 si necesitas leer el dataset de producción desde S3.

Opcionales:

- Terraform CLI, únicamente para validar o trabajar en infraestructura con
  autorización explícita.
- MinIO local, únicamente para el remote DVC `dev`. No es necesario para
  recuperar el dataset de producción.

No necesitas access keys permanentes para el onboarding. El acceso de AWS de
este proyecto usa IAM Identity Center / SSO con tu propia identidad.

### 2. Clonar el repositorio

```bash
git clone https://github.com/karenelizabg/proyecto-fase2-MLOPS.git
cd proyecto-fase2-MLOPS
git status
```

La comprobación inicial debe mostrar la rama y el estado del clon. No asumas
una ruta local concreta: trabaja desde la carpeta que acabas de clonar.

### 3. Preparar Python del pipeline

Desde `app/`, instala exactamente las dependencias fijadas por `uv.lock`:

```bash
cd app
uv sync --locked --no-build
uv run python --version
uv run pytest -q
cd ..
```

`uv` crea o utiliza `app/.venv`. Este entorno corresponde al pipeline,
quality gate, analyzers, tests y Copilot Python. No lo sustituyas por
`.venv-dvc`, que es exclusivo de DVC.

### 4. Preparar el entorno separado de DVC

Desde la raíz del repositorio:

```bash
python3.12 -m venv .venv-dvc
source .venv-dvc/bin/activate
python -m pip install 'dvc[s3]==3.67.1'
python --version
dvc --version
dvc remote list
```

En PowerShell, activa el mismo entorno con:

```powershell
py -3.12 -m venv .venv-dvc
.venv-dvc\Scripts\Activate.ps1
```

El repositorio ya está inicializado y ya contiene sus remotes. **No ejecutes
`dvc init`.**

### 5. Comprobar AWS CLI v2 (macOS)

Comprueba primero si ya está instalada:

```bash
aws --version
```

Si no aparece el comando, instala AWS CLI v2 siguiendo el instalador oficial
para macOS de [AWS](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
Por ejemplo, el instalador oficial puede ejecutarse así:

```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "/tmp/AWSCLIV2.pkg"
sudo installer -pkg "/tmp/AWSCLIV2.pkg" -target /
aws --version
```

### 6. Configurar IAM Identity Center / SSO

Cada integrante tiene su propio usuario de IAM Identity Center. No recibas ni
copies el password, la sesión SSO o las credenciales de otra persona.

Ejecuta:

```bash
aws configure sso --profile mlops-p2
```

Cuando la CLI lo solicite, responde:

| Pregunta | Valor |
|---|---|
| SSO session name | `mlops-p2` |
| SSO start URL | `https://d-90667fbedf.awsapps.com/start` |
| SSO region | `us-east-1` |
| Registration scopes | `sso:account:access` |

Se abrirá el navegador. Inicia sesión con **tu propio usuario** de IAM
Identity Center, selecciona la cuenta que te haya asignado el administrador y
elige el permission set que te haya autorizado. El README no fija nombres de
usuarios, contraseñas, cuentas ni identificadores personales.

Después inicia la sesión y comprueba la identidad efectiva:

```bash
aws sso login --profile mlops-p2
aws sts get-caller-identity --profile mlops-p2
```

La salida debe corresponder a tu sesión autorizada. Cuando expire, normalmente
basta con renovar la sesión:

```bash
aws sso login --profile mlops-p2
```

La configuración del perfil y la caché de la sesión se guardan fuera del
repositorio, en la configuración local de AWS CLI. No copies esos archivos al
proyecto ni los compartas.

### 7. Conectar DVC con el perfil local de AWS

Con `.venv-dvc` activado, configura el remote `prod` solo en tu máquina:

```bash
dvc remote modify --local prod profile mlops-p2
git check-ignore .dvc/config.local
dvc remote list
```

La opción `--local` es obligatoria: escribe el perfil en `.dvc/config.local`,
no en la configuración versionada de `.dvc/config`. Ese archivo local debe
permanecer ignorado y nunca subirse a Git.

### 8. Comprobar lectura del bucket de producción

Estas comprobaciones son de lectura y no modifican datos:

```bash
aws s3api head-bucket \
  --bucket mlops-p2-dvc-cache \
  --profile mlops-p2

aws s3api list-objects-v2 \
  --bucket mlops-p2-dvc-cache \
  --max-keys 1 \
  --query KeyCount \
  --profile mlops-p2
```

Si terminan correctamente, tu sesión puede alcanzar el bucket y tiene los
permisos requeridos para esas operaciones. Después puedes consultar el estado
de los metadatos DVC sin subir datos:

```bash
dvc status -r prod data/raw/images.dvc data/raw/annotations.dvc
```

Descarga el dataset únicamente cuando realmente lo necesites:

```bash
dvc pull -r prod data/raw/images.dvc data/raw/annotations.dvc
```

`dvc pull` materializa archivos en tu máquina, pero no sube nada a S3. No uses
`dvc push` como prueba de conectividad; publicar requiere autorización de
escritura y una tarea explícita.

### 9. Permisos y responsabilidades del administrador

Para que el flujo funcione, el administrador debe haber creado o asignado tu
usuario en IAM Identity Center, asignado la cuenta AWS correspondiente y
asignado un permission set con acceso S3. Para lectura del remote DVC se
necesitan conceptualmente permisos equivalentes a `s3:ListBucket` y
`s3:GetObject`. Publicar datasets requiere permisos adicionales de escritura
definidos por el administrador.

### 10. MinIO / `dev` (opcional)

Los remotes no son intercambiables:

- `prod` = AWS S3 compartido, bucket `mlops-p2-dvc-cache`.
- `dev` = MinIO local, bucket `dvc-cache`.

Si solo necesitas recuperar el dataset de producción, no levantes MinIO. El
flujo opcional completo está documentado en [P2-04 — MinIO local y remotes
DVC](#p2-04--minio-local-y-remotes-dvc). Sus credenciales son locales de
MinIO y no tienen relación con AWS SSO.

### 11. Terraform (opcional y autorizado)

Terraform local puede usar el perfil AWS `mlops-p2` mediante la cadena normal de
credenciales. GitHub Actions usa OIDC, que es un mecanismo distinto; OIDC de
GitHub no autentica automáticamente tu Mac. No ejecutes `terraform apply` ni
`terraform destroy` como parte del onboarding. Tampoco inicialices el backend
remoto hasta que el administrador proporcione y confirme el bucket de state.
Consulta [terraform/README.md](terraform/README.md) para la validación estática
y los límites operativos.

### 12. Variables locales y seguridad

El `.env` de la raíz es para configuración local de Compose/MinIO/Copilot. No
coloques credenciales AWS, passwords, sesiones SSO ni access keys en `.env`.
`ANTHROPIC_API_KEY` es opcional y solo se necesita para utilizar el chat
Copilot; nunca pongas una API key real en esta documentación.

Cada persona usa su propia identidad AWS, no comparte passwords, sesiones SSO
ni access keys, y mantiene `.dvc/config.local` fuera de Git.

## Requisitos

La lista completa y secuencial de instalación está en
[Onboarding de desarrollo](#onboarding-de-desarrollo). Para ejecutar la
aplicación local también se necesitan Docker y Docker Compose.

## Despliegue con un solo comando

Antes del primer arranque, crea el `.env` local para Compose y completa los
dos valores de MinIO con credenciales locales:

```bash
cp .env.example .env
chmod 600 .env
```

`ANTHROPIC_API_KEY` puede permanecer vacío si no vas a usar el chat Copilot.
No pongas credenciales AWS en este archivo.

```bash
docker compose up --build
```

Este comando levanta los servicios de MariaDB, MinIO, backend, frontend,
pipeline `app` y Copilot. El backend espera a que MariaDB y MinIO estén listos, aplica las
migraciones y siembra únicamente las categorías `dog` y `cat` antes de
arrancar; no crea imágenes demo ni hace falta ejecutar otro paso manual.

| Servicio        | URL                              |
|-----------------|-----------------------------------|
| Frontend        | http://localhost:8080            |
| Backend (API)   | http://localhost:3100            |
| Consola MinIO   | http://localhost:9001 (minioadmin/minioadmin) |

Para apagar normalmente los servicios, sin borrar los datos persistidos:

```bash
docker compose down
```

`docker compose down -v` elimina también los volúmenes de MariaDB y MinIO.
Úsalo únicamente cuando quieras reiniciar desde cero los datos locales.

Las credenciales de MariaDB/MinIO usadas en `docker-compose.yml` son las de
desarrollo del proyecto; para un despliegue real, cámbialas ahí antes de
publicar los puertos a una red no confiable.

## Desarrollo local sin Docker para las apps

Para iterar con hot reload en backend y frontend, puedes levantar solo la
infraestructura con Docker y correr los paquetes Node directamente en tu
máquina:

### 1. Infraestructura

```bash
docker run --name proyecto1-mariadb \
  -e MARIADB_ROOT_PASSWORD=password \
  -e MARIADB_DATABASE=image_repo \
  -p 3306:3306 -d mariadb:11

docker run --name proyecto1-minio \
  -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  -d quay.io/minio/minio server /data --console-address ":9001"
```

El bucket se crea automáticamente al arrancar el backend.

### 2. Backend

Antes de ejecutar esos comandos, crea `backend/.env` con la configuración de
desarrollo siguiente. Este archivo es distinto del `.env` de la raíz que usa
Docker Compose:

```dotenv
DATABASE_URL=mysql://root:password@localhost:3306/image_repo
MINIO_ENDPOINT=localhost
MINIO_PORT=9000
MINIO_USE_SSL=false
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=image-annotations
MAX_UPLOAD_SIZE_BYTES=5242880
```

No pongas credenciales AWS en `backend/.env` tampoco. Si cambias el puerto
publicado de MariaDB, ajusta `DATABASE_URL` en este archivo.

```bash
cd backend
npm ci
npm run db:migrate
npm run db:seed
npm run dev
```

Queda escuchando en `http://localhost:3000`.

Si el puerto 3306 ya está ocupado en tu máquina, publica MariaDB en otro
puerto (por ejemplo `-p 3307:3306`) y ajusta `DATABASE_URL` en `backend/.env`.
Nada está fijo en el código: puertos, credenciales y bucket salen del `.env`.

### 3. Frontend

```bash
cd frontend
npm ci
cp .env.example .env
npm run dev
```

Queda escuchando en `http://localhost:5173` y consume la API del backend a
través del proxy `/api` configurado en `vite.config.ts`.

### 4. Comprobación

```bash
curl http://localhost:3000/health
```

Respuesta esperada:

```json
{"status":"ok","database":"connected","timestamp":"..."}
```

## Producción

La forma recomendada de desplegar es `docker compose up --build` (ver
[Despliegue con un solo comando](#despliegue-con-un-solo-comando)): construye
las imágenes de backend y frontend y levanta MariaDB y MinIO junto con ellas.

Si necesitas correr el backend fuera de Docker contra tu propia
infraestructura:

```bash
cd backend
npm run build
npm run start:prod
```

El servidor de producción escucha en `http://localhost:3100`. El script usa
`cross-env`, por lo que funciona igual en Windows, macOS y Linux.

La plantilla `.env.production.example` contiene la configuración de
producción, con `PORT=3100`.

## Variables de entorno

Se copian de `.env.example`. Ningún valor real se versiona: `.gitignore`
ignora todo `.env*` salvo las plantillas de ejemplo.

| Variable                | Propósito                                      |
|-------------------------|------------------------------------------------|
| `PORT`                  | Puerto HTTP (3000 desarrollo, 3100 producción) |
| `DATABASE_URL`          | Cadena de conexión a MariaDB                   |
| `MINIO_ENDPOINT`        | Host de MinIO                                  |
| `MINIO_PORT`            | Puerto de la API de MinIO                      |
| `MINIO_USE_SSL`         | `true` o `false`                               |
| `MINIO_ACCESS_KEY`      | Credencial de acceso                           |
| `MINIO_SECRET_KEY`      | Credencial secreta                             |
| `MINIO_BUCKET`          | Bucket donde se guardan las imágenes           |
| `MAX_UPLOAD_SIZE_BYTES` | Tamaño máximo por imagen (5 MiB por defecto)   |

Los `.env` son configuración local de Compose/backend/MinIO/Copilot. Las
credenciales AWS se obtienen mediante el perfil SSO `mlops-p2`; nunca las
copies a un `.env`.

## API

| Método | Ruta                        | Descripción                                 |
|--------|-----------------------------|---------------------------------------------|
| GET    | `/health`                   | Estado del servicio y de la base de datos   |
| POST   | `/images`                   | Sube una imagen (`multipart/form-data`)     |
| GET    | `/images/search`            | Búsqueda con filtros y paginación           |
| DELETE | `/images/:id`               | Elimina imagen, binario y anotaciones       |
| GET    | `/images/:id/file`          | Sirve el binario desde MinIO                |
| PATCH  | `/images/:id/status`        | Transiciona el estado de anotación          |
| GET    | `/images/:id/annotations`   | Cajas de una imagen, con su categoría       |
| POST   | `/images/:id/annotations`   | Crea una bounding box                       |
| PATCH  | `/annotations/:id`          | Mueve, redimensiona o reclasifica una caja  |
| DELETE | `/annotations/:id`          | Elimina una caja                            |
| GET    | `/categories`               | Categorías disponibles con su color         |
| GET    | `/dashboard/summary`        | Métricas calculadas en SQL                  |
| GET    | `/export/coco`              | Descarga el dataset en formato COCO         |

### Búsqueda

`GET /images/search` acepta:

| Query param         | Descripción                                                |
|---------------------|------------------------------------------------------------|
| `q`                 | Clases con operadores, ej. `car AND person`, `car OR dog`   |
| `categories`        | Ids de categoría separados por coma                        |
| `status`            | `pending`, `in_progress`, `completed` (separados por coma)  |
| `dateFrom`/`dateTo` | Rango sobre la fecha de subida                             |
| `page`/`pageSize`   | Paginación                                                 |

Los operadores se resuelven con subconsultas `EXISTS` en SQL, nunca filtrando
en memoria. Con `AND` la imagen debe contener todas las clases; con `OR`, al
menos una. Mezclar `AND` con `OR` devuelve `400`, porque la precedencia
sería ambigua.

```bash
curl "http://localhost:3000/images/search?q=car%20AND%20person&status=pending&page=1&pageSize=24"
```

### Exportación COCO

```bash
curl -O -J http://localhost:3000/export/coco
```

```json
{
  "images":      [{ "id", "file_name", "width", "height" }],
  "annotations": [{ "id", "image_id", "category_id",
                    "bbox": [x, y, width, height],
                    "area", "iscrowd", "segmentation" }],
  "categories":  [{ "id", "name" }]
}
```

El `bbox` va en píxeles absolutos, `area` es coherente con `width × height`,
e `iscrowd` siempre está presente. Los `id` son consistentes entre las tres
secciones.

## Calidad

Desde `backend/`:

```bash
npm run typecheck   # TypeScript en modo strict
npm run lint        # Biome: cero errores y cero advertencias
npm test            # Vitest
npm run build       # Compilación a dist/
```

Desde `frontend/`:

```bash
npm run typecheck
npm run lint        # Biome: cero errores y cero advertencias
npm run build
```

## Especificaciones y pruebas

Cada regla crítica está trazada de la especificación al escenario Gherkin y
de ahí a la prueba automatizada.

| SPEC            | Regla                                   | Implementación               |
|-----------------|-----------------------------------------|------------------------------|
| SPEC-UPLOAD-001 | Tipo y tamaño de la imagen subida       | `image-upload.validation.ts` |
| SPEC-ANNOT-001  | Geometría y categoría de las cajas      | `annotation.validation.ts`   |
| SPEC-COCO-001   | Estructura y consistencia del JSON COCO | `coco-export.builder.ts`     |
| SPEC-SEARCH-001 | Operadores `AND` / `OR` de búsqueda     | `search-query.parser.ts`     |
| SPEC-VALID-001  | Validación de la frontera HTTP con Zod  | `annotation.validation.ts`   |
| SPEC-DASH-001   | Métricas del dashboard desde SQL        | `dashboard.builder.ts`       |

```text
backend/specs/<nombre>.spec.md
        ↓
backend/features/<nombre>.feature    (Given / When / Then)
        ↓
backend/tests/<nombre>.test.ts       (Vitest)
        ↓
backend/src/logic/<nombre>.ts        (implementación)
```

Las pruebas están diseñadas para fallar si la lógica se rompe: invertir
`width` y `height` en la exportación COCO, permitir un `categoryId` no
positivo o dejar de validar `imageId` hace fallar la suite.

## Fuera de alcance

El entrenamiento del modelo y MLOps corresponden a una fase posterior.

## Etapas del proyecto

El proyecto se construyó por etapas, cada una sobre la anterior:

| Etapa | Qué aportó                                                                 |
|-------|----------------------------------------------------------------------------|
| 1     | Esqueleto: TypeScript, Biome, arquitectura UI/Logic/Data, esquema Drizzle. |
| 2     | Persistencia: MariaDB, MinIO, migraciones, upload de imágenes, seeder.     |
| 3     | Frontend React: portal de anotación, canvas, dashboard y búsqueda.         |
| 4     | Integración final: lógica de negocio, COCO, dashboard y validación Zod.    |

### Qué agrega la etapa final (integración)

Esta etapa conecta el frontend con el backend y completa lo que faltaba para
que el portal funcione de punta a punta:

- **Exportación COCO** (`GET /export/coco`): documento JSON descargable con
  `images`, `annotations` y `categories`, con ids consistentes entre
  secciones (SPEC-COCO-001).
- **Métricas del dashboard** (`GET /dashboard/summary`): totales, objetos por
  clase y progreso de anotación, todo calculado en SQL (SPEC-DASH-001).
- **Búsqueda por clases con operadores** en `GET /images/search`: `AND` / `OR`
  resueltos con subconsultas `EXISTS` en SQL, más filtros por categoría,
  estado y rango de fechas (SPEC-SEARCH-001).
- **Validación de la frontera HTTP con Zod**: todo body, query param y route
  param se valida antes de llegar a la base de datos, con errores tipados que
  la UI mapea a códigos HTTP (SPEC-VALID-001).
- **Reglas de anotación**: la caja debe caber dentro de la imagen, el área la
  calcula el backend, y una imagen sin cajas no puede quedar como completada
  (SPEC-ANNOT-001).

### Notas de puesta en marcha

- Usa `npm install` la primera vez en cada paquete (`backend/` y `frontend/`).
  `node_modules` no se versiona: se reconstruye desde `package-lock.json`.
- El backend valida sus variables de entorno al arrancar (fail-fast con Zod).
  Si falta `backend/.env` o alguna variable, el proceso termina indicando
  cuáles faltan; usa el bloque de variables de backend documentado arriba.
- Si publicaste MariaDB en un puerto distinto al 3306 (por ejemplo 3307
  porque el 3306 ya estaba ocupado), ajusta `DATABASE_URL` en `backend/.env`
  para que coincida.
- El frontend habla con el backend a través del proxy `/api` de Vite en
  desarrollo. `VITE_API_BASE_URL` puede dejarse en `/api`; en producción se
  apunta a la URL real del backend.

## P2-04 — MinIO local y remotes DVC

Esta sección contiene los detalles del remote DVC opcional de desarrollo. Para
el onboarding completo, empieza por [Onboarding de desarrollo](#onboarding-de-desarrollo).
No necesitas MinIO para leer el dataset compartido de producción.

- `dev` usa `s3://dvc-cache` con endpoint `http://localhost:9000` (MinIO local).
- `prod` usa `s3://mlops-p2-dvc-cache` en AWS S3.
- `mlops-p2-dataset-releases` se reserva para releases finales del dataset; no es un remote DVC.

### Resumen rápido

- `dev` → MinIO local, bucket `dvc-cache`.
- `prod` → AWS S3, bucket `mlops-p2-dvc-cache`.
- `mlops-p2-dataset-releases` → releases finales del dataset.
- Git versiona la configuración y los archivos `.dvc`; los binarios se guardan en los remotes.
- Las credenciales de MinIO son locales de cada integrante; no son credenciales
  de AWS ni se usan para `prod`.
- Cada integrante necesita su propio acceso SSO a AWS para `prod`.
- No se comparten contraseñas, sesiones SSO, access keys, secret keys ni tokens.

### Flujo opcional: preparar MinIO local

Solo realiza estos pasos si necesitas usar el remote `dev`. Desde la raíz del
proyecto, crea tu archivo `.env` local a partir de la plantilla:

```bash
test -e .env || cp .env.example .env
chmod 600 .env
```

Completa:

```text
MINIO_ROOT_USER=
MINIO_ROOT_PASSWORD=
```

con valores locales propios.

Puedes generar una contraseña con:

```bash
openssl rand -hex 32
```

No uses claves AWS en `.env`. La autenticación de AWS se configura con IAM
Identity Center / SSO y el perfil local `mlops-p2`.

`frontend/.env.example` es independiente y no cambia para este ticket.

Levanta MinIO:

```bash
docker compose up -d --no-deps minio
```

### Instalar DVC

Instala DVC con soporte S3 en un entorno Python separado del entorno de `app/`:

```bash
python3.12 -m venv .venv-dvc
. .venv-dvc/bin/activate
python -m pip install 'dvc[s3]==3.67.1'
python --version
dvc --version
dvc remote list
```

El repositorio ya contiene la inicialización de DVC y la configuración de los
remotes. **No ejecutes `dvc init`.**

### Configurar `dev` con MinIO local

Carga las variables de tu `.env`:

```bash
set -a
. ./.env
set +a
```

Configura las credenciales de MinIO únicamente de forma local y solo para
`dev`:

```bash
dvc remote modify --local dev access_key_id "$MINIO_ROOT_USER"
dvc remote modify --local dev secret_access_key "$MINIO_ROOT_PASSWORD"
chmod 600 .dvc/config.local
```

No omitas `--local`: `.dvc/config.local` es local, está ignorado por Git y no
debe subirse al repositorio. No mezcles estas credenciales con el perfil SSO
de AWS usado por `prod`.

No agregues `.env` ni `.dvc/config.local` a Git.

### Crear el bucket local `dvc-cache`

Si el bucket `dvc-cache` todavía no existe en MinIO, créalo con:

```bash
python - <<'PY'
import os
from botocore.session import get_session
from botocore.exceptions import ClientError

client = get_session().create_client(
    's3',
    endpoint_url='http://localhost:9000',
    region_name='us-east-1',
    aws_access_key_id=os.environ['MINIO_ROOT_USER'],
    aws_secret_access_key=os.environ['MINIO_ROOT_PASSWORD'],
)

try:
    client.head_bucket(Bucket='dvc-cache')
except ClientError as error:
    if error.response['ResponseMetadata']['HTTPStatusCode'] != 404:
        raise
    client.create_bucket(Bucket='dvc-cache')

print('Bucket local dvc-cache disponible')
PY
```

Este paso solo opera contra MinIO local en `localhost:9000`.

No modifica el bucket `image-annotations` usado por el portal.

### Verificar los remotes

Ejecuta:

```bash
dvc remote list -v
```

La salida debe incluir:

```text
dev     s3://dvc-cache
prod    s3://mlops-p2-dvc-cache
```

### Subir y bajar archivos con `dev`

Primero registra el archivo o directorio con DVC:

```bash
dvc add ruta/al/dataset
```

Para subirlo a MinIO:

```bash
dvc push -r dev
```

Para recuperarlo:

```bash
dvc pull -r dev
```

Los archivos `.dvc` generados se comparten mediante Git.

Los binarios se guardan en MinIO, no directamente en GitHub.

### AWS S3 y remote `prod`

La configuración completa de AWS CLI, IAM Identity Center / SSO, el perfil
`mlops-p2`, las comprobaciones de lectura y la conexión local de DVC está en
[Onboarding de desarrollo](#onboarding-de-desarrollo). `prod` usa AWS S3 real,
sin endpoint personalizado, en `us-east-1`:

| Bucket | Uso |
|---|---|
| `mlops-p2-dvc-cache` | Remote DVC `prod`. |
| `mlops-p2-dataset-releases` | Releases finales del dataset. |

Para lectura del remote DVC, el permission set debe tener permisos
conceptualmente equivalentes a `s3:ListBucket` y `s3:GetObject`. Si además
publicas datasets, el administrador debe asignarte permisos de escritura. No
uses `dvc push` como prueba de conexión.

### Flujo recomendado para el equipo

1. Hacer `git pull` para obtener los metadatos `.dvc` más recientes.
2. Activar el entorno de DVC.
3. Iniciar sesión con AWS SSO si se va a usar `prod`.
4. Ejecutar `dvc status -r prod data/raw/images.dvc data/raw/annotations.dvc`
   para consultar el estado sin subir datos.
5. Ejecutar `dvc pull -r prod data/raw/images.dvc data/raw/annotations.dvc`
   solo si necesitas materializar el dataset localmente.
6. Si eres una persona mantenedora autorizada, agregar o actualizar datos,
   ejecutar `dvc add <ruta>` y publicar con `dvc push -r prod` como una
   operación explícita, no como prueba de conexión.
7. Versionar con Git los archivos `.dvc` y los cambios de código
   correspondientes.

No subas los binarios grandes directamente al repositorio de GitHub.

### Seguridad y validación

Los siguientes archivos o datos no deben versionarse:

- `.env`
- `.dvc/config.local`
- credenciales AWS, incluidas access keys y secret keys
- sesiones y tokens SSO
- credenciales reales de MinIO

Cada persona debe usar su propia identidad AWS. No compartas passwords,
sesiones SSO, access keys ni tokens, y no pongas credenciales AWS en ningún
`.env`.

Comprueba que los archivos privados estén ignorados:

```bash
git check-ignore .env .dvc/config.local
```

La salida debe incluir:

```text
.env
.dvc/config.local
```

Comprueba que `.env.example` sí pueda versionarse:

```bash
git check-ignore .env.example
```

Ese comando no debe mostrar salida.

Comprueba que no existan access keys AWS con prefijo `AKIA` en el historial:

```bash
git log --all -p -S 'AKIA'
```

La salida debe estar vacía.

P2-04 externaliza las credenciales MinIO usadas por Docker Compose y evita agregar secretos AWS al repositorio.

No se reescribe el historial de Git ni se modifica la configuración heredada de MariaDB.

### Criterios de aceptación

Antes de cerrar P2-04, verificar:

- `docker compose up -d --no-deps minio` levanta MinIO.
- `.env.example` existe y no contiene credenciales reales.
- `dvc remote list -v` muestra `dev` y `prod`.
- `dvc push -r dev` funciona para una persona autorizada que use MinIO local.
- `aws s3api head-bucket`, `aws s3api list-objects-v2` y `dvc status -r prod`
  funcionan con el perfil SSO autorizado; `dvc pull` se usa solo cuando se
  necesita materializar el dataset.
- `dvc push -r prod` se prueba únicamente con autorización explícita de
  escritura; no es una prueba de conectividad.
- `git log --all -p -S 'AKIA'` no devuelve resultados.
- `.env` y `.dvc/config.local` permanecen fuera de Git.
- Los buckets `mlops-p2-dvc-cache` y `mlops-p2-dataset-releases` existen en AWS.

## P2-42 — Pipeline DVC completo (`dvc.yaml`)

Hasta este ticket, el dataset se manejaba con `dvc add` suelto: reproducible
como almacenamiento de archivos, pero sin un pipeline declarado con
dependencias/salidas. `dvc.yaml` separa el cálculo del reporte (`quality_report`)
de la decisión de la compuerta (`quality_gate`) y agrega `split` como etapa
posterior. Un reporte `failed` hace que DVC termine con código distinto de cero
y evita ejecutar las etapas dependientes.

```bash
dvc repro
```

- **`app/dvc_quality_report_stage.py`** calcula `reports/quality.json` sin
  decidir si el dataset puede avanzar. **`app/dvc_gate_stage.py`** lee ese
  reporte y devuelve `exit 1` cuando `status: failed`; solo en caso aprobado
  escribe `reports/.quality_gate.passed`, que es la dependencia explícita de
  `split`.
- **`reports/quality.json` es un `metrics`, no un `outs`**, con
  `cache: false`: es un reporte chico y legible, pensado para diffs de PR y
  `dvc metrics diff`, no un artefacto binario que amerite el object store
  de DVC.
- El pipeline no usa `always_changed`: una segunda ejecución de `dvc repro`
  puede reutilizar el run-cache y no rehacer etapas cuando sus entradas no
  cambiaron.
- Los remotes `dev`/`prod` de P2-04 ya existían; lo que faltaba en un
  checkout nuevo era el paso local `dvc remote modify --local dev
  access_key_id/secret_access_key` (con `$MINIO_ROOT_USER`/
  `$MINIO_ROOT_PASSWORD`) y crear el bucket `dvc-cache` si no existía —
  ambos ya documentados arriba en P2-04, solo faltaba ejecutarlos en este
  checkout.

### Criterios de aceptación

- `dvc.yaml` define `quality_report`, `quality_gate` y `split` con dependencias
  y salidas reales; un `failed` bloquea el downstream.
- `dvc.lock` y `dvc.yaml` versionados en Git; los datos siguen fuera de Git.
- `dvc repro` regenera `reports/quality.json`; si el reporte queda `failed`,
  termina con código distinto de cero y no ejecuta `split`.
- `dvc push`/`dvc pull` funcionan contra `dev` y `prod` (verificado: 612
  archivos sincronizados en `dev`, `prod` ya en uso durante todo el proyecto).

## P2-45 — Versionado semántico y diff entre releases

Depende de P2-42. Un release congela `dataset_version` (formato
`vMAJOR.MINOR.PATCH`) junto con su `quality.json` y `splits.json` bajo
`reports/releases/<version>/`, y agrega la entrada al catálogo
`reports/versions.json` (`VersionsReport`, contrato v1.0 de P2-12).

```bash
# Desde app/, con las mismas variables placeholder que P2-42 (ver dvc_gate_stage.py):
uv run python -m presentation.release cut v0.1.0
uv run python -m presentation.release diff v0.1.0 v0.2.0
```

- **Content hash DEV/PROD**: el hash del dataset ya es el md5 en
  `data/raw/annotations.dvc`/`data/raw/images.dvc` — el mismo valor sin
  importar el remote, por construcción de DVC. Verificar que ambos
  remotes lo tengan de verdad es `dvc status -r dev` y `dvc status -r
  prod`, ambos reportando "Cache and remote 'X' are in sync." — no hace
  falta recalcular nada; reimplementarlo sería redundante con lo que DVC
  ya garantiza.
- **`splits.json` nunca se había escrito a disco**: P2-32 dejó
  `split_dataset()`/`build_splits_report()` puros a propósito (ver
  `app/splits/README.md`, "antes de persistir artefactos hay que acordar
  su ubicación/ignore o seguimiento DVC"), porque `DatasetRelease` exige
  `quality_file` y `splits_file`, este ticket fue quien tuvo que decidirlo:
  `reports/releases/<version>/splits.json`, escrito por `cut_release()`.
- **El release respeta la compuerta**: el catálogo histórico `v0.1.0` puede
  conservar un reporte `failed`, pero `cut_release()` rechaza cualquier nuevo
  corte cuyo reporte esté en `status: failed`.
- **Un release es inmutable**: `cut_release()` rechaza un `version` que ya
  existe en el catálogo en vez de sobreescribirlo.
- **`diff_releases()` no vuelve a correr el gate**: lee los dos
  `quality.json` ya congelados — un diff no debe poder ver un dataset
  distinto al que el release realmente describió en su momento.

### Criterios de aceptación

- `dvc status -r dev` y `dvc status -r prod` reportan "in sync" (content
  hash idéntico, verificado).
- `reports/versions.json` sigue el contrato `VersionsReport` v1.0 y usa
  versionado semántico (`v0.1.0` histórico y `v0.1.1` válido, ver
  `reports/releases/`).
- `presentation.release diff <a> <b>` genera un diff real entre dos
  releases (conteo por categoría y status de cada check).

## Frente 1 — Arquitectura y entorno del pipeline de calidad

El portal de anotación (arriba) ya no es el entregable de la Fase 2: es la
fuente del COCO crudo. El entregable es un pipeline en Python que mide la
calidad de ese COCO, decide si se libera y versiona el resultado con DVC.

Este frente deja listo el esqueleto; la lógica de cada tier la completan los
frentes 2 a 6.

### Capas (`app/`)

El pipeline vive en `app/`, como paquete Python independiente (hermano de
`backend/` y `frontend/`), con una carpeta por capa:

```text
app/
  ingestion/      Tier 1 — COCO crudo del Proyecto 1
  analyzers/      Tier 2 — 5 analizadores de calidad (objetos pequeños,
                  desbalance, duplicados, cajas inválidas, sesgo espacial)
  policies/       Tier 3 — compuerta de calidad (policies/quality.yaml)
  splits/         Tier 4 — split estratificado train/val/test
  storage/        Tier 5 — MariaDB y MinIO/S3 (DVC)
  presentation/   Expone los resultados a la app web y al Dataset Copilot
```

**Regla de la compuerta de acoplamiento:** solo `storage/` importa `os`
(para leer variables de entorno), crea clientes `boto3`/`Minio(` o abre un
`create_engine`/`pymysql.connect`. Todas las demás capas son funciones puras
que reciben los datos ya cargados como argumento — así se pueden probar con
`pytest` sin levantar MariaDB/MinIO reales. Se verifica con:

```bash
grep -rn "os\.environ\|os\.getenv\|boto3\.client\|Minio(\|create_engine\|pymysql\.connect" app/analyzers/
```

(sin resultados) y con `app/tests/test_architecture.py`, que corre lo mismo
en CI.

### Levantar todo

```bash
docker compose up
```

Además de `mariadb`, `minio`, `backend` y `frontend` (portal P1, se mantiene
porque la cola de re-anotación —cuando la compuerta bloquea el release—
ocurre ahí), se agrega el servicio `app`: el pipeline Python, que reutiliza
el mismo MariaDB y el mismo MinIO del portal (mismas credenciales de
`.env`, sin variables nuevas). Al arrancar, `app` valida que puede
conectarse a ambos, ejecuta el quality gate y puede regenerar
`reports/quality.json`. Un estado `failed` queda registrado en los logs y debe
revisarse antes de promover o publicar el dataset; levantar `app` no sustituye
la compuerta de DVC.

El servicio `copilot` (P2-52) usa la misma imagen que `app` y atiende el chat
de la pantalla Copilot a través de nginx (`/copilot-api/`), sin publicar
puertos. Necesita `ANTHROPIC_API_KEY` en `.env` (opcional: sin ella todo
arranca y el chat explica qué falta). Detalles en `app/copilot/README.md`.

### Python y lockfile

`app/pyproject.toml` fija `requires-python = "==3.12.*"` y `app/Dockerfile`
usa `python:3.12-slim` — misma versión en ambos lados. Las dependencias
quedan resueltas y pineadas en `app/uv.lock` (generado con `uv lock`, no a
mano); el `Dockerfile` instala desde ese lockfile con
`uv sync --locked`, así que build local y build en CI siempre resuelven
exactamente las mismas versiones.

Para trabajar en `app/` localmente con [uv](https://docs.astral.sh/uv/):

```bash
cd app
uv sync            # crea .venv con dependencias + grupo dev (ruff, pytest)
uv run pytest -q
uv run ruff check .
```

### Supuestos de este frente pendientes de confirmar con Karen/Heri

Lo siguiente se infirió a partir del diagrama de tiers y los mockups del
profe, y del trabajo ya mergeado de DVC (P2-04); si Karen decide otra cosa,
son fáciles de mover porque todo el pipeline está aislado en `app/`:

- El portal Node (`backend`/`frontend`) se queda corriendo junto al pipeline
  en el mismo `docker-compose.yml`, en vez de retirarse porque "el portal ya
  no es el entregable".
- El pipeline reutiliza el MariaDB/MinIO del portal (misma base
  `image_repo`, mismo bucket `image-annotations`) en vez de tener su propia
  infraestructura de datos en dev.
- Quién es responsable del frente 4 (compuerta) no estaba claro en el
  reparto compartido — confirmar con Karen.

## P2-36 — Settings persistente

`/pipeline/settings` tiene dos formularios independientes. La API Node
existente ofrece `GET /settings`, `PUT /settings/quality` y
`PUT /settings/splits` (desde el navegador, `/api/settings/...`).

- Quality permite editar threshold/action de los siete checks reales y
  width_px/height_px de objetos pequeños. La similitud pHash es un umbral de
  detección; el cumplimiento sigue exigiendo cero pares.
- Splits permite editar train/val/test (fracciones estrictamente entre 0 y 1,
  suma 1 con tolerancia 1e-6) y seed (entero seguro de JavaScript).
- `cross_split_leakage` se preserva sin exponerlo. No se publican credenciales,
  rutas, variables de infraestructura ni una supuesta versión activa.
- GET devuelve `{quality, splits}`. Cada PUT recibe directamente su sección
  completa y devuelve esa sección validada. Campos desconocidos o valores
  inválidos producen 400; los errores internos producen 500.
- Persisten en `app/policies/quality.yaml` y `app/splits/splits.yaml`. Se
  conservan comentarios y campos no editables; el backend escribe un temporal,
  sincroniza/cierra y renombra en el mismo directorio. Serializa escrituras
  dentro de su proceso. No hay transacción entre ambos archivos ni control de
  edición obsoleta: la última escritura válida gana.

Guardar **no ejecuta el pipeline**, no crea releases y no cambia reportes
existentes. Los valores afectan la siguiente ejecución de quality/release.
El contenedor Python ejecuta el gate al arrancar, no observa archivos para
recalcular automáticamente. Los comandos de pipeline/release existentes
siguen siendo operaciones explícitas.

Compose comparte los directorios de políticas y splits: backend RW bajo
`/pipeline`, Python RO bajo `/app`. No se publican mediante Nginx. Se montan
directorios para que los reemplazos atómicos sean visibles. En desarrollo,
el backend resuelve el directorio `app/` hermano; `PIPELINE_CONFIG_ROOT` es
una opción de despliegue confiable, nunca un parámetro HTTP.

Python carga la política YAML al construir Settings. `QUALITY` del entorno
o de `.env` se ignora, incluso si contiene JSON inválido: no puede sustituir
silenciosamente la política gestionada por UI. El resto de variables de
infraestructura conserva su semántica. Una política inyectada explícitamente
por código sigue siendo válida para tests/operaciones explícitas.
Cada release recibe una política para quality y pHash, y una SplitsConfig
cargada una vez; ya no recarga otra política para agrupar duplicados.

Estos YAML siguen siendo configuración versionada en Git; guardar puede
dejar cambios locales que deben revisarse. DVC observa `policies/` y
`splits/splits.yaml` desde `quality_report`/`split`. Desde P2-53, quality_gate
también registra ratios y seed de splits. Cambiarlos
afecta la próxima evaluación de leakage y el próximo corte de release,
sin reescribir los splits congelados.

Se añadió `yaml` como dependencia directa del backend para leer/escribir
YAML sin un parser artesanal. El backend actual no tiene autenticación ni
autorización: esta edición está destinada al despliegue controlado existente,
no constituye un panel administrativo protegido para exposición pública.
