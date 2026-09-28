# Image Annotation Repo

Monolito para un portal web de anotación de imágenes.

En esta etapa el backend permite almacenar imágenes en MinIO, guardar sus
metadatos en MariaDB y preparar la información necesaria para futuras
anotaciones y exportación al formato COCO.

> **Estado actual: Fase 2 — Backend / Data**

## Tecnologías

- TypeScript
- Node.js + Express
- Drizzle ORM
- MariaDB
- MinIO
- Zod 4
- Multer
- Sharp
- Vitest
- Biome

## Arquitectura

El proyecto mantiene separación estricta entre tres capas:

```text
UI → Logic → Data
              ├── MariaDB
              └── MinIO
```

- `src/ui`: endpoints HTTP con Express.
- `src/logic`: reglas de negocio y validaciones.
- `src/data`: acceso a MariaDB y MinIO.

La capa UI nunca accede directamente a MariaDB ni a MinIO.

## Estructura principal

```text
src/
├── config/
│   └── env.ts
├── ui/
│   └── server.ts
├── logic/
│   ├── health.service.ts
│   ├── startup.service.ts
│   ├── image-upload.validation.ts
│   ├── image-upload.service.ts
│   └── index.ts
└── data/
    ├── db/
    │   ├── client.ts
    │   ├── schema.ts
    │   └── migrations/
    ├── repositories/
    │   └── image.repository.ts
    ├── storage/
    │   ├── minio.client.ts
    │   └── minio.storage.ts
    ├── seed.ts
    └── index.ts

specs/
└── image-upload.spec.md

features/
└── image-upload.feature

tests/
└── image-upload.test.ts
```

## Instalación

Clona el repositorio e instala las dependencias:

```bash
npm ci
```

Crea tu archivo local de configuración:

```bash
cp .env.example .env
```

`.env` está ignorado por Git y no debe versionarse.

## Variables de entorno

Configuración de desarrollo:

```env
NODE_ENV=development
PORT=3000

DATABASE_URL=mysql://root:password@localhost:3306/image_repo

DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=password
DB_NAME=image_repo

MINIO_ENDPOINT=localhost
MINIO_PORT=9000
MINIO_USE_SSL=false
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=image-annotations

MAX_UPLOAD_SIZE_BYTES=5242880
```

Los valores reales deben almacenarse únicamente en `.env`.

Para producción existe `.env.production.example` y el servidor está
configurado para utilizar el puerto `3100`.

## Levantar MariaDB

Para desarrollo se puede utilizar Docker:

```bash
docker run --name proyecto1-mariadb \
  -e MARIADB_ROOT_PASSWORD=password \
  -e MARIADB_DATABASE=image_repo \
  -p 3306:3306 \
  -d mariadb:11
```

## Levantar MinIO

```bash
docker run --name proyecto1-minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  -d quay.io/minio/minio server /data --console-address ":9001"
```

MinIO utiliza:

```text
API:      http://localhost:9000
Consola:  http://localhost:9001
```

El backend comprueba automáticamente que el bucket configurado exista y lo
crea si es necesario.

No es necesario crear el bucket manualmente.

## Migraciones

Aplicar todas las migraciones versionadas:

```bash
npm run db:migrate
```

Las migraciones crean y actualizan las tablas:

```text
images
categories
annotations
```

El esquema incluye llaves primarias, llaves foráneas, restricciones UNIQUE e
índices para búsquedas y filtros.

## Seeder

Ejecutar:

```bash
npm run db:seed
```

El seeder crea únicamente las categorías `dog` y `cat`; no genera imágenes de
ejemplo ni objetos `sample-red.png`/`sample-blue.png` en MinIO.

Es idempotente, por lo que puede ejecutarse varias veces:

```bash
npm run db:seed
npm run db:seed
```

sin duplicar registros ni objetos en MinIO.

## Ejecutar en desarrollo

```bash
npm run dev
```

El servidor queda disponible en:

```text
http://localhost:3000
```

## Ejecutar en producción

Primero compila:

```bash
npm run build
```

Luego:

```bash
npm run start:prod
```

El servidor de producción utiliza:

```text
http://localhost:3100
```

## Endpoints actuales

### GET `/`

Información básica del proyecto.

### GET `/health`

Comprueba que el backend y MariaDB estén disponibles.

Ejemplo:

```bash
curl http://localhost:3000/health
```

### POST `/images`

Recibe una imagen mediante `multipart/form-data`.

El nombre del campo debe ser:

```text
image
```

Ejemplo:

```bash
curl -X POST \
  -F "image=@foto.png;type=image/png" \
  http://localhost:3000/images
```

Tipos permitidos:

```text
image/jpeg
image/png
image/webp
```

El tamaño máximo se configura mediante:

```text
MAX_UPLOAD_SIZE_BYTES
```

El flujo de persistencia es:

```text
HTTP Upload
    ↓
Multer
    ↓
Logic
    ├── Zod
    └── Sharp
    ↓
Data
    ├── MinIO   → archivo
    └── MariaDB → metadatos
```

Las dimensiones reales de la imagen son obtenidas con Sharp.

Si el archivo se guarda en MinIO pero falla la escritura en MariaDB, el
objeto de MinIO se elimina para evitar archivos huérfanos.

## Base de datos

### images

Guarda los metadatos de cada imagen.

El archivo no se almacena en MariaDB. `storage_key` referencia el objeto
correspondiente dentro de MinIO.

### categories

Guarda las clases disponibles para anotaciones y su color.

### annotations

Guarda las bounding boxes asociadas a una imagen y categoría.

El esquema utiliza la convención COCO:

```text
[x, y, width, height]
```

e incluye:

```text
area
iscrowd
```

para facilitar la futura exportación del dataset.

## Tests y TDD

Ejecutar:

```bash
npm test
```

Actualmente existen pruebas asociadas a:

```text
SPEC-UPLOAD-001
```

La trazabilidad es:

```text
specs/image-upload.spec.md
        ↓
features/image-upload.feature
        ↓
tests/image-upload.test.ts
        ↓
src/logic/image-upload.validation.ts
```

El historial de Git conserva evidencia del ciclo:

```text
RED → GREEN → REFACTOR
```

## Calidad de código

```bash
npm run typecheck
npm run lint
npm run format
npm run build
npm test
```

Antes de integrar cambios todos estos comandos deben terminar correctamente.

## Flujo completo desde cero

Con MariaDB y MinIO levantados:

```bash
npm ci
cp .env.example .env
npm run db:migrate
npm run db:seed
npm run dev
```

Después:

```bash
curl http://localhost:3000/health
```

debe responder correctamente.

## Alcance actual de la Fase 2

Implementado:

```text
Drizzle + MariaDB
Migraciones versionadas
MinIO
Creación automática del bucket
Upload de imágenes
Validación con Zod
Validación de imágenes con Sharp
Persistencia de archivos en MinIO
Persistencia de metadatos en MariaDB
Seeder idempotente
Tests con Vitest
SPEC + Gherkin + TDD
Puertos 3000 desarrollo / 3100 producción
```

Todavía fuera del alcance de esta etapa:

```text
Interfaz gráfica completa
Crear/mover/redimensionar bounding boxes
Dashboard
Búsqueda avanzada
Exportación COCO descargable
Entrenamiento del modelo
MLOps
```
