# SPEC-VALID-001 — Validación de la frontera HTTP con Zod

## Objetivo

Garantizar que **todo** dato que entra por HTTP (body, query params y route
params) se valide con Zod **antes** de alcanzar la capa de datos, y que los
errores devuelvan el código HTTP correcto con un mensaje accionable.

## Problema que resuelve

Los esquemas de route param (`idParamSchema`) y de búsqueda
(`imageSearchSchema`) se escribieron y se probaron unitariamente, pero nunca
se conectaron a las rutas reales de `src/ui/server.ts`: cada ruta parseaba el
`:id` a mano con `Number.parseInt`, que **trunca en vez de rechazar** un
valor decimal. Ejemplo reproducido:

| Petición                          | Respuesta anterior                          | Problema                                   |
|------------------------------------|----------------------------------------------|---------------------------------------------|
| `GET /images/1.5/annotations`      | `200` con las anotaciones de la imagen **1** | `parseInt("1.5", 10)` trunca a `1`, no rechaza |
| `GET /images/search?status=bogus`  | `200` ignorando el filtro inválido           | El filtro se armaba a mano, sin Zod          |

Un dato mal formado debe producir `400 Bad Request` (petición inválida), no
un `200`/`404` que oculta el error.

## Reglas

1. **Todo body y route param se parsea con Zod** antes de usarse. Ningún
   servicio ni ruta recibe un dato externo con un tipo declarado sin
   verificar en tiempo de ejecución; se acepta `unknown` y se hace
   `safeParse`.
2. **Prohibido usar aserciones de tipo** (`as`) para describir datos externos.
   La forma de un dato externo se establece parseando, no afirmando.
3. **Los route params se validan con Zod.** Un `:id` no numérico, decimal,
   cero o negativo devuelve `400` (nunca se trunca ni se redondea).
4. **Los tipos se infieren del esquema** con `z.infer<>`; no se declaran
   interfaces duplicadas a mano para datos externos.
5. **Semántica de errores:**
   - Dato mal formado o regla de negocio violada → `400 Bad Request`
   - Recurso inexistente (id válido que no está en BD) → `404 Not Found`
   - Archivo que excede la cuota → `413 Payload Too Large`
6. La validación de forma ocurre **antes** de cualquier consulta a la base de
   datos o a MinIO.
7. **Un esquema exportado que ningún endpoint usa es un bug.** Si cambia el
   diseño de una ruta, el esquema que la validaba se actualiza o se borra en
   el mismo cambio — no se deja como código muerto "por si acaso".

## Taxonomía de errores

Para que la capa UI no dependa de comparar cadenas de texto, la capa Logic
lanza errores tipados:

| Clase             | Significado                                | HTTP |
|-------------------|----------------------------------------------|------|
| `ValidationError` | Dato inválido o regla de negocio violada     | 400  |
| `NotFoundError`   | El recurso no existe en la base de datos     | 404  |

La capa UI mapea la clase del error al código HTTP. No inspecciona mensajes.

## Fronteras cubiertas

| Frontera                            | Esquema Zod                     | Ubicación                    |
|--------------------------------------|----------------------------------|-------------------------------|
| Variables de entorno                 | `envSchema`                      | `src/config/env.ts`           |
| Archivo subido (MIME y tamaño)       | `imageUploadSchema`              | `image-upload.validation.ts`  |
| Body de `POST /images/:id/annotations` | `createAnnotationForImageSchema` | `annotation.validation.ts`  |
| Body de `PATCH /annotations/:id`     | `patchAnnotationSchema`          | `annotation.validation.ts`    |
| Body de `PATCH /images/:id/status`   | `imageStatusTransitionSchema`    | `annotation.validation.ts`    |
| Query params de `GET /images/search` | `imageSearchSchema`              | `annotation.validation.ts`    |
| Route param `:id` (imagen o anotación) | `idParamSchema`                 | `annotation.validation.ts`    |

Cada uno de estos esquemas se importa y se usa en `src/ui/server.ts` o en el
servicio de `src/logic/` que atiende la ruta correspondiente; no hay esquemas
exportados sin un punto de uso en producción.

## Flujo esperado

UI (dato crudo `unknown`) → Logic (`safeParse` + errores tipados) → Data
