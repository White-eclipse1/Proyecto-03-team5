# SPEC-DASH-001 ÔÇö M├®tricas del dashboard

## Objetivo

Exponer en `GET /dashboard/summary` las m├®tricas del dataset calculadas
**desde la base de datos** con consultas agregadas, nunca con valores fijos
en el c├│digo ni filtrando en memoria.

## Contrato

```json
{
  "imagesUploaded":  0,
  "imagesAnnotated": 0,
  "boundingBoxes":   0,
  "categoriesCount": 0,
  "objectsPerClass": [{ "categoryId": 1, "name": "car", "color": "#3498DB", "count": 4 }],
  "annotationProgress": { "annotated": 0, "pending": 0 },
  "recentUploads": [{ "id": 1, "thumbnailUrl": "/images/1/file", "status": "pending" }]
}
```

## Reglas

1. `imagesUploaded` es el total de im├ígenes registradas, sumando todos los
   estados.
2. `imagesAnnotated` cuenta ├║nicamente las im├ígenes en estado `completed`.
3. `boundingBoxes` es el total de anotaciones del dataset.
4. `categoriesCount` es el total de categor├¡as registradas.
5. `objectsPerClass` agrupa las anotaciones por categor├¡a, con el `name` y el
   `color` persistidos, ordenadas de mayor a menor conteo. Una categor├¡a sin
   anotaciones no aparece.
6. `annotationProgress.annotated` son las im├ígenes `completed`;
   `annotationProgress.pending` agrupa `pending` + `in_progress`, es decir
   todo lo que falta por terminar.
7. La suma `annotated + pending` debe ser igual a `imagesUploaded`.
8. `recentUploads` devuelve las im├ígenes m├ís recientes por `created_at`, con
   la URL del binario servida por el backend (nunca una URL de MinIO).
9. Todos los conteos provienen de `COUNT` / `GROUP BY` en SQL.

## Flujo esperado

UI (`GET /dashboard/summary`) ÔåÆ Logic (`getDashboardSummary`) ÔåÆ Data (agregados SQL)
