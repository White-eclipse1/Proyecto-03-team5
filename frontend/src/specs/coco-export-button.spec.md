# SPEC-COCO-UI-001 — Descarga del dataset COCO desde el Dashboard

## Objetivo

Permitir que el usuario descargue el dataset completo en formato COCO
desde el Dashboard utilizando el endpoint existente del backend.

## Reglas de negocio

1. El Dashboard debe mostrar una acción visible llamada "Exportar COCO".
2. La acción debe utilizar el endpoint GET /export/coco.
3. El frontend no debe construir el JSON COCO ni consultar MariaDB o MinIO.
4. La descarga debe delegarse completamente al backend.
5. El endpoint debe respetar la configuración de URL del API del frontend.

## Dependencia

La estructura y validez del archivo descargado está definida por
SPEC-COCO-001 en el backend.