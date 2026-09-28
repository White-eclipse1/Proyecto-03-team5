# SPEC-UPLOAD-001 — Carga de imágenes

## Objetivo

Permitir que el usuario cargue imágenes al portal de anotación de forma segura.

## Reglas de negocio

1. Solo se aceptan archivos de imagen con tipos MIME permitidos.
2. El archivo debe respetar el tamaño máximo configurado.
3. Una imagen válida debe almacenarse en MinIO.
4. Los metadatos de la imagen deben almacenarse en MariaDB.
5. La UI no debe acceder directamente a MinIO ni a MariaDB.
6. Si la validación falla, la imagen no debe almacenarse.
7. El `storageKey` guardado en MariaDB debe identificar el objeto almacenado en MinIO.

## Tipos permitidos

- image/jpeg
- image/png
- image/webp

## Flujo esperado

UI → Logic → Data → MinIO / MariaDB