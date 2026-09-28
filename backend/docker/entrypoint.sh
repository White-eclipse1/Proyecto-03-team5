#!/bin/sh
set -e

node docker/wait-for.mjs "${DB_HOST}" "${DB_PORT}"
node docker/wait-for.mjs "${MINIO_ENDPOINT}" "${MINIO_PORT}"

echo "[entrypoint] Aplicando migraciones..."
npm run db:migrate

echo "[entrypoint] Sembrando datos de ejemplo..."
npm run db:seed

echo "[entrypoint] Arrancando servidor..."
exec node dist/ui/server.js
