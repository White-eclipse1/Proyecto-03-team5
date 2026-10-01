#!/usr/bin/env bash
# Issue #36 — migrar el MinIO local a la imagen pública de Chainguard.
#
# La imagen nueva no lee los buckets de un volumen `minio_data` creado con la
# imagen anterior (quay.io/minio/minio). Este script respalda esos buckets en un
# volumen de Docker y los restaura en el MinIO nuevo. Ver README, sección
# "Migración de MinIO (issue #36)".
#
# Uso, desde la raíz del repo y con el .env del proyecto:
#   bash scripts/minio-migrate.sh inventory   # solo lectura: qué hay en tu MinIO
#   bash scripts/minio-migrate.sh backup      # copia los buckets a <proyecto>_minio_backup
#   bash scripts/minio-migrate.sh restore     # los sube al MinIO nuevo (ya levantado)
#
# `backup` necesita la imagen vieja en caché (quay.io/minio/minio): ya no se puede
# descargar. Si no la tienes, tampoco tienes datos que migrar.
set -euo pipefail

OLD_IMAGE="quay.io/minio/minio"
PROJECT="${COMPOSE_PROJECT_NAME:-$(basename "$PWD" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')}"
DATA_VOLUME="${PROJECT}_minio_data"
BACKUP_VOLUME="${PROJECT}_minio_backup"
NETWORK="${PROJECT}_default"

# Git Bash convierte rutas como /data en rutas de Windows; aquí son rutas del contenedor.
export MSYS_NO_PATHCONV=1

die() { echo "ERROR: $*" >&2; exit 1; }

load_credentials() {
  [[ -f .env ]] || die "falta .env en $(pwd); ejecuta el script desde la raíz del repo"
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
  [[ -n "${MINIO_ROOT_USER:-}" && -n "${MINIO_ROOT_PASSWORD:-}" ]] ||
    die "MINIO_ROOT_USER y MINIO_ROOT_PASSWORD deben estar definidos en .env"
}

volume_exists() { docker volume inspect "$1" >/dev/null 2>&1; }

require_old_image() {
  docker image inspect "$OLD_IMAGE" >/dev/null 2>&1 ||
    die "no tienes $OLD_IMAGE en caché (ya no se puede descargar); no hay datos de la imagen vieja que respaldar"
}

# Lista buckets y objetos leyendo el volumen en modo solo lectura.
list_volume() {
  docker run --rm -v "$1":/data:ro --entrypoint sh "${2:-alpine}" -c '
    found=0
    for bucket in /data/*/; do
      name=$(basename "$bucket")
      [ "$name" = ".minio.sys" ] && continue
      found=1
      echo "bucket $name: $(find "$bucket" -name xl.meta | wc -l) objetos"
      find "$bucket" -name xl.meta | sed "s#^/data/##; s#/xl.meta\$##" | head -20 | sed "s/^/    /"
    done
    [ "$found" = 1 ] || echo "(sin buckets)"'
}

cmd_inventory() {
  volume_exists "$DATA_VOLUME" || { echo "No existe $DATA_VOLUME: no hay nada que migrar."; return; }
  echo "Contenido de $DATA_VOLUME (máximo 20 objetos por bucket):"
  list_volume "$DATA_VOLUME"
  cat <<'EOF'

Cómo leerlo:
  dvc-cache             remote `dev` de DVC; el dataset oficial está en `prod` (AWS S3).
  image-annotations     imágenes subidas al portal en local; seed/sample-*.png son de prueba.
  mlflow                artefactos de MLflow (OPS-03), si ya corriste experimentos.
EOF
}

cmd_backup() {
  load_credentials
  require_old_image
  volume_exists "$DATA_VOLUME" || die "no existe $DATA_VOLUME"
  if docker ps --format '{{.Names}}' | grep -q "^${PROJECT}-minio-"; then
    die "MinIO del proyecto sigue corriendo; primero: docker compose stop"
  fi
  docker volume create "$BACKUP_VOLUME" >/dev/null

  echo "Respaldando $DATA_VOLUME → $BACKUP_VOLUME con la imagen vieja..."
  docker run --rm \
    -v "$DATA_VOLUME":/data \
    -v "$BACKUP_VOLUME":/backup \
    -e MINIO_ROOT_USER -e MINIO_ROOT_PASSWORD \
    --entrypoint sh "$OLD_IMAGE" -c '
      # Esta imagen no trae awk/sed/grep/find: solo shell, curl y mc.
      minio server /data --address 127.0.0.1:9000 >/tmp/minio.log 2>&1 &
      for _ in $(seq 60); do curl -sf http://127.0.0.1:9000/minio/health/live >/dev/null 2>&1 && break; sleep 1; done
      export MC_HOST_old="http://${MINIO_ROOT_USER}:${MINIO_ROOT_PASSWORD}@127.0.0.1:9000"
      mc ls old >/dev/null || { cat /tmp/minio.log; exit 1; }
      for dir in /data/*/; do
        bucket=$(basename "$dir")
        [ "$bucket" = ".minio.sys" ] && continue
        mc mirror --overwrite --preserve "old/$bucket" "/backup/$bucket" >/dev/null
        echo "  $bucket: $(mc ls --recursive "/backup/$bucket" | wc -l) archivos"
      done'
  echo "Respaldo listo en el volumen $BACKUP_VOLUME."
}

cmd_restore() {
  load_credentials
  require_old_image
  volume_exists "$BACKUP_VOLUME" || die "no existe $BACKUP_VOLUME; corre primero: backup"
  docker ps --format '{{.Names}}' | grep -q "^${PROJECT}-minio-" ||
    die "levanta primero el MinIO nuevo: docker compose up -d minio"

  echo "Restaurando $BACKUP_VOLUME → MinIO nuevo..."
  # Se usa el `mc` que trae la imagen vieja como cliente; el servidor es el nuevo.
  docker run --rm --network "$NETWORK" \
    -v "$BACKUP_VOLUME":/backup:ro \
    -e MINIO_ROOT_USER -e MINIO_ROOT_PASSWORD \
    --entrypoint sh "$OLD_IMAGE" -c '
      export MC_HOST_new="http://${MINIO_ROOT_USER}:${MINIO_ROOT_PASSWORD}@minio:9000"
      for _ in $(seq 60); do mc ls new >/dev/null 2>&1 && break; sleep 1; done
      for dir in /backup/*/; do
        [ -d "$dir" ] || continue
        bucket=$(basename "$dir")
        mc mb --ignore-existing "new/$bucket" >/dev/null
        mc mirror --overwrite "$dir" "new/$bucket" >/dev/null
        echo "  $bucket: $(mc ls --recursive "new/$bucket" | wc -l) objetos"
      done'
  echo "Listo. Revisa el portal y, cuando todo esté bien: docker volume rm $BACKUP_VOLUME"
}

case "${1:-}" in
  inventory) cmd_inventory ;;
  backup) cmd_backup ;;
  restore) cmd_restore ;;
  *) die "uso: bash scripts/minio-migrate.sh {inventory|backup|restore}" ;;
esac
