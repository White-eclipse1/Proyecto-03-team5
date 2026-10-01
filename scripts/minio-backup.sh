#!/usr/bin/env bash
# Issue #36 — respaldo opcional del MinIO local antes de cambiar su imagen.
#
# La imagen nueva (cgr.dev/chainguard/minio) lee los volúmenes `minio_data` creados
# con la anterior: actualizar no requiere migrar nada. Este script es una precaución
# por si quieres una copia antes de que la versión nueva de MinIO toque tu volumen.
# Ver README, sección "Cambio de imagen de MinIO (issue #36)".
#
# Uso, desde la raíz del repo, con el .env del proyecto y el stack apagado
# (`docker compose down`, sin -v):
#   bash scripts/minio-backup.sh inventory   # solo lectura: qué hay en tu MinIO
#   bash scripts/minio-backup.sh backup      # copia los buckets a <proyecto>_minio_backup
#   bash scripts/minio-backup.sh restore     # devuelve el respaldo a <proyecto>_minio_data
#
# Solo usa imágenes públicas. Cada paso compara la lista de objetos (ruta y tamaño)
# de origen y destino, y termina con error si no coinciden.
set -euo pipefail

SERVER_IMAGE="cgr.dev/chainguard/minio:latest"
CLIENT_IMAGE="cgr.dev/chainguard/minio-client:latest"
PROJECT="${COMPOSE_PROJECT_NAME:-$(basename "$PWD" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')}"
DATA_VOLUME="${PROJECT}_minio_data"
BACKUP_VOLUME="${PROJECT}_minio_backup"
TEMP_NAME="${PROJECT}-minio-backup-tmp"

# Git Bash convierte rutas como /data en rutas de Windows; aquí son rutas del contenedor.
export MSYS_NO_PATHCONV=1

die() { echo "ERROR: $*" >&2; exit 1; }

volume_exists() { docker volume inspect "$1" >/dev/null 2>&1; }

load_credentials() {
  [[ -f .env ]] || die "falta .env en $(pwd); ejecuta el script desde la raíz del repo"
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
  [[ -n "${MINIO_ROOT_USER:-}" && -n "${MINIO_ROOT_PASSWORD:-}" ]] ||
    die "MINIO_ROOT_USER y MINIO_ROOT_PASSWORD deben estar definidos en .env"
}

# Dos servidores MinIO sobre el mismo volumen lo corrompen: el del proyecto debe
# estar apagado (y sin contenedor, para poder borrar el volumen después).
require_volume_unused() {
  local users
  users=$(docker ps -a --filter "volume=$1" --format '{{.Names}}')
  [[ -z "$users" ]] ||
    die "el volumen $1 lo usa: $users. Primero: docker compose down (sin -v)"
}

# Buckets = carpetas de primer nivel del volumen (sin la de sistema de MinIO).
list_buckets() {
  docker run --rm -v "$1":/data:ro alpine sh -c '
    for dir in /data/*/; do
      [ -d "$dir" ] || continue
      name=$(basename "$dir")
      [ "$name" = ".minio.sys" ] || echo "$name"
    done'
}

# Servidor MinIO temporal sobre DATA_VOLUME, en una red propia.
start_temp_server() {
  docker network create "$TEMP_NAME" >/dev/null
  docker run -d --name "$TEMP_NAME" --network "$TEMP_NAME" --user 0:0 \
    -v "$DATA_VOLUME":/data -e MINIO_ROOT_USER -e MINIO_ROOT_PASSWORD \
    "$SERVER_IMAGE" server /data >/dev/null
  for _ in $(seq 60); do
    mc ls minio >/dev/null 2>&1 && return 0
    sleep 1
  done
  docker logs "$TEMP_NAME" >&2 || true
  die "el MinIO temporal no respondió en 60 s"
}

BACKUP_INCOMPLETE=0

cleanup() {
  docker rm -f "$TEMP_NAME" >/dev/null 2>&1 || true
  docker network rm "$TEMP_NAME" >/dev/null 2>&1 || true
  # Un respaldo que falló a medias no debe pasar por bueno ni bloquear el siguiente intento.
  if [[ "$BACKUP_INCOMPLETE" == 1 ]]; then
    docker volume rm "$BACKUP_VOLUME" >/dev/null 2>&1 || true
    echo "El respaldo falló; se borró el volumen incompleto $BACKUP_VOLUME." >&2
  fi
}

mc() {
  docker run --rm -i --network "$TEMP_NAME" --user 0:0 \
    -e MC_HOST_minio="http://${MINIO_ROOT_USER}:${MINIO_ROOT_PASSWORD}@${TEMP_NAME}:9000" \
    -v "$BACKUP_VOLUME":/backup "$CLIENT_IMAGE" "$@"
}

# "ruta<TAB>tamaño" de cada objeto, ordenado: lo que se compara entre origen y destino.
object_list() {
  mc ls --recursive --json "$1" |
    sed -nE 's/.*"size":([0-9]+),"key":"(([^"\\]|\\.)*)".*/\2	\1/p' | LC_ALL=C sort
}

count_lines() {
  if [[ -z "$1" ]]; then echo 0; else printf '%s\n' "$1" | wc -l | tr -d ' '; fi
}

cmd_inventory() {
  volume_exists "$DATA_VOLUME" || { echo "No existe $DATA_VOLUME: no hay nada que respaldar."; return; }
  echo "Contenido de $DATA_VOLUME (máximo 20 objetos por bucket):"
  docker run --rm -v "$DATA_VOLUME":/data:ro alpine sh -c '
    found=0
    for bucket in /data/*/; do
      name=$(basename "$bucket")
      [ "$name" = ".minio.sys" ] && continue
      found=1
      echo "bucket $name: $(find "$bucket" -name xl.meta | wc -l) objetos"
      find "$bucket" -name xl.meta | sed "s#^/data/##; s#/xl.meta\$##" | head -20 | sed "s/^/    /"
    done
    [ "$found" = 1 ] || echo "(sin buckets)"'
  cat <<'EOF'

Cómo leerlo:
  dvc-cache             remote `dev` de DVC; el dataset oficial está en `prod` (AWS S3).
  image-annotations     imágenes subidas al portal en local; seed/sample-*.png son de prueba.
  mlflow                artefactos de MLflow (OPS-03), si ya corriste experimentos.
EOF
}

cmd_backup() {
  load_credentials
  volume_exists "$DATA_VOLUME" || die "no existe $DATA_VOLUME"
  require_volume_unused "$DATA_VOLUME"
  if volume_exists "$BACKUP_VOLUME"; then
    die "ya existe $BACKUP_VOLUME; restáuralo o bórralo (docker volume rm $BACKUP_VOLUME) antes de otro respaldo"
  fi
  local buckets
  buckets=$(list_buckets "$DATA_VOLUME")
  [[ -n "$buckets" ]] || { echo "$DATA_VOLUME no tiene buckets: no hay nada que respaldar."; return; }

  docker volume create "$BACKUP_VOLUME" >/dev/null
  BACKUP_INCOMPLETE=1
  trap cleanup EXIT
  start_temp_server

  echo "Respaldando $DATA_VOLUME → $BACKUP_VOLUME..."
  local bucket source copy
  for bucket in $buckets; do
    mc mirror --overwrite --preserve "minio/$bucket" "/backup/$bucket" >/dev/null
    # En variables y no con <(...): así un `mc ls` fallido detiene el script (set -e)
    # en vez de producir dos listas vacías que "coinciden".
    source=$(object_list "minio/$bucket")
    copy=$(object_list "/backup/$bucket")
    [[ "$source" == "$copy" ]] ||
      die "el respaldo de $bucket no coincide con el original; $BACKUP_VOLUME está incompleto"
    echo "  $bucket: $(count_lines "$copy") objetos verificados"
  done
  BACKUP_INCOMPLETE=0
  echo "Respaldo verificado en el volumen $BACKUP_VOLUME."
}

cmd_restore() {
  load_credentials
  volume_exists "$BACKUP_VOLUME" || die "no existe $BACKUP_VOLUME; corre primero: backup"
  volume_exists "$DATA_VOLUME" && require_volume_unused "$DATA_VOLUME"
  local buckets
  buckets=$(list_buckets "$BACKUP_VOLUME")
  [[ -n "$buckets" ]] || die "$BACKUP_VOLUME está vacío"

  docker volume create "$DATA_VOLUME" >/dev/null
  trap cleanup EXIT
  start_temp_server

  echo "Restaurando $BACKUP_VOLUME → $DATA_VOLUME..."
  local bucket saved restored missing
  for bucket in $buckets; do
    mc mb --ignore-existing "minio/$bucket" >/dev/null
    mc mirror --overwrite "/backup/$bucket" "minio/$bucket" >/dev/null
    saved=$(object_list "/backup/$bucket")
    restored=$(object_list "minio/$bucket")
    # Todo lo respaldado debe estar en el destino (puede haber objetos extra nuevos).
    missing=$(LC_ALL=C comm -23 <(printf '%s\n' "$saved") <(printf '%s\n' "$restored"))
    [[ -z "$missing" ]] || die "faltan objetos de $bucket en el destino:"$'\n'"$missing"
    echo "  $bucket: $(count_lines "$saved") objetos verificados"
  done
  echo "Restauración verificada. Levanta el stack (docker compose up -d) y revisa el portal;"
  echo "cuando todo esté bien: docker volume rm $BACKUP_VOLUME"
}

case "${1:-}" in
  inventory) cmd_inventory ;;
  backup) cmd_backup ;;
  restore) cmd_restore ;;
  *) die "uso: bash scripts/minio-backup.sh {inventory|backup|restore}" ;;
esac
