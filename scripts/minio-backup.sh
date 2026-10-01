#!/usr/bin/env bash
# Issue #36 — respaldo opcional del MinIO local antes de cambiar su imagen.
#
# La imagen nueva (cgr.dev/chainguard/minio) lee los volúmenes `minio_data` creados
# con la anterior: actualizar no requiere migrar nada. Este script es una precaución
# por si la versión nueva de MinIO actualiza el formato del volumen. Ver README,
# sección "Cambio de imagen de MinIO (issue #36)".
#
# El respaldo es una copia exacta de los archivos del volumen, hecha con un
# contenedor `alpine` que monta el original en SOLO LECTURA: ningún servidor MinIO
# lo toca, y la copia conserva el formato anterior (sirve incluso para volver a la
# imagen vieja). Cada copia se verifica con el sha256 de todos los archivos.
#
# Uso, desde la raíz del repo y con el stack apagado (`docker compose down`, sin -v):
#   bash scripts/minio-backup.sh inventory   # solo lectura: qué hay en tu MinIO
#   bash scripts/minio-backup.sh backup      # copia <proyecto>_minio_data → <proyecto>_minio_backup
#   bash scripts/minio-backup.sh restore     # copia el respaldo a un <proyecto>_minio_data vacío
set -euo pipefail

TOOL_IMAGE="alpine:3"
PROJECT="${COMPOSE_PROJECT_NAME:-$(basename "$PWD" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9_-')}"
DATA_VOLUME="${PROJECT}_minio_data"
BACKUP_VOLUME="${PROJECT}_minio_backup"

# Git Bash convierte rutas como /data en rutas de Windows; aquí son rutas del contenedor.
export MSYS_NO_PATHCONV=1

die() { echo "ERROR: $*" >&2; exit 1; }

volume_exists() { docker volume inspect "$1" >/dev/null 2>&1; }

# Copiar los archivos de un MinIO en marcha da una copia inconsistente, y un
# contenedor detenido sigue asociado al volumen (docker volume rm fallaría después).
require_volume_unused() {
  local users
  users=$(docker ps -a --filter "volume=$1" --format '{{.Names}}')
  [[ -z "$users" ]] ||
    die "el volumen $1 lo usa: $users. Primero: docker compose down (sin -v)"
}

volume_is_empty() {
  [[ -z "$(docker run --rm -v "$1":/data:ro "$TOOL_IMAGE" ls -A /data)" ]]
}

# Copia SRC (solo lectura) a DST y compara el sha256 de cada archivo de ambos lados.
# Imprime la cantidad de archivos verificados; termina con error si algo no coincide.
copy_and_verify() {
  docker run --rm -v "$1":/src:ro -v "$2":/dst "$TOOL_IMAGE" sh -euc '
    cp -a /src/. /dst/
    cd /src && find . -type f -exec sha256sum {} + > /tmp/src.list
    cd /dst && find . -type f -exec sha256sum {} + > /tmp/dst.list
    sort -k 2 /tmp/src.list > /tmp/src.sorted
    sort -k 2 /tmp/dst.list > /tmp/dst.sorted
    cmp -s /tmp/src.sorted /tmp/dst.sorted || { echo "las sumas sha256 no coinciden" >&2; exit 1; }
    wc -l < /tmp/src.sorted'
}

cmd_inventory() {
  volume_exists "$DATA_VOLUME" || { echo "No existe $DATA_VOLUME: no hay nada que respaldar."; return; }
  echo "Contenido de $DATA_VOLUME (máximo 20 objetos por bucket):"
  docker run --rm -v "$DATA_VOLUME":/data:ro "$TOOL_IMAGE" sh -c '
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
  volume_exists "$DATA_VOLUME" || die "no existe $DATA_VOLUME: no hay nada que respaldar"
  require_volume_unused "$DATA_VOLUME"
  if volume_exists "$BACKUP_VOLUME"; then
    die "ya existe $BACKUP_VOLUME; restáuralo o bórralo (docker volume rm $BACKUP_VOLUME) antes de otro respaldo"
  fi

  docker volume create "$BACKUP_VOLUME" >/dev/null
  echo "Copiando $DATA_VOLUME → $BACKUP_VOLUME (el original se monta de solo lectura)..."
  local files
  if ! files=$(copy_and_verify "$DATA_VOLUME" "$BACKUP_VOLUME"); then
    # Un respaldo a medias no debe pasar por bueno ni bloquear el siguiente intento.
    docker volume rm "$BACKUP_VOLUME" >/dev/null 2>&1 || true
    die "el respaldo falló; se borró el volumen incompleto $BACKUP_VOLUME"
  fi
  echo "Respaldo verificado: $files archivos con el mismo sha256 en $BACKUP_VOLUME."
}

cmd_restore() {
  volume_exists "$BACKUP_VOLUME" || die "no existe $BACKUP_VOLUME; corre primero: backup"
  if volume_exists "$DATA_VOLUME"; then
    require_volume_unused "$DATA_VOLUME"
    # Mezclar el respaldo con los archivos actuales dejaría un volumen inconsistente.
    volume_is_empty "$DATA_VOLUME" ||
      die "$DATA_VOLUME no está vacío; si quieres reemplazarlo: docker volume rm $DATA_VOLUME"
  fi

  docker volume create "$DATA_VOLUME" >/dev/null
  echo "Copiando $BACKUP_VOLUME → $DATA_VOLUME..."
  local files
  files=$(copy_and_verify "$BACKUP_VOLUME" "$DATA_VOLUME") ||
    die "la restauración falló; $DATA_VOLUME puede estar incompleto (bórralo y vuelve a intentar)"
  echo "Restauración verificada: $files archivos con el mismo sha256."
  echo "Levanta el stack (docker compose up -d) y, cuando todo esté bien: docker volume rm $BACKUP_VOLUME"
}

case "${1:-}" in
  inventory) cmd_inventory ;;
  backup) cmd_backup ;;
  restore) cmd_restore ;;
  *) die "uso: bash scripts/minio-backup.sh {inventory|backup|restore}" ;;
esac
