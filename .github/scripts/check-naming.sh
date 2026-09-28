#!/usr/bin/env bash
# Comprueba la convención de nombre de rama y de título de PR (ver CONTRIBUTING.md).
#
# Uso: bash .github/scripts/check-naming.sh "<rama>" "<título del PR>"
#
# Lo llama .github/workflows/pr-hygiene.yml, y sirve igual en local antes de abrir el PR:
#   bash .github/scripts/check-naming.sh "$(git branch --show-current)" "P2-47: calidad y CI"
set -u

# Sin esto, en algunas configuraciones regionales [a-z] también acepta mayúsculas.
export LC_ALL=C

BRANCH="${1:-}"
TITLE="${2:-}"

TYPES='feat|fix|test|chore|docs|refactor|ci|build|perf|style'
# p2-52-copilot-llm-client · p2-22-23-24-quality-gate · p2-26-p2-27-lotes · fix/nombre-corto
BRANCH_RE="^(p2-[0-9]+(-[0-9]+)*(-p2-[0-9]+(-[0-9]+)*)*-[a-z0-9]+(-[a-z0-9]+)*|($TYPES)/[a-z0-9]+(-[a-z0-9]+)*)\$"
# P2-52: texto · P2-22/23/24: texto · P2-26 P2-27: texto · fix: texto · feat(ui): texto
TITLE_RE="^(P2-[0-9]+(/[0-9]+| P2-[0-9]+)*|($TYPES)(\\([a-z0-9-]+\\))?): .+"

status=0

if ! [[ "$BRANCH" =~ $BRANCH_RE ]]; then
  echo "::error title=Nombre de rama::'$BRANCH' no sigue la convención. Usa p2-NN-descripcion" \
    "(varios tickets: p2-22-23-24-descripcion) o tipo/descripcion con tipo en ($TYPES)," \
    "todo en minúsculas y con guiones. Ver CONTRIBUTING.md." >&2
  status=1
fi

if ! [[ "$TITLE" =~ $TITLE_RE ]]; then
  echo "::error title=Título del PR::'$TITLE' no sigue la convención. Usa 'P2-NN: descripción'" \
    "(varios tickets: 'P2-NN P2-MM: ...' o 'P2-22/23/24: ...') o 'tipo: descripción' con" \
    "tipo en ($TYPES). Ver CONTRIBUTING.md." >&2
  status=1
fi

if [[ "$status" -eq 0 ]]; then
  echo "La rama y el título cumplen la convención."
fi
exit "$status"
