#!/usr/bin/env bash
# rebuild.sh — the safe rebuild path for Eidetic docker compose services.
#
# Usage: ./rebuild.sh [service ...] [--frontend] [-h|--help]
#   (default services: api arq-worker scheduler metrics-worker)
#
# Why not `docker compose up --build <svc>`: compose also rebuilds every
# image the requested service depends_on, not just the one you asked for.
# `docker compose up -d --build frontend` was observed rebuilding the api
# image too (frontend has no real dependency on api's image, but compose
# still walked the graph) — that is what exhausted host memory/disk during
# the 2026-09-20 incident (see BUGS.md). This script always `build`s ONLY
# the named services, then recreates with --no-deps --no-build so `up`
# cannot pull in anything else.
#
# Why prune is mandatory here: each backend image build leaves ~4 GB of
# build cache behind, and this host runs near-full (~180 GB of 221 GB
# baseline used even before a rebuild). Skipping the prune step after a
# build is how the host hit 100% disk and crash-looped Postgres (BUGS.md,
# 2026-09-20 — "PANIC: could not write to file ... No space left on
# device"). So prune runs after every successful build here, unconditionally
# — it is not an optional cleanup step.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage: ./rebuild.sh [service ...] [--frontend] [-h|--help]

Rebuilds the named docker compose services the safe way: builds ONLY the
requested images, then recreates ONLY those containers. Never runs
`up --build`, which also rebuilds every image the service depends on.

  service ...   one or more compose service names
                (default: api arq-worker scheduler metrics-worker)
  --frontend    shorthand for rebuilding the frontend image
  -h, --help    show this help and exit

Env overrides:
  MIN_FREE_GB      minimum free GB required on / to proceed (default: 15)
  HEALTH_URL       health endpoint polled when 'api' is rebuilt
                    (default: http://localhost:8000/health)
  HEALTH_WAIT_MAX  max seconds to wait for HEALTH_URL (default: 120)

Examples:
  ./rebuild.sh                      # rebuild api arq-worker scheduler metrics-worker
  ./rebuild.sh api                  # rebuild just api
  ./rebuild.sh --frontend           # rebuild frontend
  ./rebuild.sh api --frontend       # rebuild api and frontend

Must be run from the docker/ directory (same convention as backup.sh).
EOF
}

# ── must be run from docker/ ────────────────────────────────────────────────
if [[ "$(pwd)" != "$SCRIPT_DIR" ]]; then
  echo "[rebuild] error: run this script from the docker/ directory (cd docker && ./rebuild.sh)" >&2
  exit 1
fi
if [[ ! -f docker-compose.yml ]]; then
  echo "[rebuild] error: docker-compose.yml not found in $(pwd)" >&2
  exit 1
fi

MIN_FREE_GB="${MIN_FREE_GB:-15}"
HEALTH_URL="${HEALTH_URL:-http://localhost:8000/health}"
HEALTH_WAIT_MAX="${HEALTH_WAIT_MAX:-120}"

DEFAULT_SERVICES=(api arq-worker scheduler metrics-worker)
SERVICES=()
FRONTEND=false

# ── parse args ───────────────────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --frontend) FRONTEND=true ;;
    -*)
      echo "[rebuild] error: unknown flag: $arg" >&2
      usage >&2
      exit 1
      ;;
    *) SERVICES+=("$arg") ;;
  esac
done

if [[ "$FRONTEND" == true ]]; then
  SERVICES+=(frontend)
fi

if [[ ${#SERVICES[@]} -eq 0 ]]; then
  SERVICES=("${DEFAULT_SERVICES[@]}")
fi

echo "[rebuild] services: ${SERVICES[*]}"

# ── validate service names against compose before touching anything ────────
VALID_SERVICES="$(docker compose config --services 2>/dev/null || true)"
for s in "${SERVICES[@]}"; do
  if ! grep -qx "$s" <<<"$VALID_SERVICES"; then
    echo "[rebuild] error: '$s' is not a service in docker-compose.yml" >&2
    echo "[rebuild] known services: $(tr '\n' ' ' <<<"$VALID_SERVICES")" >&2
    exit 1
  fi
done

# ── pre-flight: disk + memory ───────────────────────────────────────────────
free_gb() {
  df --output=avail -BG / | tail -n1 | tr -dc '0-9'
}

echo "[rebuild] pre-flight:"
echo "  free disk on /: $(df -h / | tail -n1 | awk '{print $4}') (threshold: ${MIN_FREE_GB}G)"
free -h | sed 's/^/  /'

FREE_GB="$(free_gb)"
if [[ "$FREE_GB" -lt "$MIN_FREE_GB" ]]; then
  echo "[rebuild] error: only ${FREE_GB}G free on / (need >= ${MIN_FREE_GB}G)." >&2
  echo "[rebuild] prune before retrying:" >&2
  echo "  docker builder prune -f && docker image prune -f" >&2
  echo "  docker system df                 # see what is using space" >&2
  echo "  docker system df -v              # per-image/volume breakdown" >&2
  exit 1
fi

# ── build only the requested services ───────────────────────────────────────
echo "[rebuild] building: ${SERVICES[*]}"
docker compose build "${SERVICES[@]}"

# ── recreate only the requested containers, no dependency pulls ────────────
echo "[rebuild] recreating: ${SERVICES[*]}"
docker compose up -d --no-deps --no-build --force-recreate "${SERVICES[@]}"

# ── mandatory prune (see header comment — this is not optional on this host) ─
DISK_BEFORE="$(df -h / | tail -n1 | awk '{print $4}')"
echo "[rebuild] pruning build cache + dangling images..."
docker builder prune -f
docker image prune -f
DISK_AFTER="$(df -h / | tail -n1 | awk '{print $4}')"
echo "[rebuild] free disk on /: ${DISK_BEFORE} -> ${DISK_AFTER}"

# ── wait for /health if api was rebuilt ─────────────────────────────────────
contains_api=false
for s in "${SERVICES[@]}"; do
  [[ "$s" == "api" ]] && contains_api=true
done

if [[ "$contains_api" == true ]]; then
  echo "[rebuild] waiting for ${HEALTH_URL} (max ${HEALTH_WAIT_MAX}s)..."
  waited=0
  body=""
  until body="$(curl -fsS "$HEALTH_URL" 2>/dev/null)"; do
    sleep 3
    waited=$((waited + 3))
    if [[ "$waited" -ge "$HEALTH_WAIT_MAX" ]]; then
      echo "[rebuild] error: ${HEALTH_URL} did not come up within ${HEALTH_WAIT_MAX}s" >&2
      exit 1
    fi
  done
  echo "[rebuild] health: ${body}"
fi

echo "[rebuild] done."
