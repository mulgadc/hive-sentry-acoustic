#!/usr/bin/env bash
set -euo pipefail

# Helper script to build and run the Jetson container with audio and models mounted.
# Usage examples:
#   ./scripts/run_jetson.sh build
#   ./scripts/run_jetson.sh up
#   ./scripts/run_jetson.sh exec
#   ./scripts/run_jetson.sh down
#   ./scripts/run_jetson.sh doctor
#
# Environment overrides:
#   L4T_TAG=r35.4.1-py3 MODELS_DIR=/path/to/models ./scripts/run_jetson.sh build

CWD_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$CWD_DIR"

export L4T_TAG=${L4T_TAG:-r36.2.0-py3}
export MODELS_DIR=${MODELS_DIR:-./src/droneprint/models}

# Prevent host overrides from forcing wrong platform during native builds
if [[ "${DOCKER_DEFAULT_PLATFORM:-}" == "linux/amd64" ]]; then
  echo "Warning: DOCKER_DEFAULT_PLATFORM=linux/amd64 is set. Clearing it for native aarch64 build on Jetson." >&2
  unset DOCKER_DEFAULT_PLATFORM
fi

doctor() {
  echo "--- Platform diagnostics ---"
  echo -n "uname -m: "; uname -m || true
  echo -n "docker arch: "; docker info --format '{{.Architecture}}' || true
  echo -n "docker default platform: "; echo "${DOCKER_DEFAULT_PLATFORM:-<unset>}"
  echo -n "nvidia runtime available: "; (docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia' && echo yes) || echo no
  echo -n "compose file: "; ls -1 docker-compose.jetson.yml || true
  echo -n "L4T_TAG: "; echo "${L4T_TAG}"
  echo "----------------------------"
}

case "${1:-}" in
  build)
    echo "Building image for Jetson (L4T_TAG=${L4T_TAG})..."
    doctor
    # Use plain progress to surface any exec format errors clearly
    docker compose -f docker-compose.jetson.yml build --progress=plain
    ;;
  build-nocache|rebuild)
    echo "Rebuilding image without cache (L4T_TAG=${L4T_TAG})..."
    doctor
    docker compose -f docker-compose.jetson.yml build --no-cache --pull --progress=plain
    ;;
  up)
    echo "Starting container..."
    docker compose -f docker-compose.jetson.yml up -d
    ;;
  down)
    echo "Stopping container..."
    docker compose -f docker-compose.jetson.yml down
    ;;
  exec)
    # Drop into the running container shell
    docker compose -f docker-compose.jetson.yml exec rtp bash
    ;;
  logs)
    docker compose -f docker-compose.jetson.yml logs -f
    ;;
  doctor)
    doctor
    ;;
  *)
    echo "Usage: $0 {build|build-nocache|rebuild|up|down|exec|logs|doctor}"
    exit 1
    ;;
esac
