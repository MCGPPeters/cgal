#!/usr/bin/env bash
# Run the CGAL/Monty benchmark stack with whichever container engine is
# available. Pass-through: any args/env vars normally given to
# `docker compose` work unchanged.
#
# Usage:
#   ./compose.sh up --build
#   SUITE=noise ./compose.sh up
#   ENGINE=podman ./compose.sh up        # force podman-compose
#   ENGINE=docker ./compose.sh up        # force docker compose
set -euo pipefail

cd "$(dirname "$0")"

ENGINE="${ENGINE:-auto}"

detect() {
    if [[ "${ENGINE}" == "docker" ]]; then echo docker; return; fi
    if [[ "${ENGINE}" == "podman" ]]; then echo podman; return; fi
    if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
        echo docker
    elif command -v podman-compose >/dev/null 2>&1; then
        echo podman
    elif command -v podman >/dev/null 2>&1 && podman compose version >/dev/null 2>&1; then
        echo podman-native
    else
        echo "No supported compose tool found (docker compose, podman-compose, podman compose)." >&2
        exit 127
    fi
}

ENGINE_RESOLVED="$(detect)"

case "${ENGINE_RESOLVED}" in
    docker)
        echo "[compose] using docker compose -f docker-compose.yml"
        exec docker compose -f docker-compose.yml "$@"
        ;;
    podman)
        echo "[compose] using podman-compose -f podman-compose.yml"
        exec podman-compose -f podman-compose.yml "$@"
        ;;
    podman-native)
        echo "[compose] using podman compose -f podman-compose.yml"
        exec podman compose -f podman-compose.yml "$@"
        ;;
esac
