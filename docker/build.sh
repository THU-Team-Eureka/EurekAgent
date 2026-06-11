#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IMAGE="${EUREKA_DOCKER_IMAGE:-eureka-agent:node22-bookworm}"
BASE_IMAGE="${EUREKA_DOCKER_BASE_IMAGE:-node:22-bookworm}"
INSTALL_CLAUDE_CODE="${EUREKA_DOCKER_INSTALL_CLAUDE_CODE:-1}"

docker build \
  --network host \
  ${http_proxy:+--build-arg http_proxy="$http_proxy"} \
  ${https_proxy:+--build-arg https_proxy="$https_proxy"} \
  ${HTTP_PROXY:+--build-arg HTTP_PROXY="$HTTP_PROXY"} \
  ${HTTPS_PROXY:+--build-arg HTTPS_PROXY="$HTTPS_PROXY"} \
  ${NO_PROXY:+--build-arg NO_PROXY="$NO_PROXY"} \
  ${no_proxy:+--build-arg no_proxy="$no_proxy"} \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg INSTALL_CLAUDE_CODE="$INSTALL_CLAUDE_CODE" \
  -t "$IMAGE" \
  -f "$SCRIPT_DIR/Dockerfile" \
  "$PROJECT_ROOT"
