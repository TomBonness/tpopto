#!/usr/bin/env bash
set -euo pipefail

IMAGE='lmsysorg/sglang:glm-5.3-flash@sha256:3a97bd50034ca60c6e6c86b8e36a73675d261f6a5eb71197796aee5175409290'
DEPLOY_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Pinned SGLang image is not present on the retained OS volume; refusing an implicit large pull." >&2
  exit 1
fi

exec docker run --rm \
  --network host \
  --entrypoint python3 \
  -v /workspace:/workspace \
  -v "$DEPLOY_DIR:/deploy:ro" \
  "$IMAGE" \
  /deploy/stage-dflash2.py
