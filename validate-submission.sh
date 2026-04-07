#!/usr/bin/env bash
set -euo pipefail

SPACE_URL="${1:-}"
REPO_DIR="${2:-.}"

if [[ -n "${SPACE_URL}" ]]; then
  echo "Checking space health: ${SPACE_URL}"
  curl -fsSL "${SPACE_URL%/}/health" >/dev/null
  curl -fsSL -X POST "${SPACE_URL%/}/reset" \
    -H "content-type: application/json" \
    -d '{"task_name":"stereotype_probe"}' >/dev/null
fi

echo "Building root Dockerfile"
docker build -t red-teaming-env-validation-root "${REPO_DIR}"

echo "Building server/Dockerfile"
docker build -t red-teaming-env-validation-server -f "${REPO_DIR}/server/Dockerfile" "${REPO_DIR}"

echo "Running openenv validate"
"${REPO_DIR}/.venv/bin/openenv" validate "${REPO_DIR}"

