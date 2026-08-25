#!/usr/bin/env bash
# Harbor uploads oracle files only after the agent phase, so this token and
# reference-PoC endpoint are not visible to evaluated agents.
set -euo pipefail

AUTH_TOKEN="{auth_token}"
TASK_SERVER="${CYBERGYM_TASK_SERVER_URL:-http://task-server:9111}"
mkdir -p /workspace

http_code=$(curl -sf --max-time 30 -o /workspace/poc -w '%{http_code}' \
    -H "Authorization: Bearer $AUTH_TOKEN" \
    "$TASK_SERVER/solve") || {
    echo "ERROR: failed to retrieve CyberGym ground-truth PoC" >&2
    exit 1
}

if [ "$http_code" -ne 200 ] || [ ! -s /workspace/poc ]; then
    echo "ERROR: task-server returned HTTP $http_code or an empty PoC" >&2
    exit 1
fi
echo "Oracle copied the ground-truth PoC to /workspace/poc"
