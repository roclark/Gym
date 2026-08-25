#!/usr/bin/env bash
# Agent-facing CyberGym PoC submission helper.
set -euo pipefail

case "${1:-}" in
    ""|-h|--help)
        echo "Usage: $0 <poc_file_path>"
        echo "Submit a raw proof-of-concept input to the vulnerable target."
        exit 0
        ;;
esac

POC_FILE="$1"
if [ ! -f "$POC_FILE" ]; then
    echo "Error: file not found: $POC_FILE" >&2
    exit 1
fi

TASK_SERVER="${CYBERGYM_TASK_SERVER_URL:-http://task-server:9111}"
WORKSPACE_DIR="${CYBERGYM_WORKSPACE_DIR:-/workspace}"
AGENT_LOGS_DIR="${CYBERGYM_AGENT_LOGS_DIR:-/logs/agent}"

RESPONSE=$(curl -sf --max-time 70 -X POST -F "poc=@${POC_FILE}" "$TASK_SERVER/submit" 2>&1) || {
    status=$?
    echo "Error: task-server submission failed (curl exit $status)" >&2
    echo "$RESPONSE" >&2
    exit 1
}

EXIT_CODE=$(printf '%s' "$RESPONSE" | grep -o '"exit_code": *[-0-9]*' | grep -o '[-0-9]*$' || true)
if [ -z "$EXIT_CODE" ]; then
    echo "Error: unexpected task-server response: $RESPONSE" >&2
    exit 1
fi

umask 077
mkdir -p "$WORKSPACE_DIR/submissions" "$AGENT_LOGS_DIR/artifacts/submissions"
shopt -s nullglob
existing=("$WORKSPACE_DIR"/submissions/poc_*)
submission_number=$((${#existing[@]} + 1))
submission_name=$(printf 'poc_%03d' "$submission_number")
cp "$POC_FILE" "$WORKSPACE_DIR/submissions/$submission_name"
cp "$POC_FILE" "$WORKSPACE_DIR/poc"
cp "$POC_FILE" "$AGENT_LOGS_DIR/artifacts/submissions/$submission_name"
cp "$POC_FILE" "$AGENT_LOGS_DIR/artifacts/poc"

{
    echo "=== $submission_name ==="
    echo "$RESPONSE"
    echo
} >> "$AGENT_LOGS_DIR/artifacts/submit_log.txt"

echo
echo "$RESPONSE"
