#!/usr/bin/env bash
# Terminate a Lambda Labs instance by ID.
# Usage: ./teardown.sh <instance_id>
set -euo pipefail
: "${LAMBDA_API_KEY:?Set LAMBDA_API_KEY}"
[ $# -eq 1 ] || { echo "usage: $0 <instance_id>" >&2; exit 1; }
curl -fsS -u "${LAMBDA_API_KEY}:" -H 'Content-Type: application/json' \
  -d "{\"instance_ids\":[\"$1\"]}" \
  https://cloud.lambdalabs.com/api/v1/instance-operations/terminate
echo
