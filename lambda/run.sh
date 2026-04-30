#!/usr/bin/env bash
# One-shot Lambda Labs runner: launch a GPU instance, run the CGAL benchmark
# inside Docker, fetch results, terminate the instance.
#
# Required env:
#   LAMBDA_API_KEY         Lambda Cloud API key (https://cloud.lambdalabs.com/api-keys)
# Optional env (with defaults):
#   INSTANCE_TYPE=gpu_1x_a10
#   REGION=                (empty = first region with capacity)
#   SSH_KEY_NAME=cgal-bench
#   SSH_PRIVATE_KEY=$HOME/.ssh/id_ed25519
#   CGAL_REF=main
#   MONTY_REF=cgal/main
#   SUITE=smoke            (smoke | noise | full)
#   N_EVAL_EPOCHS=         (override per-suite default)
#   KEEP_ALIVE=false       (true = don't terminate at the end)
#
# Cost guard: SUITE=smoke on gpu_1x_a10 typically takes 20-40 min ≈ $0.30-0.50.
# SUITE=full is hours; check before unleashing.
set -euo pipefail

: "${LAMBDA_API_KEY:?Set LAMBDA_API_KEY (https://cloud.lambdalabs.com/api-keys)}"

INSTANCE_TYPE="${INSTANCE_TYPE:-gpu_1x_a10}"
REGION="${REGION:-}"
SSH_KEY_NAME="${SSH_KEY_NAME:-cgal-bench}"
SSH_PRIVATE_KEY="${SSH_PRIVATE_KEY:-$HOME/.ssh/id_ed25519}"
SSH_PUBLIC_KEY="${SSH_PUBLIC_KEY:-${SSH_PRIVATE_KEY}.pub}"
CGAL_REF="${CGAL_REF:-main}"
MONTY_REF="${MONTY_REF:-cgal/main}"
SUITE="${SUITE:-smoke}"
N_EVAL_EPOCHS="${N_EVAL_EPOCHS:-}"
KEEP_ALIVE="${KEEP_ALIVE:-false}"

API="https://cloud.lambdalabs.com/api/v1"
AUTH=(-u "${LAMBDA_API_KEY}:")
HERE="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${HERE}/results-$(date -u +%Y%m%dT%H%M%SZ)"

say() { printf '\n[lambda] %s\n' "$*"; }
die() { printf '\n[lambda] ERROR: %s\n' "$*" >&2; exit 1; }

require() {
  command -v "$1" >/dev/null 2>&1 || die "missing tool: $1"
}
require curl
require jq
require ssh
require scp

[ -f "${SSH_PRIVATE_KEY}" ] || die "no private key at ${SSH_PRIVATE_KEY}"
[ -f "${SSH_PUBLIC_KEY}" ]  || die "no public key at ${SSH_PUBLIC_KEY}"

# ---- 1. Make sure our SSH key is registered with Lambda --------------------
say "checking SSH key '${SSH_KEY_NAME}' on Lambda Cloud..."
keys_json="$(curl -fsS "${AUTH[@]}" "${API}/ssh-keys")"
if ! echo "${keys_json}" | jq -e --arg n "${SSH_KEY_NAME}" \
      '.data[] | select(.name == $n)' >/dev/null; then
  say "registering public key as '${SSH_KEY_NAME}'..."
  pub="$(cat "${SSH_PUBLIC_KEY}")"
  curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' \
    -d "$(jq -n --arg n "${SSH_KEY_NAME}" --arg k "${pub}" \
            '{name:$n, public_key:$k}')" \
    "${API}/ssh-keys" >/dev/null
fi

# ---- 2. Pick a region with capacity for the requested instance type --------
say "checking capacity for ${INSTANCE_TYPE}..."
types_json="$(curl -fsS "${AUTH[@]}" "${API}/instance-types")"
if [ -z "${REGION}" ]; then
  REGION="$(echo "${types_json}" | jq -r --arg t "${INSTANCE_TYPE}" \
    '.data[$t].regions_with_capacity_available[0].name // empty')"
fi
if [ -z "${REGION}" ]; then
  say "no capacity for ${INSTANCE_TYPE} right now. Available types with capacity:"
  echo "${types_json}" | jq -r '
    .data | to_entries[] |
    select(.value.regions_with_capacity_available | length > 0) |
    "  \(.key) -> \([.value.regions_with_capacity_available[].name] | join(","))  $\(.value.instance_type.price_cents_per_hour/100)/h"
  '
  die "retry later, or pick another INSTANCE_TYPE"
fi
say "launching ${INSTANCE_TYPE} in ${REGION}..."

# ---- 3. Launch the instance -------------------------------------------------
launch_payload="$(jq -n \
  --arg region "${REGION}" \
  --arg type   "${INSTANCE_TYPE}" \
  --arg key    "${SSH_KEY_NAME}" \
  '{region_name:$region, instance_type_name:$type, ssh_key_names:[$key],
    quantity:1, name:"cgal-bench"}')"

launch_resp="$(curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d "${launch_payload}" "${API}/instance-operations/launch")"
INSTANCE_ID="$(echo "${launch_resp}" | jq -r '.data.instance_ids[0]')"
[ -n "${INSTANCE_ID}" ] && [ "${INSTANCE_ID}" != "null" ] \
  || die "launch failed: ${launch_resp}"
say "instance id: ${INSTANCE_ID}"

cleanup() {
  local rc=$?
  if [ "${KEEP_ALIVE}" = "true" ]; then
    say "KEEP_ALIVE=true, leaving instance ${INSTANCE_ID} running."
    say "ssh:    ssh -i ${SSH_PRIVATE_KEY} ubuntu@${IP:-<pending>}"
    say "kill:   ${HERE}/teardown.sh ${INSTANCE_ID}"
    return ${rc}
  fi
  say "terminating instance ${INSTANCE_ID}..."
  curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' \
    -d "$(jq -n --arg id "${INSTANCE_ID}" '{instance_ids:[$id]}')" \
    "${API}/instance-operations/terminate" >/dev/null || true
  return ${rc}
}
trap cleanup EXIT

# ---- 4. Poll until the instance is active and has an IP --------------------
say "waiting for instance to become active..."
IP=""
for _ in $(seq 1 60); do
  inst_json="$(curl -fsS "${AUTH[@]}" "${API}/instances/${INSTANCE_ID}")"
  status="$(echo "${inst_json}" | jq -r '.data.status')"
  IP="$(echo "${inst_json}" | jq -r '.data.ip // empty')"
  if [ "${status}" = "active" ] && [ -n "${IP}" ]; then
    break
  fi
  printf '  status=%s ip=%s\n' "${status}" "${IP:-<pending>}"
  sleep 10
done
[ -n "${IP}" ] || die "instance never became active"
say "instance is up at ${IP}"

# Wait for SSH to actually answer (Lambda often beats sshd by a few seconds).
say "waiting for sshd..."
for _ in $(seq 1 30); do
  if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
        -o UserKnownHostsFile=/dev/null \
        -i "${SSH_PRIVATE_KEY}" "ubuntu@${IP}" true 2>/dev/null; then
    break
  fi
  sleep 5
done

SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -i "${SSH_PRIVATE_KEY}" "ubuntu@${IP}")
SCP=(scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
     -i "${SSH_PRIVATE_KEY}")

# ---- 5. Push bootstrap script and run it ----------------------------------
say "uploading bootstrap script..."
"${SCP[@]}" "${HERE}/bootstrap.sh" "ubuntu@${IP}:/tmp/bootstrap.sh"
"${SSH[@]}" 'chmod +x /tmp/bootstrap.sh'

say "starting benchmark (logs streamed below)..."
"${SSH[@]}" \
  "CGAL_REF='${CGAL_REF}' MONTY_REF='${MONTY_REF}' \
   SUITE='${SUITE}' N_EVAL_EPOCHS='${N_EVAL_EPOCHS}' \
   /tmp/bootstrap.sh 2>&1 | tee /tmp/cgal-bench.log"

# ---- 6. Pull results -------------------------------------------------------
mkdir -p "${RESULTS_DIR}"
say "fetching logs and results into ${RESULTS_DIR}..."
"${SCP[@]}" -r "ubuntu@${IP}:/home/ubuntu/cgal/docker/logs" \
  "${RESULTS_DIR}/" || true
"${SCP[@]}" "ubuntu@${IP}:/tmp/cgal-bench.log" \
  "${RESULTS_DIR}/cgal-bench.log" || true

say "DONE. Results in ${RESULTS_DIR}"
if [ -f "${RESULTS_DIR}/logs/COMPARISON_REPORT.md" ]; then
  say "open ${RESULTS_DIR}/logs/COMPARISON_REPORT.md"
fi
