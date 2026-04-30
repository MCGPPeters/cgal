#!/usr/bin/env bash
# Provision the GPU VM, run the CGAL/Monty benchmark matrix on it,
# pull results back via scp, and (optionally) deallocate the VM.
#
# Required env (or pass on the command line):
#   RG=cgal-bench LOCATION=swedencentral SSH_PUB=$HOME/.ssh/id_ed25519.pub
#
# Optional:
#   SUITE=noise|smoke|full   (default: noise)
#   VM_SIZE=Standard_NC4as_T4_v3
#   USE_SPOT=true
#   SKIP_QUOTA_CHECK=false
set -euo pipefail

RG="${RG:-cgal-bench}"
LOCATION="${LOCATION:-swedencentral}"
SSH_PUB_PATH="${SSH_PUB:-$HOME/.ssh/id_ed25519.pub}"
SUITE="${SUITE:-noise}"
VM_SIZE="${VM_SIZE:-Standard_NC4as_T4_v3}"
USE_SPOT="${USE_SPOT:-true}"

if ! command -v az >/dev/null; then
    echo "az CLI not found. Install with: brew install azure-cli" >&2
    exit 127
fi

echo "[deploy] target subscription: $(az account show --query name -o tsv 2>/dev/null || echo 'NONE — run az login first')"

# --- Quota pre-flight (very common cause of provisioning failure) -----
if [[ "${SKIP_QUOTA_CHECK:-false}" != "true" ]]; then
    case "${VM_SIZE}" in
        Standard_NC*T4*)         family="standardNCASv3T4Family" ;;
        Standard_NC*v3)          family="standardNCSv3Family" ;;
        Standard_NV*A10_v5)      family="standardNVADSA10v5Family" ;;
        Standard_D*s_v5)         family="standardDSv5Family" ;;
        *)                       family="" ;;
    esac
    if [[ -n "${family}" ]]; then
        echo "[deploy] checking '${family}' quota in ${LOCATION}…"
        az vm list-usage -l "${LOCATION}" --query "[?name.value=='${family}'] | [0]" -o table || true
        echo "[deploy] (zero current usage + zero limit means 'request quota in the portal first')"
    fi
fi

az group create -n "${RG}" -l "${LOCATION}" >/dev/null
echo "[deploy] resource group ready: ${RG} (${LOCATION})"

DEPLOY_OUT=$(az deployment group create \
    -g "${RG}" \
    -f "$(dirname "$0")/main.bicep" \
    -p sshPublicKey="$(cat "${SSH_PUB_PATH}")" \
    -p suite="${SUITE}" \
    -p vmSize="${VM_SIZE}" \
    -p useSpot="${USE_SPOT}" \
    -p nEvalEpochs="${N_EVAL_EPOCHS:-}" \
    -o json)

SSH_CMD=$(echo "${DEPLOY_OUT}" | jq -r '.properties.outputs.sshCommand.value')
WATCH=$(echo "${DEPLOY_OUT}" | jq -r '.properties.outputs.watchLog.value')
FETCH=$(echo "${DEPLOY_OUT}" | jq -r '.properties.outputs.fetchResults.value')

cat <<EOF
[deploy] VM up. Bootstrap (drivers + docker + data + benchmark) is
running in the background; the matrix takes 30–90 min.

  Watch progress:   ${WATCH}
  SSH in:           ${SSH_CMD}
  When finished:    ${FETCH}
                    (then open ./azure-results/COMPARISON_REPORT.md)
  Tear down:        az group delete -n ${RG} --yes --no-wait

EOF
