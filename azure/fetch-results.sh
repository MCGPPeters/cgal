#!/usr/bin/env bash
# Pull the benchmark logs + report back from the Azure VM.
set -euo pipefail
RG="${RG:-cgal-bench}"
USER="${USER_:-cgal}"
DEST="${DEST:-./azure-results}"
FQDN=$(az vm show -d -g "${RG}" --name "$(az vm list -g "${RG}" --query '[0].name' -o tsv)" --query fqdns -o tsv)
mkdir -p "${DEST}"
scp -o StrictHostKeyChecking=accept-new -r "${USER}@${FQDN}:/opt/cgal/docker/logs/." "${DEST}/"
echo "==> ${DEST}/COMPARISON_REPORT.md"
