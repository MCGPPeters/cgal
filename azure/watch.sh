#!/usr/bin/env bash
# Tail the cloud-init / benchmark log on the most recently provisioned VM.
set -euo pipefail
RG="${RG:-cgal-bench}"
USER="${USER_:-cgal}"
FQDN=$(az vm show -d -g "${RG}" --name "$(az vm list -g "${RG}" --query '[0].name' -o tsv)" --query fqdns -o tsv)
ssh -o StrictHostKeyChecking=accept-new "${USER}@${FQDN}" tail -f /var/log/cgal-bench.log
