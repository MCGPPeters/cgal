#!/usr/bin/env bash
# Survey GPU + CPU quota across the regions where CGAL/Monty is most
# likely to fit, so you can pick a {region, SKU} pair that already has
# capacity instead of waiting on a quota request.
#
# Usage:
#   ./find-quota.sh                 # default region + family list
#   REGIONS="westeurope eastus2" ./find-quota.sh
set -euo pipefail

REGIONS="${REGIONS:-northeurope swedencentral westeurope uksouth francecentral eastus2 southcentralus}"

# (display name, quota family value, min vCPUs we need)
FAMILIES=(
    "T4 GPU         | standardNCASv3T4Family   | 4"
    "V100 GPU       | standardNCSv3Family      | 6"
    "A10 GPU        | standardNVADSA10v5Family | 6"
    "M60 GPU        | standardNVSv3Family      | 12"
    "Dsv5 (CPU)     | standardDSv5Family       | 8"
)

if ! command -v az >/dev/null; then
    echo "az CLI not found. Install with: brew install azure-cli" >&2
    exit 127
fi

printf '%-18s %-26s %-15s %-7s %-7s %-7s\n' "FAMILY" "QUOTA NAME" "REGION" "LIMIT" "USED" "OK?"
printf '%.0s-' {1..90}; echo
for region in ${REGIONS}; do
    for row in "${FAMILIES[@]}"; do
        IFS='|' read -r label family minv <<<"$(echo "$row" | tr -d ' ')"
        usage_json=$(az vm list-usage -l "$region" \
            --query "[?name.value=='$family'] | [0]" -o json 2>/dev/null || echo "{}")
        limit=$(echo "$usage_json" | jq -r '.limit // 0')
        used=$(echo "$usage_json"  | jq -r '.currentValue // 0')
        avail=$(( limit - used ))
        ok="-"
        if (( avail >= minv )); then ok="YES"; fi
        printf '%-18s %-26s %-15s %-7s %-7s %-7s\n' "$label" "$family" "$region" "$limit" "$used" "$ok"
    done
done
echo
echo "Pick the first row marked YES, then:"
echo "  LOCATION=<region> VM_SIZE=<sku> ./deploy.sh"
