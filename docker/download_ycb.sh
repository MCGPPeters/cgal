#!/usr/bin/env bash
# Download YCB dataset (via habitat-sim's downloader) and pretrained Monty
# models needed for the small `randrot_noise_10distinctobj_surf_agent`
# benchmark.
#
# Outputs:
#   ${MONTY_DATA:-$HOME/tbp/data}/habitat/objects/ycb/...
#   ${MONTY_MODELS:-$HOME/tbp/results/monty/pretrained_models}/pretrained_ycb_v12/...
#
# Usage:
#   ./download_ycb.sh                # download both into default ~/tbp paths
#   MONTY_DATA=/data MONTY_MODELS=/models ./download_ycb.sh
#
# Requires: python with `habitat_sim` importable, curl, tar.
set -euo pipefail

DATA_DIR="${MONTY_DATA:-$HOME/tbp/data}"
MODELS_DIR="${MONTY_MODELS:-$HOME/tbp/results/monty/pretrained_models}"

mkdir -p "${DATA_DIR}/habitat" "${MODELS_DIR}"

echo "==> YCB data → ${DATA_DIR}/habitat"
if [[ -d "${DATA_DIR}/habitat/objects/ycb" ]]; then
    echo "    (already present, skipping)"
else
    python -m habitat_sim.utils.datasets_download \
        --uids ycb \
        --data-path "${DATA_DIR}/habitat"
fi

echo "==> pretrained_ycb_v12 → ${MODELS_DIR}"
if [[ -d "${MODELS_DIR}/pretrained_ycb_v12" ]]; then
    echo "    (already present, skipping)"
else
    curl -L "https://tbp-pretrained-models-public-c9c24aef2e49b897.s3.us-east-2.amazonaws.com/tbp.monty/pretrained_ycb_v12.tgz" \
        | tar -xzf - -C "${MODELS_DIR}"
fi

echo "==> done"
ls -1 "${DATA_DIR}/habitat/objects/ycb" 2>/dev/null | head -5 || true
ls -1 "${MODELS_DIR}/pretrained_ycb_v12" 2>/dev/null | head -5 || true
