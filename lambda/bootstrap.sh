#!/usr/bin/env bash
# Runs ON the Lambda Labs instance after it's up.
# The Lambda image (lambda-stack-22-04) ships with NVIDIA drivers, CUDA,
# Docker, and the NVIDIA container toolkit preinstalled. We just clone +
# build + run.
set -euxo pipefail

# Lambda's image has docker installed but ubuntu isn't in the docker group
# until next login. Use sudo for the rest of this session, forcing HOME to
# /home/ubuntu so compose's `~/tbp/...` volume references resolve correctly
# (sudo otherwise resets HOME to /root, which would create root-owned
# directories the container's mambauser can't write to).
DOCKER="sudo HOME=${HOME} docker"

CGAL_REF="${CGAL_REF:-main}"
MONTY_REF="${MONTY_REF:-cgal/main}"
SUITE="${SUITE:-smoke}"
N_EVAL_EPOCHS="${N_EVAL_EPOCHS:-}"

cd "${HOME}"
[ -d cgal ] || git clone --branch "${CGAL_REF}" --depth 1 \
  https://github.com/MCGPPeters/cgal.git
[ -d tbp.monty ] || git clone --branch "${MONTY_REF}" --depth 1 \
  https://github.com/MCGPPeters/tbp.monty.git

mkdir -p "${HOME}/tbp/data" "${HOME}/tbp/results/monty/pretrained_models"

cd "${HOME}/cgal/docker"

# Build the monty image (has habitat-sim, micromamba env, etc.)
$DOCKER compose -f docker-compose.yml build monty

# Download YCB + pretrained models inside the image (needs habitat-sim).
$DOCKER compose -f docker-compose.yml run --rm --entrypoint bash monty -lc \
  'micromamba run -n base /workspace/cgal/docker/download_ycb.sh'

# Run the suite. GPU is auto-detected via the nvidia runtime (preinstalled).
SUITE="${SUITE}" N_EVAL_EPOCHS="${N_EVAL_EPOCHS}" \
  $DOCKER compose -f docker-compose.yml up --abort-on-container-exit

# Hand logs back to ubuntu so scp can pull them.
sudo chown -R ubuntu:ubuntu "${HOME}/cgal/docker/logs" 2>/dev/null || true
