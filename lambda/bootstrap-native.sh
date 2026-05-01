#!/usr/bin/env bash
# Docker-free bootstrap. Runs ON the Lambda Labs (or any Ubuntu 22.04+)
# instance with NVIDIA drivers + CUDA already installed (lambda-stack
# image satisfies this).
#
# What it does:
#   1. Install micromamba (no sudo needed).
#   2. Clone cgal + tbp.monty.
#   3. Create a single conda env with habitat-sim 0.3.1 (headless, GPU)
#      and the Monty deps.
#   4. pip install -e tbp.monty + cgal.
#   5. Download YCB + pretrained models.
#   6. Run the suite directly (no Docker, no OTel collector).
#      OTel SDK exports are wired to the never-listening localhost:4317;
#      the BatchSpanProcessor logs and drops them silently.
#
# Env (all optional):
#   CGAL_REF=main, MONTY_REF=cgal/main, SUITE=smoke, N_EVAL_EPOCHS=, RUN_ONLY=both
set -euxo pipefail

CGAL_REF="${CGAL_REF:-main}"
MONTY_REF="${MONTY_REF:-cgal/main}"
SUITE="${SUITE:-smoke}"
N_EVAL_EPOCHS="${N_EVAL_EPOCHS:-}"
RUN_ONLY="${RUN_ONLY:-both}"

cd "${HOME}"

# --- 0. system deps ---------------------------------------------------------
# git-lfs: required by habitat-sim's YCB downloader.
# libnvidia-gl-*-server + libegl1: lambda-stack ships only the *server*
#   NVIDIA driver, which omits libEGL_nvidia / libGLX_nvidia. Without
#   these, habitat-sim's WindowlessEglApplication fails with
#   "cannot get default EGL display: EGL_BAD_PARAMETER".
NEED_APT=()
command -v git-lfs >/dev/null 2>&1 || NEED_APT+=(git-lfs)
ldconfig -p | grep -q libEGL_nvidia || {
  DRV_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \
                | head -1 | cut -d. -f1)"
  NEED_APT+=("libnvidia-gl-${DRV_MAJOR}-server" libegl1)
}
if [ ${#NEED_APT[@]} -gt 0 ]; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${NEED_APT[@]}"
  command -v git-lfs >/dev/null 2>&1 && git lfs install || true
fi

# --- 1. micromamba -----------------------------------------------------------
if ! command -v micromamba >/dev/null 2>&1; then
  mkdir -p "${HOME}/.local/bin"
  curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C "${HOME}/.local" bin/micromamba
  export PATH="${HOME}/.local/bin:${PATH}"
fi
export PATH="${HOME}/.local/bin:${PATH}"
export MAMBA_ROOT_PREFIX="${HOME}/micromamba"
eval "$(micromamba shell hook -s bash)"

# --- 2. clone repos ----------------------------------------------------------
[ -d cgal ]      || git clone --branch "${CGAL_REF}"  --depth 1 https://github.com/MCGPPeters/cgal.git
[ -d tbp.monty ] || git clone --branch "${MONTY_REF}" --depth 1 https://github.com/MCGPPeters/tbp.monty.git

# --- 3. env ------------------------------------------------------------------
if ! micromamba env list | grep -q '^\s*monty\s'; then
  micromamba create -y -n monty -f cgal/docker/env.yml
fi
micromamba activate monty

# --- 4. editable installs + OTel sdk ----------------------------------------
# Mirror the Docker entrypoint: install monty + cgal with --no-deps, then add
# only the deps Monty actually uses at runtime. This sidesteps torch-sparse,
# whose setup.py imports torch at install time and breaks `pip install -e`.
pip install --no-input --no-deps -e ./tbp.monty -e ./cgal
# Pin numpy<2 because the conda-installed `quaternion` package is built against
# numpy 1.x and breaks at import time on numpy 2.x.
pip install --no-input "numpy<2" \
  hydra-core==1.3.2 omegaconf==2.3.0 wandb \
  torch torchvision pillow scikit-image \
  "opencv-python<4.11" \
  pydantic>=2.10.6 sympy torch-geometric>=2.1.0.post1 \
  opentelemetry-api==1.27.0 \
  opentelemetry-sdk==1.27.0 \
  opentelemetry-exporter-otlp==1.27.0
# torch 2.x ships with numpy 2 wheels; force-downgrade numpy if pip resolved up.
pip install --no-input --force-reinstall --no-deps "numpy<2"

# torch >=2.6 defaults `torch.load(..., weights_only=True)` which rejects
# Monty's pretrained checkpoints (they pickle GraphObjectModel instances).
# Patch in `weights_only=False` since we trust our own checkpoints.
MONTY_EXP_PY="${HOME}/tbp.monty/src/tbp/monty/frameworks/experiments/monty_experiment.py"
if grep -q "torch.load(model_path)" "${MONTY_EXP_PY}"; then
  sed -i "s|torch.load(model_path)|torch.load(model_path, weights_only=False)|g" \
    "${MONTY_EXP_PY}"
fi

# --- 5. data + models --------------------------------------------------------
mkdir -p "${HOME}/tbp/data/habitat" \
         "${HOME}/tbp/results/monty/pretrained_models" \
         "${HOME}/cgal/docker/logs" \
         "${HOME}/cgal/docker/otel-out"
bash "${HOME}/cgal/docker/download_ycb.sh"

# --- 6. run the suite --------------------------------------------------------
# Adapt the in-container paths to the on-host paths.
# The original run_benchmark.sh hard-codes /workspace/... and /logs.
# We rewrite into a temp script with the right roots.
RUN_DIR="${HOME}/cgal/run"
mkdir -p "${RUN_DIR}"
sed \
  -e "s|/workspace/monty|${HOME}/tbp.monty|g" \
  -e "s|/workspace/cgal|${HOME}/cgal|g" \
  -e "s|/workspace/run_with_otel.py|${HOME}/cgal/docker/run_with_otel.py|g" \
  -e "s|/workspace/generate_report.py|${HOME}/cgal/docker/generate_report.py|g" \
  -e "s|/workspace|${HOME}/cgal/docker|g" \
  -e "s|/logs|${HOME}/cgal/docker/logs|g" \
  -e "s|/otel-out|${HOME}/cgal/docker/otel-out|g" \
  "${HOME}/cgal/docker/run_benchmark.sh" > "${RUN_DIR}/run_benchmark.sh"
chmod +x "${RUN_DIR}/run_benchmark.sh"

# Same path patch for run_with_otel.py (it does runpy.run_path('/workspace/monty/run.py')).
sed -i "s|/workspace/monty/run.py|${HOME}/tbp.monty/run.py|g" \
  "${HOME}/cgal/docker/run_with_otel.py"

# Disable OTel network export — no collector running, no need for warnings.
export OTEL_SDK_DISABLED=true

# habitat-sim EGL on cloud GPUs: restrict EGL to the NVIDIA ICD so that
# WindowlessEglApplication's CUDA-device matching doesn't get confused by
# the mesa software EGL devices.
export __EGL_VENDOR_LIBRARY_FILENAMES="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# wandb: experiments wire up WandbWrapper unconditionally; disable network
# logging so they don't fail looking for an API key.
export WANDB_MODE=disabled

SUITE="${SUITE}" N_EVAL_EPOCHS="${N_EVAL_EPOCHS}" RUN_ONLY="${RUN_ONLY}" \
  bash "${RUN_DIR}/run_benchmark.sh"

echo "[bootstrap] done. Report at ${HOME}/cgal/docker/logs/COMPARISON_REPORT.md"
