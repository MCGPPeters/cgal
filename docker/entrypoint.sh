#!/usr/bin/env bash
# Container entrypoint.
#
# 1. Install the bind-mounted Monty source as an editable package the
#    first time the container starts (idempotent: skips if already
#    importable).
# 2. Install the bind-mounted CGAL package as editable so Monty can
#    `import cgal` for the learning-module mixins.
# 3. Make sure the `tbp` data/results dirs exist on the mounted volume.
# 4. Hand off to the user-supplied command (default: bash).
set -euo pipefail

cd /workspace/monty

if ! micromamba run -n base python -c "import tbp.monty" 2>/dev/null; then
    echo "[entrypoint] Installing tbp.monty (editable)…"
    micromamba run -n base pip install --no-deps -e .
    micromamba run -n base pip install --no-cache-dir \
        hydra-core==1.3.2 omegaconf==2.3.0 wandb torch torchvision \
        pillow scikit-image
fi

if [ -d /workspace/cgal ] && ! micromamba run -n base python -c "import cgal" 2>/dev/null; then
    echo "[entrypoint] Installing cgal (editable)…"
    micromamba run -n base pip install --no-deps -e /workspace/cgal
fi

mkdir -p "${MONTY_DATA}" "${MONTY_MODELS}" "${MONTY_LOGS}" "${WANDB_DIR}"

exec micromamba run -n base "$@"
