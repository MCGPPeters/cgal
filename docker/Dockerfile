# syntax=docker/dockerfile:1.6
#
# Monty + Habitat-Sim runtime image.
#
# - Built on micromamba so we can pull `habitat-sim` from the AIHabitat
#   conda channel (the official, supported install path on Linux x86_64).
# - Adds an OpenTelemetry SDK so the benchmark runner can ship traces
#   and metrics to the OTel collector defined in docker-compose.yml.
# - The actual Monty source tree and the CGAL repo are bind-mounted at
#   runtime — see docker-compose.yml — so iterating on code does not
#   require a rebuild.
#
ARG MAMBA_DOCKERFILE_ACTIVATE=1
# habitat-sim only ships linux/amd64 conda builds (no arm64), so pin
# the platform explicitly. On Apple Silicon this runs under QEMU and
# is significantly slower — fine for smoke runs, painful for full matrices.
FROM --platform=linux/amd64 mambaorg/micromamba:1.5.10

# Apt deps for habitat (headless GL stack) + git for editable installs.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential \
        libgl1 libglib2.0-0 libxrender1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*
USER mambauser

# Conda env: habitat-sim from aihabitat, scientific stack from conda-forge.
COPY --chown=mambauser:mambauser env.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml \
    && micromamba clean --all -y

# OpenTelemetry SDK + exporters (pip; not on conda-forge for this combo).
RUN micromamba run -n base pip install --no-cache-dir \
        opentelemetry-api==1.27.0 \
        opentelemetry-sdk==1.27.0 \
        opentelemetry-exporter-otlp==1.27.0

ENV PYTHONUNBUFFERED=1 \
    MONTY_DATA=/data \
    MONTY_MODELS=/models \
    MONTY_LOGS=/logs \
    WANDB_MODE=offline \
    WANDB_DIR=/logs/wandb \
    OTEL_SERVICE_NAME=monty \
    OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317 \
    OTEL_EXPORTER_OTLP_PROTOCOL=grpc

WORKDIR /workspace/monty

# The Monty source tree is bind-mounted at /workspace/monty by
# docker-compose. CGAL is mounted at /workspace/cgal. We install Monty
# editable on first run via the entrypoint.
COPY --chown=mambauser:mambauser entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
