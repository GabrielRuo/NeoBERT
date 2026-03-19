# Full-featured Modal-like dev image that installs project deps and creates Modal-style mountpoints.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /workspace

# system deps commonly needed for building wheels and runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates libffi-dev liblzma-dev \
      libbz2-dev libssl-dev zlib1g-dev \
 && rm -rf /var/lib/apt/lists/*

# copy minimal metadata and source so editable install can succeed at build time
COPY pyproject.toml /workspace/
COPY src /workspace/src

# upgrade pip and install project in editable mode with modal/dev extras
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install -e ".[modal,dev]" \
 || (echo "editable install failed; trying fallback to install project deps only" && pip install --no-deps -e /workspace)

# Create Modal-like mountpoints and caches
RUN mkdir -p /workspace/conf /workspace/src /runs /data /cache/hf

# Copy checkpoint into image
COPY ./runs/logs/checkpoints/mop_2025-12-02_16-36-59/model_checkpoints/40000/state_dict.pt /runs/logs/checkpoints/mop_2025-12-02_16-36-59/model_checkpoints/40000/state_dict.pt

# Default envs (override in docker-compose or at runtime)
ENV TRANSFORMERS_OFFLINE=0 \
    HF_HUB_OFFLINE=0 \
    HF_DATASETS_OFFLINE=0 \
    WANDB_MODE=offline \
    HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf

WORKDIR /workspace

# Default interactive shell
CMD ["/bin/bash"]