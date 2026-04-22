# Modal-like dev image with configurable dependency profiles.
FROM python:3.11-slim

# Install build-essential for C++ compiler (required by PyTorch and other Python packages)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

WORKDIR /workspace

# Keep system deps minimal for development containers.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
     git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# copy minimal metadata and source so editable install can succeed at build time
COPY pyproject.toml /workspace/
COPY src /workspace/src

# Choose extras at build time. Default is lightweight dev profile.
# Examples:
#   --build-arg NEOBERT_EXTRAS="dev,modal,train_base"
#   --build-arg NEOBERT_EXTRAS="dev,modal,train_base,train_gpu"
ARG NEOBERT_EXTRAS="dev,modal,train_base"

# Torch install mode:
#   cpu  -> install CPU-only wheels from PyTorch CPU index (default)
#   cuda -> install default Linux wheel set (can download large CUDA deps)
#   none -> skip torch installation
ARG TORCH_BACKEND="cpu"

# Install torch backend first so transitive torch requirements (e.g. via peft)
# do not pull default Linux CUDA wheels (nvidia_* packages) during editable install.
# In cpu mode we also expose the PyTorch CPU index while installing extras.
RUN python -m pip install --upgrade pip setuptools wheel \
 && if [ "${TORCH_BACKEND}" = "cpu" ]; then \
            pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.*"; \
        elif [ "${TORCH_BACKEND}" = "cuda" ]; then \
            pip install "torch==2.5.*"; \
        else \
            echo "Skipping torch preinstall (TORCH_BACKEND=${TORCH_BACKEND})"; \
        fi \
 && if [ "${TORCH_BACKEND}" = "cpu" ]; then \
            PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu pip install -e ".[${NEOBERT_EXTRAS}]"; \
        else \
            pip install -e ".[${NEOBERT_EXTRAS}]"; \
        fi \
 || (echo "editable install failed; trying fallback to install project deps only" && pip install --no-deps -e /workspace)

# Create Modal-like mountpoints and caches
RUN mkdir -p /workspace/conf /workspace/src /runs /data /workspace/.cache/hf

# Checkpoints are expected via mounted volumes (for faster, smaller builds).

# Default envs (override in docker-compose or at runtime)
ENV TRANSFORMERS_OFFLINE=0 \
    HF_HUB_OFFLINE=0 \
    HF_DATASETS_OFFLINE=0 \
    WANDB_MODE=offline \
    HF_HOME=/workspace/.cache/hf \
    TRANSFORMERS_CACHE=/workspace/.cache/hf

WORKDIR /workspace

# Default interactive shell
CMD ["/bin/bash"]