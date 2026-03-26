FROM nvcr.io/nvidia/pytorch:26.02-py3

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies, preserving the container's native
# PyTorch/torchvision/torchaudio (required for Transformer Engine compatibility)
COPY pyproject.toml .
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --no-deps -e . && \
    pip install --no-cache-dir -r requirements-docker.txt

# Install flash attention
RUN MAX_JOBS=8 pip install --no-cache-dir --no-build-isolation flash-attn

# Default working directory for mounted code
WORKDIR /workspace
