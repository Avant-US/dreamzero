FROM nvcr.io/nvidia/pytorch:26.02-py3

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cu129

# Install flash attention
RUN MAX_JOBS=8 pip install --no-cache-dir --no-build-isolation flash-attn

# Default working directory for mounted code
WORKDIR /workspace
