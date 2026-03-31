#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
IMAGE_NAME="dreamzero"
CONTAINER_NAME="dreamzero-dev"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT_DIR="/mnt/r/huggingface_checkpoints"
HF_REPO="GEAR-Dreams/DreamZero-DROID"
GPUS="${1:-0,1}"

# ─── Step 1: Build Docker image if it doesn't exist ─────────────────────────
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "==> Docker image '$IMAGE_NAME' not found. Building..."
    docker build -t "$IMAGE_NAME" "$REPO_DIR"
else
    echo "==> Docker image '$IMAGE_NAME' already exists. Skipping build."
fi

# ─── Step 2: Start or reattach container ─────────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        echo "==> Container '$CONTAINER_NAME' is running. Attaching..."
        docker exec -it "$CONTAINER_NAME" bash
        exit 0
    else
        echo "==> Container '$CONTAINER_NAME' exists but stopped. Starting..."
        docker start -ai "$CONTAINER_NAME"
        exit 0
    fi
fi

echo "==> Creating new container '$CONTAINER_NAME'..."
docker run --gpus "\"device=$GPUS\"" -it \
    --net=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p 5000:5000 \
    -v "$REPO_DIR":/workspace \
    -v /mnt/r:/checkpoints \
    -w /workspace \
    --name "$CONTAINER_NAME" \
    "$IMAGE_NAME" \
    bash -c "
        set -e

        # ── Step 3: Reinstall editable package (mount overlays container install) ──
        echo '==> Installing editable package...'
        pip install --no-deps -e .

        # ── Step 4: Download checkpoints if not present ──
        if [ ! -f /checkpoints/huggingface_checkpoints/experiment_cfg/conf.yaml ]; then
            echo '==> Downloading model checkpoints from HuggingFace...'
            python -c \"
from huggingface_hub import snapshot_download
snapshot_download('$HF_REPO', local_dir='/checkpoints/huggingface_checkpoints')
\"
        else
            echo '==> Checkpoints already downloaded.'
        fi

        # ── Step 5: Create symlink ──
        if [ ! -L /workspace/huggingface_checkpoints ]; then
            ln -sf /checkpoints/huggingface_checkpoints /workspace/huggingface_checkpoints
            echo '==> Symlinked /checkpoints/huggingface_checkpoints -> /workspace/huggingface_checkpoints'
        fi

        echo ''
        echo '=========================================='
        echo '  DreamZero container ready!'
        echo '  Checkpoints: /workspace/huggingface_checkpoints'
        echo '=========================================='
        echo ''

        exec bash
    "
