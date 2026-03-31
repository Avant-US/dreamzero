#!/usr/bin/env bash
#
# docker_bench.sh — Build, setup, and run the DreamZero inference benchmark.
#
# Usage:
#   ./scripts/inference/docker_bench.sh build     # Build the Docker image
#   ./scripts/inference/docker_bench.sh setup      # Create container + install + download weights
#   ./scripts/inference/docker_bench.sh start      # Start the inference server
#   ./scripts/inference/docker_bench.sh test       # Run the test client
#   ./scripts/inference/docker_bench.sh stop       # Stop and remove the container
#   ./scripts/inference/docker_bench.sh gpu        # Show nvidia-smi
#
# Environment variables (override defaults):
#   MODEL_PATH            — path to model weights inside container (default: /mnt/huggingface_checkpoints)
#   ATTENTION_BACKEND     — FA2, FA3, TE, or torch (default: FA2)
#   DYNAMIC_CACHE         — true/false (default: true)
#   TORCH_COMPILE         — true=disabled, false=enabled (default: false, i.e. encoder/VAE compile ON)
#   COMPILE_DIT           — true/false, compile DiT _forward_blocks (default: false, REGRESSION)
#   CUDA_GRAPH_DIT        — true/false, CUDA graph capture for DiT diffusion loop (default: false)
#   NUM_DIT_STEPS         — number of diffusion steps (default: 16)
#   PORT                  — server port (default: 5000)
#   SP_SIZE               — sequence parallelism degree (default: 1). Must divide NUM_GPUS and 40 (num_heads).
#   NUM_GPUS              — number of GPUs to use (default: 2)
#   CONTAINER_NAME        — docker container name (default: dreamzero-bench)

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-dreamzero-bench}"
MODEL_PATH="${MODEL_PATH:-/mnt/huggingface_checkpoints}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FA2}"
DYNAMIC_CACHE="${DYNAMIC_CACHE:-true}"
TORCH_COMPILE="${TORCH_COMPILE:-false}"
COMPILE_DIT="${COMPILE_DIT:-false}"
CUDA_GRAPH_DIT="${CUDA_GRAPH_DIT:-false}"
NUM_DIT_STEPS="${NUM_DIT_STEPS:-16}"
PORT="${PORT:-5000}"
SP_SIZE="${SP_SIZE:-1}"
NUM_GPUS="${NUM_GPUS:-2}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

cmd_build() {
    echo "==> Building dreamzero Docker image..."
    docker build -t dreamzero "$REPO_DIR"
}

cmd_setup() {
    echo "==> Creating container ${CONTAINER_NAME}..."
    docker run --gpus all -d \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p "${PORT}:${PORT}" \
        -v "${REPO_DIR}:/workspace" \
        -v /mnt:/mnt \
        -w /workspace \
        --name "${CONTAINER_NAME}" \
        dreamzero \
        sleep infinity

    echo "==> Installing package (editable, no deps)..."
    docker exec "${CONTAINER_NAME}" pip install --no-deps -e .

    if [ ! -d "/mnt/huggingface_checkpoints" ]; then
        echo "==> Downloading model weights to /mnt/huggingface_checkpoints..."
        docker exec "${CONTAINER_NAME}" python -c \
            "from huggingface_hub import snapshot_download; snapshot_download('GEAR-Dreams/DreamZero-DROID', local_dir='/mnt/huggingface_checkpoints')"
    else
        echo "==> Model weights already exist at /mnt/huggingface_checkpoints, skipping download."
    fi

    echo "==> Setup complete. Run: $0 start"
}

cmd_start() {
    echo "==> Starting inference server on port ${PORT}..."
    echo "    ATTENTION_BACKEND=${ATTENTION_BACKEND}"
    echo "    DYNAMIC_CACHE_SCHEDULE=${DYNAMIC_CACHE}"
    echo "    DISABLE_TORCH_COMPILE=${TORCH_COMPILE}"
    echo "    COMPILE_DIT=${COMPILE_DIT}"
    echo "    CUDA_GRAPH_DIT=${CUDA_GRAPH_DIT}"
    echo "    NUM_DIT_STEPS=${NUM_DIT_STEPS}"
    echo "    MODEL_PATH=${MODEL_PATH}"
    echo "    SP_SIZE=${SP_SIZE}"
    echo "    NUM_GPUS=${NUM_GPUS}"

    # Build CUDA_VISIBLE_DEVICES list: 0,1,...,NUM_GPUS-1
    CUDA_DEVS=$(seq -s, 0 $((NUM_GPUS - 1)))

    docker exec -it "${CONTAINER_NAME}" bash -c "\
        DYNAMIC_CACHE_SCHEDULE=${DYNAMIC_CACHE} \
        DISABLE_TORCH_COMPILE=${TORCH_COMPILE} \
        COMPILE_DIT=${COMPILE_DIT} \
        CUDA_GRAPH_DIT=${CUDA_GRAPH_DIT} \
        NUM_DIT_STEPS=${NUM_DIT_STEPS} \
        ATTENTION_BACKEND=${ATTENTION_BACKEND} \
        CUDA_VISIBLE_DEVICES=${CUDA_DEVS} \
        torchrun --standalone --nproc_per_node=${NUM_GPUS} \
            socket_test_optimized_AR.py \
            --port ${PORT} --enable-dit-cache --model-path ${MODEL_PATH} --sp-size ${SP_SIZE}"
}

cmd_test() {
    echo "==> Running test client..."
    docker exec -it "${CONTAINER_NAME}" python test_client_AR.py --port "${PORT}"
}

cmd_stop() {
    echo "==> Stopping and removing container ${CONTAINER_NAME}..."
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
}

cmd_gpu() {
    docker exec "${CONTAINER_NAME}" nvidia-smi
}

case "${1:-help}" in
    build) cmd_build ;;
    setup) cmd_setup ;;
    start) cmd_start ;;
    test)  cmd_test ;;
    stop)  cmd_stop ;;
    gpu)   cmd_gpu ;;
    *)
        echo "Usage: $0 {build|setup|start|test|stop|gpu}"
        echo ""
        echo "  build  — Build the Docker image"
        echo "  setup  — Create container, install deps, download model weights"
        echo "  start  — Launch the inference server (run test in another terminal)"
        echo "  test   — Run the test client against the running server"
        echo "  stop   — Stop and remove the container"
        echo "  gpu    — Show nvidia-smi"
        exit 1
        ;;
esac
