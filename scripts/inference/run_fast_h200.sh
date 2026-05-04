#!/usr/bin/env bash
#
# run_fast_h200.sh — Fast inference launcher for 8x H200 with SP=4.
#
# Optimized configuration:
#   - COMPILE_DIT=true (reduce-overhead CUDA graphs on DiT blocks)
#   - Encoder compile (torch.compile on text/image/VAE encoders)
#   - FA2 attention backend (TE falls back to FlashAttention-2 on conda)
#   - OVERLAP_VAE_DIT=true (overlap VAE encode with DiT)
#   - NO static KV cache (dynamic cache is faster due to no .item() graph breaks)
#   - KV_INIT_CACHE_THRESH=0 (no KV init skip — bit-exact KV conditioning)
#
# Numerical accuracy vs main branch (eager):
#   - max_err ~0.05, correlation 0.998
#   - Error sources: encoder compile (mode=default), FA2 vs SDPA, VAE overlap
#   - DiT itself is bit-exact (torch.compile + reduce-overhead = 0 error)
#
# Steady-state speed: ~0.33-0.49s per chunk (after compilation warmup)
# First-request latency: ~24-30s (one-time inductor compilation)
#
# Usage:
#   ./scripts/inference/run_fast_h200.sh start
#   ./scripts/inference/run_fast_h200.sh test
#   ./scripts/inference/run_fast_h200.sh bench
#
# Override defaults via env vars:
#   NUM_GPUS=4 SP_SIZE=2 PORT=5001 ./scripts/inference/run_fast_h200.sh start

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/localssd/dreamzero/huggingface_checkpoints}"
PORT="${PORT:-5000}"
SP_SIZE="${SP_SIZE:-4}"
NUM_GPUS="${NUM_GPUS:-8}"
NUM_DIT_STEPS="${NUM_DIT_STEPS:-16}"
FP8_INFERENCE="${FP8_INFERENCE:-false}"

# --- Core optimizations ---
COMPILE_DIT="${COMPILE_DIT:-true}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TE}"        # Falls back to FA2 on conda
DISABLE_TORCH_COMPILE="${DISABLE_TORCH_COMPILE:-false}"  # Compile encoders too
OVERLAP_VAE_DIT="${OVERLAP_VAE_DIT:-true}"

# --- Disabled optimizations (preserve accuracy) ---
STATIC_KV_CACHE="${STATIC_KV_CACHE:-false}"         # Dynamic cache is faster
KV_INIT_CACHE_THRESH="${KV_INIT_CACHE_THRESH:-0}"   # No KV init skip

# --- Compile settings ---
PYNCCL_ALLTOALL="${PYNCCL_ALLTOALL:-${COMPILE_DIT}}"
COMPILE_WARMUP_CHUNKS="${COMPILE_WARMUP_CHUNKS:-2}"
# max-autotune: benchmarks cuBLAS algorithms + triton configs at compile time,
# AND captures CUDA graphs. ~10-20% faster kernels vs reduce-overhead alone.
# Tradeoff: first compile is much slower (minutes), but cached by inductor.
COMPILE_DIT_MODE="${COMPILE_DIT_MODE:-max-autotune}"

# --- Other defaults ---
DYNAMIC_CACHE="${DYNAMIC_CACHE:-true}"
CUDA_GRAPH_DIT="${CUDA_GRAPH_DIT:-false}"
LOAD_TRT_ENGINE="${LOAD_TRT_ENGINE:-}"

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# CUDA_HOME resolution
ensure_cuda_home_stub() {
    local conda_env
    conda_env=$(python -c "import sys; print(sys.prefix)" 2>/dev/null || echo "")
    local real_cuda="${conda_env}/targets/x86_64-linux"
    if [ -x "${real_cuda}/bin/nvcc" ]; then
        export CUDA_HOME="${real_cuda}"
        return
    fi
    local stub_dir="${REPO_DIR}/.cuda_home_stub"
    local nvcc_path="${stub_dir}/bin/nvcc"
    if [ ! -x "${nvcc_path}" ]; then
        local cuda_ver
        cuda_ver=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "12.9")
        mkdir -p "${stub_dir}/bin"
        cat > "${nvcc_path}" <<EOF
#!/usr/bin/env bash
cat <<NVCC_OUT
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release ${cuda_ver}, V${cuda_ver}.0
NVCC_OUT
EOF
        chmod +x "${nvcc_path}"
    fi
    export CUDA_HOME="${stub_dir}"
}

cmd_start() {
    echo "==> Starting FAST inference server on port ${PORT}..."
    echo "    NUM_GPUS=${NUM_GPUS}"
    echo "    SP_SIZE=${SP_SIZE}            (CFG=$((NUM_GPUS / SP_SIZE)) x SP=${SP_SIZE})"
    echo "    ATTENTION_BACKEND=${ATTENTION_BACKEND}"
    echo "    COMPILE_DIT=${COMPILE_DIT} (mode=${COMPILE_DIT_MODE})"
    echo "    ENCODER_COMPILE=$([ "${DISABLE_TORCH_COMPILE}" = "false" ] && echo "true" || echo "false")"
    echo "    OVERLAP_VAE_DIT=${OVERLAP_VAE_DIT}"
    echo "    STATIC_KV_CACHE=${STATIC_KV_CACHE}"
    echo "    KV_INIT_CACHE_THRESH=${KV_INIT_CACHE_THRESH}"
    echo "    MODEL_PATH=${MODEL_PATH}"

    if [ ! -d "${MODEL_PATH}" ]; then
        echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
        exit 1
    fi

    ensure_cuda_home_stub
    echo "    CUDA_HOME=${CUDA_HOME}"

    HF_CACHE_BASE="${HF_CACHE_BASE:-/mnt/localssd/hf_cache}"
    mkdir -p "${HF_CACHE_BASE}" "${HF_CACHE_BASE}/xet"

    CUDA_DEVS=$(seq -s, 0 $((NUM_GPUS - 1)))

    cd "${REPO_DIR}"
    env \
        CUDA_HOME="${CUDA_HOME}" \
        HF_HOME="${HF_CACHE_BASE}" \
        HF_HUB_CACHE="${HF_CACHE_BASE}/hub" \
        HUGGINGFACE_HUB_CACHE="${HF_CACHE_BASE}/hub" \
        HF_XET_CACHE="${HF_CACHE_BASE}/xet" \
        DYNAMIC_CACHE_SCHEDULE="${DYNAMIC_CACHE}" \
        DISABLE_TORCH_COMPILE="${DISABLE_TORCH_COMPILE}" \
        COMPILE_DIT="${COMPILE_DIT}" \
        CUDA_GRAPH_DIT="${CUDA_GRAPH_DIT}" \
        STATIC_KV_CACHE="${STATIC_KV_CACHE}" \
        PYNCCL_ALLTOALL="${PYNCCL_ALLTOALL}" \
        NUM_DIT_STEPS="${NUM_DIT_STEPS}" \
        ATTENTION_BACKEND="${ATTENTION_BACKEND}" \
        FP8_INFERENCE="${FP8_INFERENCE}" \
        CUDA_VISIBLE_DEVICES="${CUDA_DEVS}" \
        NUM_GPUS="${NUM_GPUS}" \
        SP_SIZE="${SP_SIZE}" \
        COMPILE_DIT_MODE="${COMPILE_DIT_MODE}" \
        COMPILE_DIT_FULLGRAPH="${COMPILE_DIT_FULLGRAPH:-false}" \
        TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}" \
        TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS="${TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS:-TRITON,ATen}" \
        TORCHINDUCTOR_COORDINATE_DESCENT_TUNING="${TORCHINDUCTOR_COORDINATE_DESCENT_TUNING:-1}" \
        KV_INIT_CACHE_THRESH="${KV_INIT_CACHE_THRESH}" \
        OVERLAP_VAE_DIT="${OVERLAP_VAE_DIT}" \
        COMPILE_WARMUP_CHUNKS="${COMPILE_WARMUP_CHUNKS}" \
        CFG_SCALE="${CFG_SCALE:-5.0}" \
        NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-16}" \
        PROFILE_INFERENCE="${PROFILE_INFERENCE:-}" \
        PROFILE_LOG_FILE="${PROFILE_LOG_FILE:-/mnt/localssd/dreamzero_profile.jsonl}" \
        $([ -n "${LOAD_TRT_ENGINE}" ] && echo "LOAD_TRT_ENGINE=${LOAD_TRT_ENGINE}") \
        torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
            socket_test_optimized_AR.py \
            --port "${PORT}" \
            --enable-dit-cache \
            --model-path "${MODEL_PATH}" \
            --sp-size "${SP_SIZE}"
}

cmd_test() {
    echo "==> Running test client on port ${PORT}..."
    cd "${REPO_DIR}"
    python test_client_AR.py --port "${PORT}"
}

cmd_bench() {
    local report_dir="${REPORT_DIR:-${REPO_DIR}/bench_reports}"
    mkdir -p "${report_dir}"
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local tag="fast_sp${SP_SIZE}"
    local report_path="${report_dir}/${ts}_${tag}.json"
    local num_chunks="${NUM_CHUNKS:-15}"
    local warmup_chunks="${WARMUP_CHUNKS:-3}"
    local profile_log="${PROFILE_LOG_FILE:-/mnt/localssd/dreamzero_profile.jsonl}"

    echo "==> Benchmarking against server on port ${PORT}"
    echo "    report: ${report_path}"
    : > "${profile_log}" 2>/dev/null || true

    cd "${REPO_DIR}"
    python test_client_AR.py \
        --port "${PORT}" \
        --num-chunks "${num_chunks}" \
        --warmup-chunks "${warmup_chunks}" \
        --no-reset \
        --report "${report_path}" \
        --profile-log "${profile_log}"
}

case "${1:-help}" in
    start) cmd_start ;;
    test)  cmd_test ;;
    bench) cmd_bench ;;
    *)
        echo "Usage: $0 {start|test|bench}"
        echo ""
        echo "Fast inference config: COMPILE_DIT + encoder compile + FA2 + OVERLAP_VAE"
        echo "Steady-state: ~0.33-0.49s | Max error vs main: ~0.05 | Correlation: 0.998"
        echo ""
        echo "Key env overrides:"
        echo "  DISABLE_TORCH_COMPILE=true   Disable encoder compilation (bit-exact encoders)"
        echo "  OVERLAP_VAE_DIT=false         Disable VAE overlap (bit-exact VAE timing)"
        echo "  ATTENTION_BACKEND=torch       Use SDPA instead of FA2 (bit-exact attention)"
        echo "  COMPILE_DIT=false             Disable DiT compilation (fully eager, bit-exact)"
        echo ""
        echo "For bit-exact mode (0.55s, zero error):"
        echo "  DISABLE_TORCH_COMPILE=true ATTENTION_BACKEND=torch OVERLAP_VAE_DIT=false $0 start"
        exit 1
        ;;
esac
