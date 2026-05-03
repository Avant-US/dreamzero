#!/usr/bin/env bash
#
# run_sp_h200.sh — Native (conda) launcher for 8x H200 SP inference.
#
# Mirrors the env-var contract from scripts/inference/docker_bench.sh, but runs
# directly in the active Python environment (no Docker).
#
# Usage:
#   ./scripts/inference/run_sp_h200.sh start    # launch inference server
#   ./scripts/inference/run_sp_h200.sh test     # run test client (in another shell)
#
# Override defaults via env vars, e.g.:
#   NUM_GPUS=4 SP_SIZE=2 ./scripts/inference/run_sp_h200.sh start

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/localssd/dreamzero/huggingface_checkpoints}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TE}"
DYNAMIC_CACHE="${DYNAMIC_CACHE:-true}"
TORCH_COMPILE="${TORCH_COMPILE:-false}"
COMPILE_DIT="${COMPILE_DIT:-true}"
CUDA_GRAPH_DIT="${CUDA_GRAPH_DIT:-false}"
CUDA_GRAPH_DIT_MANUAL="${CUDA_GRAPH_DIT_MANUAL:-false}"
STATIC_KV_CACHE="${STATIC_KV_CACHE:-true}"
# Enable pynccl by default when COMPILE_DIT=true (needed for graph-safe NCCL in reduce-overhead)
PYNCCL_ALLTOALL="${PYNCCL_ALLTOALL:-${COMPILE_DIT}}"
NUM_DIT_STEPS="${NUM_DIT_STEPS:-16}"
PORT="${PORT:-5000}"
SP_SIZE="${SP_SIZE:-4}"
NUM_GPUS="${NUM_GPUS:-8}"
FP8_INFERENCE="${FP8_INFERENCE:-false}"
LOAD_TRT_ENGINE="${LOAD_TRT_ENGINE:-}"

# Optimization: skip KV init on continuation chunks (reuse from previous diffusion)
KV_INIT_CACHE_THRESH="${KV_INIT_CACHE_THRESH:-1}"
# Optimization: overlap VAE encode with DiT on continuation chunks
OVERLAP_VAE_DIT="${OVERLAP_VAE_DIT:-true}"
# Warmup: pre-compile CUDA graphs at startup (avoids slow first-client experience)
# 2 covers both unique shapes (initial + continuation). 0 = no warmup.
COMPILE_WARMUP_CHUNKS="${COMPILE_WARMUP_CHUNKS:-2}"

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# CUDA_HOME resolution —
#   Prefer the real cuda-toolkit installed via conda (has bin/nvcc, include/,
#   lib/, nvvm/ — full layout that torch.compile + triton expect).
#   Fall back to the stub if no real toolkit is present. The stub only
#   satisfies deepspeed's version probe and WILL break torch.compile /
#   cudagraph capture for anything non-trivial.
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
# Stub nvcc used only to satisfy deepspeed's CUDA version probe.
cat <<NVCC_OUT
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on stub
Cuda compilation tools, release ${cuda_ver}, V${cuda_ver}.0
Build stub
NVCC_OUT
EOF
        chmod +x "${nvcc_path}"
        echo "==> Created CUDA_HOME stub at ${stub_dir} (nvcc reports CUDA ${cuda_ver})"
    fi
    export CUDA_HOME="${stub_dir}"
}

cmd_start() {
    echo "==> Starting inference server on port ${PORT}..."
    echo "    NUM_GPUS=${NUM_GPUS}"
    echo "    SP_SIZE=${SP_SIZE}            (CFG=$((NUM_GPUS / SP_SIZE)) x SP=${SP_SIZE})"
    echo "    ATTENTION_BACKEND=${ATTENTION_BACKEND}"
    echo "    COMPILE_DIT=${COMPILE_DIT}"
    echo "    FP8_INFERENCE=${FP8_INFERENCE}"
    echo "    DYNAMIC_CACHE=${DYNAMIC_CACHE}"
    echo "    NUM_DIT_STEPS=${NUM_DIT_STEPS}"
    echo "    MODEL_PATH=${MODEL_PATH}"
    echo "    LOAD_TRT_ENGINE=${LOAD_TRT_ENGINE:-<none>}"

    if [ ! -d "${MODEL_PATH}" ]; then
        echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}" >&2
        exit 1
    fi

    if (( NUM_GPUS % SP_SIZE != 0 )); then
        echo "ERROR: NUM_GPUS (${NUM_GPUS}) must be divisible by SP_SIZE (${SP_SIZE})" >&2
        exit 1
    fi
    if (( 40 % SP_SIZE != 0 )); then
        echo "ERROR: SP_SIZE (${SP_SIZE}) must divide num_heads (40)" >&2
        exit 1
    fi

    ensure_cuda_home_stub
    echo "    CUDA_HOME=${CUDA_HOME}"

    # Redirect HuggingFace xet + hub caches to /mnt/localssd to avoid filling
    # root FS (16+ GB of dedupe-chunk cache was getting written when the hub
    # read local safetensors on first load).
    HF_CACHE_BASE="${HF_CACHE_BASE:-/mnt/localssd/hf_cache}"
    mkdir -p "${HF_CACHE_BASE}" "${HF_CACHE_BASE}/xet"
    echo "    HF_HOME=${HF_CACHE_BASE}"

    CUDA_DEVS=$(seq -s, 0 $((NUM_GPUS - 1)))

    TRT_ENV=()
    if [ -n "${LOAD_TRT_ENGINE}" ]; then
        TRT_ENV=("LOAD_TRT_ENGINE=${LOAD_TRT_ENGINE}")
    fi

    cd "${REPO_DIR}"
    env \
        CUDA_HOME="${CUDA_HOME}" \
        HF_HOME="${HF_CACHE_BASE}" \
        HF_HUB_CACHE="${HF_CACHE_BASE}/hub" \
        HUGGINGFACE_HUB_CACHE="${HF_CACHE_BASE}/hub" \
        HF_XET_CACHE="${HF_CACHE_BASE}/xet" \
        DYNAMIC_CACHE_SCHEDULE="${DYNAMIC_CACHE}" \
        DISABLE_TORCH_COMPILE="${DISABLE_TORCH_COMPILE:-${TORCH_COMPILE}}" \
        COMPILE_DIT="${COMPILE_DIT}" \
        CUDA_GRAPH_DIT="${CUDA_GRAPH_DIT}" \
        CUDA_GRAPH_DIT_MANUAL="${CUDA_GRAPH_DIT_MANUAL}" \
        STATIC_KV_CACHE="${STATIC_KV_CACHE}" \
        PYNCCL_ALLTOALL="${PYNCCL_ALLTOALL}" \
        NUM_DIT_STEPS="${NUM_DIT_STEPS}" \
        ATTENTION_BACKEND="${ATTENTION_BACKEND}" \
        FP8_INFERENCE="${FP8_INFERENCE}" \
        CUDA_VISIBLE_DEVICES="${CUDA_DEVS}" \
        NUM_GPUS="${NUM_GPUS}" \
        SP_SIZE="${SP_SIZE}" \
        PROFILE_INFERENCE="${PROFILE_INFERENCE:-}" \
        PROFILE_LOG_FILE="${PROFILE_LOG_FILE:-/mnt/localssd/dreamzero_profile.jsonl}" \
        PROFILE_TRACE="${PROFILE_TRACE:-}" \
        PROFILE_TRACE_FILE="${PROFILE_TRACE_FILE:-}" \
        PROFILE_TRACE_AFTER_CALL="${PROFILE_TRACE_AFTER_CALL:-5}" \
        PEAK_TFLOPS="${PEAK_TFLOPS:-989}" \
        COMPILE_BLOCKS_MODE="${COMPILE_BLOCKS_MODE:-default}" \
        COMPILE_DIT_MODE="${COMPILE_DIT_MODE:-reduce-overhead}" \
        TORCHINDUCTOR_COALESCE_TILING_ANALYSIS=0 \
        TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}" \
        GRAPH_MAX_BLOCKS="${GRAPH_MAX_BLOCKS:-0}" \
        GRAPH_NOOP="${GRAPH_NOOP:-0}" \
        GRAPH_DEBUG_TEST="${GRAPH_DEBUG_TEST:-0}" \
        DUMP_FWD_KWARGS="${DUMP_FWD_KWARGS:-0}" \
        GRAPH_INLINE_TEST="${GRAPH_INLINE_TEST:-0}" \
        GRAPH_BLOCKS_PER_CHUNK="${GRAPH_BLOCKS_PER_CHUNK:-20}" \
        DUMP_CAPTURE_KWARGS="${DUMP_CAPTURE_KWARGS:-0}" \
        TEACACHE_THRESH="${TEACACHE_THRESH:-0}" \
        FUSE_QKV="${FUSE_QKV:-false}" \
        ASYNC_VAE="${ASYNC_VAE:-false}" \
        NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-16}" \
        KV_INIT_CACHE_THRESH="${KV_INIT_CACHE_THRESH}" \
        CFG_SCALE="${CFG_SCALE:-5.0}" \
        SKIP_VAE_ON_CACHE="${SKIP_VAE_ON_CACHE:-false}" \
        OVERLAP_VAE_DIT="${OVERLAP_VAE_DIT}" \
        COMPILE_WARMUP_CHUNKS="${COMPILE_WARMUP_CHUNKS}" \
        "${TRT_ENV[@]}" \
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

# Scrub anything the server wrote to its output_dir during this run.
# The server derives output_dir from the model_path, so we mirror that logic.
scrub_server_videos() {
    local parent_dir
    parent_dir="$(dirname "${MODEL_PATH}")"
    local ckpt_name
    ckpt_name="$(basename "${MODEL_PATH}")"
    # Match every "real_world_eval_gen_*" dir created today (server uses today's date).
    local today
    today="$(date +%Y%m%d)"
    local pattern="${parent_dir}/real_world_eval_gen_${today}_*/${ckpt_name}"
    local removed=0
    shopt -s nullglob
    for d in ${pattern}; do
        if [ -d "${d}" ]; then
            local n_mp4
            n_mp4=$(find "${d}" -maxdepth 1 -name "*.mp4" | wc -l)
            rm -f "${d}"/*.mp4 2>/dev/null
            rm -rf "${d}/inputs" 2>/dev/null
            removed=$((removed + n_mp4))
        fi
    done
    shopt -u nullglob
    if (( removed > 0 )); then
        echo "==> Scrubbed ${removed} video(s) from server output dirs."
    fi
}

cmd_bench() {
    local report_dir="${REPORT_DIR:-${REPO_DIR}/bench_reports}"
    mkdir -p "${report_dir}"
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local tag="sp${SP_SIZE}_${ATTENTION_BACKEND}"
    if [ "${COMPILE_DIT}" = "true" ]; then tag="${tag}_cdit"; fi
    if [ "${FP8_INFERENCE}" = "true" ]; then tag="${tag}_fp8"; fi
    local report_path="${report_dir}/${ts}_${tag}.json"

    local num_chunks="${NUM_CHUNKS:-15}"
    local warmup_chunks="${WARMUP_CHUNKS:-3}"
    local profile_log="${PROFILE_LOG_FILE:-/mnt/localssd/dreamzero_profile.jsonl}"

    echo "==> Benchmarking against server on port ${PORT}"
    echo "    report:       ${report_path}"
    echo "    profile log:  ${profile_log}"
    echo "    num-chunks:   ${num_chunks}"
    echo "    warmup:       ${warmup_chunks} (excluded from steady-state)"

    # Truncate the profile log so only *this* bench run's records get rolled up.
    # The server must have been started with PROFILE_INFERENCE=true for records
    # to exist — if not, we'll note that in the report.
    : > "${profile_log}" 2>/dev/null || true

    cd "${REPO_DIR}"
    python test_client_AR.py \
        --port "${PORT}" \
        --num-chunks "${num_chunks}" \
        --warmup-chunks "${warmup_chunks}" \
        --no-reset \
        --report "${report_path}" \
        --profile-log "${profile_log}"
    local rc=$?

    scrub_server_videos
    return ${rc}
}

case "${1:-help}" in
    start) cmd_start ;;
    test)  cmd_test ;;
    bench) cmd_bench ;;
    *)
        echo "Usage: $0 {start|test|bench}"
        echo ""
        echo "  start — Launch the inference server (8x H200, SP=4, TE, COMPILE_DIT by default)"
        echo "  test  — Run the test client against the running server (may save videos)"
        echo "  bench — Run the test client, write a JSON perf report, skip reset, scrub"
        echo "          any videos the server wrote mid-run."
        echo ""
        echo "Env for bench:"
        echo "  REPORT_DIR      — where to write reports (default: ./bench_reports)"
        echo "  NUM_CHUNKS      — total inference calls incl. initial frame (default: 15)"
        echo "  WARMUP_CHUNKS   — N calls excluded from steady-state stats (default: 3)"
        echo ""
        echo "Env for MFU + communication profiling (set before 'start'):"
        echo "  PROFILE_INFERENCE=true  — per-call phase/comm/MFU records, one JSON line per"
        echo "                            call in PROFILE_LOG_FILE (default /mnt/localssd/"
        echo "                            dreamzero_profile.jsonl). 'bench' rolls these into"
        echo "                            the report as 'profile_calls' and 'profile_summary'."
        echo "  PROFILE_TRACE=true      — dump a chrome/perfetto trace of one post-warmup"
        echo "                            call (PROFILE_TRACE_AFTER_CALL=5 skip count)."
        echo "  PEAK_TFLOPS=989         — for MFU denominator (H100/H200 BF16 dense = 989)."
        echo ""
        echo "Tip: COMPILE_DIT=true needs warmup. Run 'bench' twice — the first run"
        echo "     compiles new shapes (~3 min), the second reaches ~0.5s steady-state."
        exit 1
        ;;
esac
