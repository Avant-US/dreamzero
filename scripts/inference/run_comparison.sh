#!/usr/bin/env bash
# Run multiple inference configurations and compare results.
# Each config: start server, run bench, kill server, report.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

REPORT_DIR="/mnt/localssd/dreamzero/bench_reports/comparison_$(date +%Y%m%d_%H%M)"
mkdir -p "$REPORT_DIR"
PROFILE_LOG="/mnt/localssd/dreamzero_profile.jsonl"
SERVER_LOG="/mnt/localssd/dreamzero_logs/sp_h200.log"

run_config() {
    local name="$1"
    shift
    local envs=("$@")

    echo ""
    echo "============================================"
    echo "  Config: $name"
    echo "============================================"

    # Kill any existing server
    pkill -9 -f "torchrun.*socket_test" 2>/dev/null || true
    pkill -9 -f "socket_test_optimized_AR" 2>/dev/null || true
    sleep 3

    # Clean logs
    rm -f "$SERVER_LOG" "$PROFILE_LOG"

    # Start server with given env vars
    echo "Starting server..."
    env "${envs[@]}" \
        PROFILE_INFERENCE=true \
        PROFILE_LOG_FILE="$PROFILE_LOG" \
        nohup ./scripts/inference/run_sp_h200.sh start > "$SERVER_LOG" 2>&1 &
    disown

    # Wait for server to be ready (up to 5 min)
    local waited=0
    while ! ss -tlnp 2>/dev/null | grep -q ":5000 "; do
        sleep 5
        waited=$((waited + 5))
        if [ $waited -ge 300 ]; then
            echo "  TIMEOUT waiting for server!"
            return 1
        fi
    done
    echo "  Server ready (${waited}s)"

    # Run bench
    echo "  Running bench (15 chunks, warmup=3)..."
    local report_path="$REPORT_DIR/${name}.json"
    timeout 600 python test_client_AR.py \
        --port 5000 --num-chunks 15 --warmup-chunks 3 \
        --no-reset --report "$report_path" \
        --profile-log "$PROFILE_LOG" 2>&1 | grep -E "steady|MFU|Phases|Comm|Done"

    echo "  Report: $report_path"
}

echo "Starting comparison benchmark suite"
echo "Reports dir: $REPORT_DIR"

# Config 1: Eager baseline (SP=4, TE, no compile, no graph)
run_config "eager_sp4_te" \
    NUM_GPUS=8 SP_SIZE=4 \
    COMPILE_DIT=false CUDA_GRAPH_DIT_MANUAL=false STATIC_KV_CACHE=false

# Config 2: CUDA graph + static KV (SP=4)
run_config "cudagraph_sp4_static_kv" \
    NUM_GPUS=8 SP_SIZE=4 \
    COMPILE_DIT=false CUDA_GRAPH_DIT_MANUAL=true STATIC_KV_CACHE=true

# Config 3: CUDA graph + grow KV (SP=4)
run_config "cudagraph_sp4_grow_kv" \
    NUM_GPUS=8 SP_SIZE=4 \
    COMPILE_DIT=false CUDA_GRAPH_DIT_MANUAL=true STATIC_KV_CACHE=false

# Config 4: Eager baseline SP=1 (2 GPU)
run_config "eager_sp1" \
    NUM_GPUS=2 SP_SIZE=1 \
    COMPILE_DIT=false CUDA_GRAPH_DIT_MANUAL=false STATIC_KV_CACHE=false

# Config 5: CUDA graph + static KV (SP=1)
run_config "cudagraph_sp1_static_kv" \
    NUM_GPUS=2 SP_SIZE=1 \
    COMPILE_DIT=false CUDA_GRAPH_DIT_MANUAL=true STATIC_KV_CACHE=true

# Cleanup
pkill -9 -f "torchrun.*socket_test" 2>/dev/null || true

echo ""
echo "============================================"
echo "  COMPARISON SUMMARY"
echo "============================================"
for f in "$REPORT_DIR"/*.json; do
    name=$(basename "$f" .json)
    python3 -c "
import json
with open('$f') as fp:
    r = json.load(fp)
ss = r.get('steady_state', {})
ps = r.get('profile_summary', {})
mfu = ps.get('mfu_mean', {})
print(f'  {\"$name\":<30s}  mean={ss.get(\"mean_s\",0):.3f}s  med={ss.get(\"median_s\",0):.3f}s  p95={ss.get(\"p95_s\",0):.3f}s  MFU={mfu.get(\"mfu_pct\",0):.1f}%')
" 2>/dev/null || echo "  $name: parse error"
done

echo ""
echo "Full reports in: $REPORT_DIR"
