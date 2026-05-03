# DreamZero Reduce-Overhead Inference: Performance Analysis & Optimization Roadmap

## Current Performance

| Config | Per-Chunk (server) | Per-Step | Steps/Chunk | DiT Time |
|--------|-------------------|----------|-------------|----------|
| mode=default, COMPILE_DIT (bench) | 0.67s | 93ms | 4.3 | 0.40s |
| **mode=reduce-overhead + static KV** | **0.37s** | **75ms** | **4** | **0.30s** |

Per-step improvement: **93ms -> 75ms (20% faster, 1.24x speedup)**. Total improvement
includes skip_kv optimization (saves ~97ms KV creation) and skip_vae from caching.

## Phase Breakdown (Steady-State, 0.37s Server-Side)

```
[VAE Encode]    50ms  ████░░░░░░░░░░░░░░░░░░░░░░░░  13.5%
[DiT 4 steps]  300ms  █████████████████████████████░  81.1%
[Scheduler]     20ms  ██░░░░░░░░░░░░░░░░░░░░░░░░░░░   5.4%
                ─────
                370ms
```

## DiT Step Breakdown (75ms/step, Theoretical Analysis)

### Model Parameters
- 40 transformer blocks, dim=5120, heads=40 (10/rank), ffn=13824
- SP=4: S_local=447, S_full=1788 (after all-to-all)
- Static KV cache: max 7920 tokens (78% padding when actual ~1788)
- BF16 precision

### FLOPs per Step (per rank)

| Component | GFLOPs/block | 40 blocks | % of total |
|-----------|-------------|-----------|------------|
| QKV projection (S=447, D=5120) | 70.3 | 2,812 | 16% |
| Self-attention (Q:1788 × KV:7920) | 73.0 | 2,920 | 17% |
| Output projection | 23.4 | 936 | 5% |
| Cross-attn projections | 127.5 | 5,100 | 30% |
| Cross-attn dot products | 7.0 | 280 | 2% |
| FFN (5120→13824→5120) | 126.6 | 5,064 | 30% |
| **Total** | **~428** | **~17,100** | |

### Theoretical vs Actual

| Metric | Value |
|--------|-------|
| Total FLOPs per step | ~17.1 TFLOPS |
| H200 BF16 peak | 989 TFLOPS |
| Theoretical compute time | 17.3ms |
| Actual time | 75ms |
| **MFU (Model FLOP Utilization)** | **23.1%** |
| Communication volume (all-to-all) | ~0.56 GB |
| Theoretical comm time (NVLink) | ~1.9ms |

**23% MFU means 77% of time is overhead, not useful compute.**

## Where the 57ms Gap Goes

```
75ms actual = 17ms compute + 58ms overhead

Overhead breakdown (estimated):
  [Memory-bound ops]     15-20ms  ████████████░░░░░░░░  27%
  [CUDA graph overhead]   5-10ms  ██████░░░░░░░░░░░░░░  10%
  [KV cache waste]        5-10ms  ██████░░░░░░░░░░░░░░  10%
  [Cross-attn recompute]    ~5ms  ████░░░░░░░░░░░░░░░░   7%
  [SP all-to-all latency]  ~3ms  ███░░░░░░░░░░░░░░░░░   4%
  [Recompile overhead]     ~5ms  ████░░░░░░░░░░░░░░░░   7%
  [Other (RoPE, norms)]    ~5ms  ████░░░░░░░░░░░░░░░░   7%
```

### 1. Memory-Bound Small Operations (15-20ms)

RMSNorm, LayerNorm, RoPE, modulation (scale/shift), residual adds — all these are
**memory-bandwidth-limited** at small token counts (447 tokens × 5120 dim = 2.3M elements).
For a 2.3M-element operation, compute is negligible; time is dominated by reading/writing
to HBM at 4.8 TB/s: ~1us per op, but with 40 blocks × ~10 ops/block = 400 ops = 0.4ms.

However, torch.compile may not fully fuse all these small ops, especially with custom_ops
acting as fusion barriers. Each unfused kernel has ~5us dispatch overhead.

### 2. CUDA Graph Overhead (5-10ms)

Even with CUDAGraph Trees, each step replays the graph which includes:
- Graph tree traversal to find the right cached graph
- Static tensor copy for changed inputs (x, timestep, etc.)
- The `_static_kv_fill` guard check and potential recompile
- Python-side bookkeeping between steps

### 3. Static KV Cache Waste (5-10ms)

Attention over the full 7920-token buffer when only ~1788 tokens are valid.
That's 78% wasted compute in self-attention:
- Actual QK^T: 1788 × 1788 × 128 × 10 × 2 = 8.3 GFLOPs
- Computed QK^T: 1788 × 7920 × 128 × 10 × 2 = 36.7 GFLOPs
- Waste: 28.4 GFLOPs per block × 40 blocks = 1.14 TFLOPS wasted

### 4. Cross-Attention Recompute (~5ms)

`crossattn_cache` is never passed to blocks — text/CLIP K,V projections are recomputed
every block (512 text tokens × 5120 × 5120 × 2 = 26.8 GFLOP/block × 40 = 1.07 TFLOPS).

### 5. SP All-to-All Latency (~3ms)

4 all-to-all calls per block × 40 blocks = 160 calls. Each moves ~3.4 MB. At these
small sizes, latency (~5us per call) dominates over bandwidth. Total: ~0.8ms compute +
~2ms latency overhead.

## Optimization Roadmap (Ranked by Impact)

### Tier 1: High Impact (target: 40ms/step → 0.20s/chunk)

#### 1a. FP8 Inference (est. -30ms → 45ms/step)
H200 FP8 peak is 1979 TFLOPS (2x BF16). Linear layers are 81% of FLOPs. FP8 matmuls
would cut compute from 17ms to ~9ms, and reduce memory bandwidth pressure.
- Already have `FP8_INFERENCE` support in codebase
- Requires FP8 calibration or dynamic scaling
- Compatible with reduce-overhead mode

#### 1b. Attention Mask for Static KV (est. -5ms → 70ms/step)
Instead of attending to the full 7920-token buffer, pass a proper attention mask that
excludes unfilled positions. This eliminates 78% of self-attention compute waste.
- `TE` attention supports `attn_mask` parameter
- Or use `max_seqlen_k` parameter to limit KV length
- Zero-cost if using FlashAttention's variable-length API

#### 1c. Cache Cross-Attention KV (est. -5ms → 70ms/step)
Pass `crossattn_cache` through the block loop. Text/CLIP embeddings are constant across
diffusion steps — caching their K,V projections saves 1.07 TFLOPS/step.

### Tier 2: Medium Impact (target: 30ms/step → 0.15s/chunk)

#### 2a. Fuse Norm+Modulation+Linear Kernels (est. -5ms)
The pre-attention pattern `norm(x) * (1+scale) + shift` followed by a linear projection
is 3 separate memory-bound kernels. A custom fused kernel or better torch.compile fusion
would reduce memory traffic. TransformerEngine has `LayerNormLinear` which does this.

#### 2b. Reduce SP Overhead (est. -3ms)
- Use `torch.distributed.all_to_all` with pre-allocated output buffers
- Overlap all-to-all communication with compute (pipelined)
- Consider SP=2 instead of SP=4 (fewer comm calls, larger per-rank compute)

#### 2c. Convert _static_kv_fill to Tensor (est. -5ms warmup reduction)
Replace the Python-int KV fill tracker with a scalar tensor. Eliminates guard failures
and recompiles during the warmup phase (reduces warmup from 880s to ~600s).

### Tier 3: Speculative (target: 20ms/step → 0.10s/chunk)

#### 3a. Fewer Diffusion Steps
Dynamic cache schedule already uses 4 steps for continuations. Exploring distillation
or consistency models could reduce to 1-2 steps.

#### 3b. Block Fusion / Megakernel
Compile multiple transformer blocks into a single CUDA graph without intermediate
memory writes. This is what `GRAPH_BLOCKS_PER_CHUNK` experiments with.

#### 3c. TensorRT Compilation
The `LOAD_TRT_ENGINE` support already exists. TRT can produce more optimized kernels
than torch.compile, especially for the attention + FFN patterns.

## Optimization Roadmap (Ranked by Chunk Impact)

| # | Optimization | ms/step | ms/chunk | Complexity | reduce-overhead? | Risk |
|---|---|---|---|---|---|---|
| 1 | Fewer diffusion steps (4→2) | 75 | **150** | High (distillation) | Yes | Med |
| 2 | FP8 linear layers | 14.5 | **58** | Medium | Yes | Med |
| 3 | Async VAE encode | -- | **50** | Medium (fix TLS) | Yes | Low |
| 4 | Cross-attn KV cache | 6.6 | **20** | Low | Yes | Low |
| 5 | Reduce attn window (9→5) | 2.7 | **11** | Low (env var) | Yes | Low |
| 6 | SP=2 vs SP=4 | ~0-3 | **0-12** | Low (config) | Yes | Med |
| 7 | Comm-compute overlap | 0.5 | **2** | High | Partial | High |
| 8 | Kernel fusion (norm+linear) | 0.35 | **1.4** | Medium | Yes | Low |

**Combining #2+#3+#4+#5 = 139ms saved (370ms → ~231ms, 37% faster)**

### Key constraint: reduce-overhead shape stability

Any optimization that CHANGES tensor shapes (e.g., trimming KV cache to valid tokens)
triggers full ~80s recompiles and is NOT viable. Only shape-preserving optimizations work:
- FP8: changes dtype but not shape ✓
- Cross-attn cache: skips compute but output shapes identical ✓  
- Reduce attn window: smaller fixed buffer, constant across calls ✓
- Fewer steps: fewer calls to same compiled function ✓

## Profiling Data (Actual Measurements)

Collected with `PROFILE_INFERENCE=true` on 8x H200, SP=4, reduce-overhead mode.
16 inference calls total.

### Per-Call Timing

```
Call  0: total= 544.47s  dit=391.28s  vae=0.00s  mfu= 0.0%  (initial frame compilation)
Call  1: total=  96.89s  dit= 81.70s  vae=14.98s mfu= 0.1%  (continuation recompile)
Call  2: total=  80.47s  dit= 80.40s  vae=0.05s  mfu= 0.1%  (kv_fill recompile)
Call  3: total=  79.68s  dit= 79.61s  vae=0.05s  mfu= 0.1%  (kv_fill recompile)
Call  4: total=  69.99s  dit=  0.32s  vae=0.00s  mfu=22.8%  (last recompile + cache hit)
Call  5: total=   0.37s  dit=  0.30s  vae=0.05s  mfu=24.0%  ← STEADY STATE
Call  6: total=   0.37s  dit=  0.30s  vae=0.05s  mfu=24.3%
Call  7: total=   0.37s  dit=  0.30s  vae=0.05s  mfu=24.1%
Call  8: total=   0.63s  dit=  0.30s  vae=0.00s  mfu=24.6%  (minor kv_fill recompile)
Call  9: total=   0.38s  dit=  0.31s  vae=0.05s  mfu=23.4%
Call 10: total=   0.37s  dit=  0.30s  vae=0.05s  mfu=24.0%
Call 11: total=   0.36s  dit=  0.30s  vae=0.05s  mfu=24.3%
Call 12: total=   0.78s  dit=  0.44s  vae=0.00s  mfu=28.9%  (minor kv_fill recompile)
Call 13: total=   0.37s  dit=  0.30s  vae=0.05s  mfu=24.2%
Call 14: total=   0.37s  dit=  0.31s  vae=0.05s  mfu=23.7%
Call 15: total=   0.37s  dit=  0.31s  vae=0.05s  mfu=23.8%
```

### Steady-State Summary (calls 5-15)

| Metric | Value |
|--------|-------|
| Mean total | 431ms* |
| Mean DiT (4 steps) | 315ms* |
| Mean VAE | 41ms |
| Mean scheduler | 18ms |
| **MFU** | **24.5%** |
| **TFLOPS** | **242** |

*Includes recompile spikes at calls 8,12. Excluding spikes: 370ms total, 303ms DiT.

### Comparison with mode=default (from bench reports)

| Metric | mode=default | reduce-overhead | Change |
|--------|-------------|-----------------|--------|
| Per-step (DiT) | 93ms | 75ms | **-19%** |
| MFU | 4.9% | 24.2% | **+4.9x** |
| TFLOPS | 48.5 | 242 | **+5.0x** |
| Comm % | 9.7% | (inside graph) | — |

The 5x MFU improvement comes from CUDA graph replay eliminating kernel launch overhead,
Python loop overhead, and enabling better kernel fusion by the inductor.

### Profile JSONL

```bash
python -m profiling.analyze reduce_overhead_profile.jsonl --format markdown
```
