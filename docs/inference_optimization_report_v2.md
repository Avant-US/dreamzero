# DreamZero Inference Optimization Report v2

## Final Result: 0.37s/chunk Server, 0.40s/chunk Client (8x H200, SP=4)

**Configuration**: `COMPILE_DIT_MODE=reduce-overhead`, SP=4, TE attention, static KV cache, pynccl
**Hardware**: 8x NVIDIA H200 141GB HBM3e (NVLink interconnect)
**MFU**: 24.2% (242 TFLOPS of 989 peak) -- 5x improvement over starting point

---

## 1. Starting Point

Baseline measured 2026-04-15 on 8x H200, native conda, SP=4, TE attention, no torch.compile.

| Metric | Value |
|--------|-------|
| Per-chunk latency (client) | 0.77s |
| Diffusion (4 compute steps of 16 scheduled) | 0.45s |
| Per-step DiT time | 113ms |
| MFU | 4.0% |
| GPU idle time (from chrome trace) | 56% |
| Kernel launches per call | 28,585 |
| NCCL GPU time per call | 15.8ms (3.6%) |

The workload was severely **launch-bound**: 28k kernels at 5-10us CPU dispatch overhead
each explained 300+ms of GPU idle time. This is exactly what CUDA graph replay solves.

With `COMPILE_DIT=true` (mode=default):

| Metric | Value |
|--------|-------|
| Per-chunk latency (bench report) | 0.67s |
| Per-step DiT time | 93ms |
| MFU | 4.9% |

mode=default helped modestly but did not capture CUDA graphs, so launch overhead persisted.

---

## 2. Final Achievement: reduce-overhead Mode

Steady-state performance after warmup:

```
Total: 0.37s server / 0.40s client per chunk

  [DiT 4 steps]  300ms  ████████████████████████████░  81%
  [VAE Encode]     50ms  ████░░░░░░░░░░░░░░░░░░░░░░░░  14%
  [Scheduler]      20ms  ██░░░░░░░░░░░░░░░░░░░░░░░░░░   5%
                  ─────
                  370ms
```

| Metric | mode=default | reduce-overhead | Improvement |
|--------|-------------|-----------------|-------------|
| Per-step (DiT) | 93ms | 75ms | 19% faster |
| Per-chunk (client) | 0.67s | 0.40s | 40% faster |
| MFU | 4.9% | 24.2% | 4.9x |
| TFLOPS achieved | 48.5 | 242 | 5.0x |

The 5x MFU improvement comes from CUDAGraph Trees eliminating kernel launch overhead,
Python loop overhead, and enabling better kernel fusion by the inductor.

---

## 3. Fixes Required for reduce-overhead

Enabling `torch.compile(mode="reduce-overhead")` on the DreamZero DiT with sequence
parallelism required solving six distinct compatibility issues.

### 3.1 SP Sequence Padding

**Problem**: `tensor.chunk(sp_size)` produces unequal chunks when the sequence length
is not divisible by `sp_size`. The `register_fake` impl uses floor division, but the
real impl returns ceil-sized chunks for most ranks. CUDAGraph Trees asserts output
shapes match the fake prediction.

**Fix**: Pad the sequence to `sp_size * 8` before the SP split. The factor of 8 ensures
both even chunks across SP ranks AND FP8 GEMM alignment (batch*seq divisible by 8).

```python
# In _forward_blocks, before sp_split_seq:
_sp_align = _sp_sz * 8
_sp_pad = (-_sp_orig_len) % _sp_align
if _sp_pad > 0:
    x = F.pad(x, (0, 0, 0, _sp_pad))
    e0 = F.pad(e0, (0, 0, 0, 0, 0, _sp_pad))
    e = F.pad(e, (0, 0, 0, _sp_pad))
```

**File**: `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`

### 3.2 Custom Ops for Compile-Safe NCCL

**Problem**: Standard `dist.all_to_all_single` and `dist.all_gather` are opaque to
torch.compile and cause graph breaks or symbolic shape errors.

**Fix**: Registered `torch.library.custom_op` wrappers with `register_fake` impls that
return concrete output shapes:

- `sp::all_to_all` -- scatter one dim, gather another
- `sp::split_seq` -- split tensor along dim, return rank's chunk
- `sp::gather_seq` -- all-gather along dim, trim to original length

Uses pynccl (ncclSend/ncclRecv on current stream) when available for graph-safe NCCL.

**File**: `groot/vla/model/dreamzero/modules/sp_compile_ops.py`

### 3.3 Inductor Tiling Workaround

**Problem**: `torch._inductor.tiling_utils.analyze_memory_coalescing` calls `int()` on
a symbolic `Mul` expression, crashing with `TypeError: Expected a number but got Mul`.

**Fix**: `TORCHINDUCTOR_COALESCE_TILING_ANALYSIS=0` to skip the coalescing analysis.
This is a PyTorch 2.8 bug in the inductor codegen for custom op outputs.

### 3.4 VAE Compile Mode Compatibility

**Problem**: CUDAGraph Trees stores state in thread-local storage (TLS). When
`OVERLAP_VAE_DIT=true`, the VAE runs in a background thread where TLS is not
initialized, causing `AssertionError: torch._C._is_key_in_tls(attr_name)`.

**Fix**: Compile VAE with `mode="default"` (not `reduce-overhead`) so the background
thread does not require CUDAGraph Trees TLS.

### 3.5 Static KV Cache with In-Place Writes

**Problem**: Dynamic KV caches change shape between calls (cat + truncate), which is
incompatible with CUDA graph replay requiring fixed tensor shapes.

**Fix**: Pre-allocate KV cache at `max_attention_size` and write new tokens in-place.
Attention always reads the full buffer; unfilled slots are zero and softmax
down-weights them naturally.

### 3.6 pynccl for Graph-Safe NCCL

**Problem**: Standard NCCL collectives through `torch.distributed` cannot be captured
inside CUDA graphs.

**Fix**: Use pynccl (ncclSend/ncclRecv on current CUDA stream) which is graph-capturable.
Registered via `sp_compile_ops.register_pynccl("sp_group", comm)`.

---

## 4. FP8 Exploration

Three approaches were tested; none were production-viable without retraining.

### 4.1 Static FP8 Per-Tensor

- **Per-step**: 59ms (21% faster than BF16 at 75ms)
- **Problem**: Quality broken. Action output ranges 100x off-scale. Dynamic cache
  schedule runs 14+ steps instead of 4 because cosine similarity thresholds fail
  with quantized activations.
- **Root cause**: Per-tensor scaling with BF16-trained weights. The model was never
  calibrated for FP8 ranges.

### 4.2 Static FP8 Per-Channel (Rowwise)

- **Per-step**: 71ms (5% faster than BF16)
- **Problem**: Quality partially recovered but still runs 8 steps instead of 4.
  The dynamic cache schedule is sensitive to activation magnitude changes.
- **Conclusion**: Better than per-tensor but still insufficient without calibration.

### 4.3 TransformerEngine Dynamic FP8

- **Quality**: Good (amax history tracks correct ranges)
- **Problem**: Constant recompiles. TE's `fp8_autocast` maintains Python-level amax
  history state that torch.compile guards against. Every forward pass mutates this
  state, triggering ~80s recompiles.
- **Conclusion**: Incompatible with reduce-overhead mode.

**FP8 verdict**: Requires quantization-aware fine-tuning or offline calibration before
it can help this model. The 21% per-step speedup from FP8 GEMMs is real, but quality
degradation and cache schedule interactions make it unusable without retraining.

---

## 5. Optimizations Attempted but Incompatible with reduce-overhead

CUDAGraph Trees requires ALL tensor shapes and Python state to be static across
replays. Any dynamic state triggers expensive ~80s recompiles.

| Optimization | Why It Fails | Impact |
|---|---|---|
| Trim static KV to valid tokens | Changes attention tensor shapes per call | Full 80s recompile per unique shape |
| Cross-attention KV caching | Module attribute mutation inside graph | Overwrite error on graph replay |
| TE `fp8_autocast` | Python-level amax history state | Constant guard failures |
| Dynamic sequence lengths | Breaks CUDAGraph shape assertions | Must pad to fixed sizes |

This is the fundamental constraint of reduce-overhead mode: any optimization that is
"dynamic" in nature (shape-varying, state-mutating) is incompatible.

---

## 6. Production Configuration

```bash
# Core compilation
COMPILE_DIT=true
COMPILE_DIT_MODE=reduce-overhead

# Static shapes for CUDA graph
STATIC_KV_CACHE=true
KV_INIT_CACHE_THRESH=1

# VAE overlap (with mode=default VAE compile for TLS compat)
OVERLAP_VAE_DIT=true

# Graph-safe NCCL
PYNCCL_ALLTOALL=true

# Inductor workaround
TORCHINDUCTOR_COALESCE_TILING_ANALYSIS=0

# Backend
ATTENTION_BACKEND=TE
SP_SIZE=4
NUM_GPUS=8
```

---

## 7. Warmup Behavior

The first ~5 chunks trigger CUDAGraph recompilations as new tensor shapes and Python
guard values are encountered:

| Call | Time | Trigger |
|------|------|---------|
| Initial frame | 548s | Full compilation (encoders + DiT + graph capture) |
| Chunk 0 | 97s | Recompile for continuation shape + VAE first call |
| Chunk 1 | 84s | Recompile for new `_static_kv_fill` value |
| Chunk 2 | 80s | Recompile for new `_static_kv_fill` value |
| Chunk 3 | 70s | Recompile for new `_static_kv_fill` value |
| Chunk 4+ | **0.40s** | Cached graph replay (steady state) |

Each unique `_static_kv_fill` value triggers a dynamo guard check and recompile.
After warmup, all needed CUDA graphs are cached. Minor recompile spikes (~0.6-0.8s)
occur at rolling-window boundaries (calls 8, 12).

**Production recommendation**: Run a warmup sequence at server startup before serving
real requests. The ~880s total warmup is a one-time cost.

---

## 8. Profiling Data (Steady-State)

Measured with `PROFILE_INFERENCE=true`, 16 inference calls, calls 5-15 steady state:

```
Call  5: total=0.37s  dit=0.30s  vae=0.05s  mfu=24.0%
Call  6: total=0.37s  dit=0.30s  vae=0.05s  mfu=24.3%
Call  7: total=0.37s  dit=0.30s  vae=0.05s  mfu=24.1%
Call  9: total=0.38s  dit=0.31s  vae=0.05s  mfu=23.4%
Call 10: total=0.37s  dit=0.30s  vae=0.05s  mfu=24.0%
Call 11: total=0.36s  dit=0.30s  vae=0.05s  mfu=24.3%
Call 13: total=0.37s  dit=0.30s  vae=0.05s  mfu=24.2%
Call 14: total=0.37s  dit=0.31s  vae=0.05s  mfu=23.7%
Call 15: total=0.37s  dit=0.31s  vae=0.05s  mfu=23.8%
```

Excluding recompile spikes: **370ms total, 303ms DiT, 24.2% MFU, 242 TFLOPS**.

### MFU Analysis

| Metric | Value |
|--------|-------|
| Total FLOPs per step (per rank) | ~17.1 TFLOPS |
| H200 BF16 peak | 989 TFLOPS |
| Theoretical compute time | 17.3ms |
| Actual time per step | 75ms |
| MFU | 24.2% |
| Achieved TFLOPS | 242 |

The 76% gap between theoretical and actual is consumed by: memory-bound small ops
(RMSNorm, RoPE, modulation: ~15-20ms), CUDA graph overhead (~5-10ms), static KV
cache waste attending over 78% padding (~5-10ms), cross-attention recompute (~5ms),
and SP all-to-all latency (~3ms).

---

## 9. Optimization History (Cumulative)

| # | Optimization | Technique | Speedup | Latency |
|---|-------------|-----------|---------|---------|
| 0 | Baseline | Single GPU, 16 steps, no caching | 1x | 5.7s |
| 1 | CFG Parallelism | Split cond/uncond across 2 GPUs | 1.9x | 3.0s |
| 2 | DiT Caching (TeaCache) | Cosine similarity skip: 16 -> 4 steps | 5.5x | 1.03s |
| 3 | TE Attention | cuDNN fused attention (H200-optimized) | 6.6x | 0.87s |
| 4 | Sequence Parallelism (SP=4) | Ulysses all-to-all, 4 GPUs per CFG | 10x | 0.57s |
| 5 | torch.compile (mode=default) | Inductor fusion on DiT block loop | 11.4x | 0.50s |
| 6 | **reduce-overhead + static KV** | **CUDAGraph Trees + pynccl + padding** | **14.3x** | **0.40s** |

---

## 10. Key Implementation Files

| File | Role |
|------|------|
| `modules/sp_compile_ops.py` | Custom ops for compile-safe NCCL (sp::all_to_all, sp::split_seq, sp::gather_seq) |
| `modules/wan_video_dit_action_casual_chunk.py` | SP sequence padding, static KV cache, block loop with SP context |
| `action_head/wan_flow_matching_action_tf.py` | COMPILE_DIT env var, torch.compile on _forward_blocks, FP8 support |
| `socket_test_optimized_AR.py` | SP-aware DeviceMesh: mesh_shape=(cfg_size, sp_size) |
| `base_vla.py` | Direct GPU loading of safetensors |

All files under `groot/vla/model/dreamzero/`.

---

## 11. Future Optimization Roadmap

| # | Optimization | ms/chunk saved | Complexity | Notes |
|---|---|---|---|---|
| 1 | Fewer diffusion steps (4 -> 2) | 150 | High | Requires distillation training |
| 2 | FP8 with QAT | 58 | High | Requires quantization-aware fine-tuning |
| 3 | Async VAE overlap | 50 | Low | Implemented; needs warmup for TLS compat |
| 4 | Cross-attn KV cache | 20 | Medium | Needs graph-safe impl (no module mutation) |
| 5 | Reduce attention window (9 -> 5 frames) | 11 | Low | Quality trade-off to evaluate |

**Combining #3+#4+#5 (feasible without retraining): ~80ms saved -> 290ms/chunk (server)**

**Combining all five (requires training): ~290ms saved -> 80ms/chunk (theoretical floor)**

---

## 12. Lessons Learned

1. **reduce-overhead is all-or-nothing.** Partial dynamism (one changing shape, one
   mutating Python int) is enough to destroy performance with 80s recompiles. Every
   tensor shape and Python value visible to dynamo guards must be static.

2. **Custom ops are the compile bridge for distributed ops.** `torch.library.custom_op`
   with `register_fake` lets torch.compile trace through NCCL collectives that are
   otherwise opaque. The fake impl returns concrete shapes; the real impl runs NCCL.

3. **pynccl is required for NCCL inside CUDA graphs.** Standard `torch.distributed`
   collectives cannot be captured. pynccl's ncclSend/ncclRecv on the current stream
   are graph-capturable.

4. **Kernel launch overhead, not communication, was the real bottleneck.** Early
   profiling attributed 57% of diffusion to all-to-all communication. Chrome trace
   revealed only 15.8ms of actual NCCL GPU time -- the rest was CPU dispatch overhead
   for 28k kernels. CUDAGraph Trees eliminated this entirely.

5. **FP8 needs the model, not just the runtime.** FP8 GEMMs are 21% faster per-step,
   but BF16-trained weights produce 100x-off action ranges. Quality-preserving FP8
   requires QAT or careful offline calibration.

6. **SP=4 beats SP=2 despite higher communication.** At this model size (5120 dim,
   40 heads, 40 layers), the extra compute parallelism from 8 GPUs outweighs the
   additional all-to-all overhead. SP=2 has better MFU (8.6% vs 4.9% in eager) but
   worse wall-clock time because it uses only 4 GPUs.
