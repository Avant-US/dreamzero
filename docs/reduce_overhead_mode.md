# torch.compile reduce-overhead Mode for DreamZero Inference

## Summary

Successfully enabled `torch.compile(mode="reduce-overhead")` for the DreamZero DiT
on 8x H200 with SP=4. This mode uses CUDAGraph Trees to capture and replay CUDA
graphs, eliminating kernel launch overhead and enabling better GPU utilization.

**Steady-state result: 0.37s server-side / 0.40s client-side per chunk** (4 DiT steps
with dynamic cache schedule), compared to 0.43s with `mode="default"`.

## Configuration

```bash
COMPILE_DIT=true
COMPILE_DIT_MODE=reduce-overhead    # CUDAGraph Trees
STATIC_KV_CACHE=true                # Pre-allocated fixed-size KV buffers
KV_INIT_CACHE_THRESH=1              # Skip KV re-init on continuations
OVERLAP_VAE_DIT=false               # Must be false (TLS incompatibility)
PYNCCL_ALLTOALL=true                # Graph-safe NCCL via pynccl
TORCHINDUCTOR_COALESCE_TILING_ANALYSIS=0  # Workaround for inductor symbolic shape bug
ATTENTION_BACKEND=TE
SP_SIZE=4
NUM_GPUS=8
```

## Key Fixes Required

### 1. SP Sequence Padding (sp_split_seq shape mismatch)

**Problem:** `sp_split_seq` uses `tensor.chunk(sp_size)` which produces unequal chunks
when the sequence length isn't divisible by `sp_size`. The `register_fake` impl uses
floor division, but real impl returns ceil-sized chunks for most ranks. CUDAGraph Trees
asserts the output shape matches the fake prediction.

**Fix:** Pad the sequence to the nearest multiple of `sp_size` before the SP split in
`_forward_blocks`. This ensures all ranks get equal chunks and the reverse all-to-all
(scatter sequence) has a divisible scatter dimension.

```python
# In _forward_blocks, before sp_split_seq:
_sp_pad = (-_sp_orig_len) % _sp_sz
if _sp_pad > 0:
    x = F.pad(x, (0, 0, 0, _sp_pad))
    e0 = F.pad(e0, (0, 0, 0, 0, 0, _sp_pad))
    e = F.pad(e, (0, 0, 0, _sp_pad))
```

File: `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`

### 2. Custom Ops for Compile-Safe NCCL (sp_compile_ops.py)

**Problem:** Standard `dist.all_to_all_single` and `dist.all_gather` are opaque to
torch.compile and cause graph breaks or symbolic shape errors.

**Fix:** Registered `torch.library.custom_op` wrappers with `register_fake` impls that
return concrete output shapes:
- `sp::all_to_all` — scatter one dim, gather another
- `sp::split_seq` — split tensor along dim, return rank's chunk
- `sp::gather_seq` — all-gather along dim, trim to original length

Uses pynccl (ncclSend/ncclRecv on current stream) when available for graph-safe NCCL.

File: `groot/vla/model/dreamzero/modules/sp_compile_ops.py`

### 3. Inductor Tiling Analysis Workaround

**Problem:** `torch._inductor.tiling_utils.analyze_memory_coalescing` calls `int()` on
a symbolic `Mul` expression, crashing with `TypeError: Expected a number but got Mul`.

**Fix:** Set `TORCHINDUCTOR_COALESCE_TILING_ANALYSIS=0` to skip the coalescing analysis
pass. This is a PyTorch bug in the inductor codegen for custom op outputs.

### 4. VAE Thread-Local Storage Incompatibility

**Problem:** CUDAGraph Trees stores state in thread-local storage (TLS). When
`OVERLAP_VAE_DIT=true`, the VAE runs in a background thread where TLS isn't initialized,
causing `AssertionError: torch._C._is_key_in_tls(attr_name)`.

**Fix:** Either:
- Set `OVERLAP_VAE_DIT=false` (simplest, used here)
- Compile VAE with `mode="default"` instead of `mode="reduce-overhead"`

### 5. Static KV Cache for Fixed Shapes

**Problem:** Dynamic KV caches change shape between calls (cat + truncate), which is
incompatible with CUDA graph replay requiring fixed tensor shapes.

**Fix:** Pre-allocate KV cache at `max_attention_size` and write new tokens in-place.
Attention always reads the full buffer (unfilled slots are zero, softmax down-weights
them).

## Performance Results

### Steady-State (after warmup)

| Metric | Value |
|--------|-------|
| Total server time | 0.37s |
| DiT diffusion (4 steps) | 0.30s |
| Per DiT step | 75ms |
| VAE encode | 0.05s |
| Scheduler | 0.02s |
| Client round-trip | 0.40s |

### Comparison

| Mode | Per-Chunk (client) | Per-Step |
|------|-------------------|----------|
| mode=default, COMPILE_DIT | 0.43s | 105ms |
| **mode=reduce-overhead** | **0.40s** | **75ms** |
| Improvement | **-7%** | **-29%** |

### Warmup Cost

The first few calls trigger recompilations as CUDAGraph Trees encounters new tensor
shapes (different sequence lengths, different `_static_kv_fill` values):

| Call | Time | Reason |
|------|------|--------|
| Initial frame | 548s | Full compilation (encoders + DiT + graph capture) |
| Chunk 0 | 97s | Recompile for continuation shape + VAE first call |
| Chunk 1 | 84s | Recompile for new KV fill level |
| Chunk 2 | 80s | Recompile for new KV fill level |
| Chunk 3 | 70s | Recompile for new KV fill level |
| Chunk 4+ | **0.40s** | Cached graph replay |

After ~5 warmup chunks, all needed CUDA graphs are cached and steady-state is reached.
Minor recompile spikes occur at chunks 7 (0.66s) and 11 (0.82s) when `_static_kv_fill`
hits the rolling window threshold.

## Architecture

```
Client Request
    |
    v
[VAE Encode: 0.05s] -- mode=default compile (thread-safe)
    |
    v
[DiT _forward_blocks: 0.30s / 4 steps] -- mode=reduce-overhead (CUDAGraph Trees)
    |   |
    |   +-- sp_split_seq (padded to multiple of SP_SIZE)
    |   +-- 40 transformer blocks with static KV cache
    |   +-- sp_all_to_all (pynccl graph-safe NCCL)
    |   +-- sp_gather_seq (trim to original length)
    |
    v
[Scheduler: 0.02s]
    |
    v
Action Output
```

## Known Limitations

1. **Warmup cost:** ~880s total for first 5 calls. In production, run a warmup sequence
   at server startup.

2. **`_static_kv_fill` recompiles:** The Python-int KV fill tracker triggers guard
   failures when its value changes. Converting to a tensor-based index would eliminate
   these recompiles.

3. **Docker required:** The `TORCHINDUCTOR_COALESCE_TILING_ANALYSIS=0` workaround and
   specific PyTorch version (2.8+) are needed. The Docker image `dreamzero:latest`
   includes these.

4. **OVERLAP_VAE_DIT=false:** Background VAE overlap is disabled, adding ~0.05s per
   chunk vs the overlapped path.

## Future Optimization Opportunities

1. **Tensor-based KV fill index:** Replace `self._static_kv_fill` (Python int) with a
   scalar tensor to eliminate per-call recompiles during warmup.

2. **Warmup at startup:** Pre-run a warmup sequence with representative shapes to
   pre-populate all CUDA graphs before serving real requests.

3. **FP8 inference:** Combine reduce-overhead with FP8 quantization for further speedup.

4. **Reduce DiT steps:** Current 4 steps at 75ms/step. Fewer steps or distillation
   could reduce further.

5. **Overlap VAE with DiT:** Fix TLS issue by initializing CUDAGraph Trees context in
   the background thread, or use a separate compile mode for VAE.
