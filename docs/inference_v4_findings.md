# DreamZero Inference Optimization v4 — Findings

## Production Config (0.33s/chunk, 8x H200, SP=4)

```bash
./scripts/inference/run_fast_h200.sh start
```

- `COMPILE_DIT=true` (reduce-overhead CUDA graphs)
- `STATIC_KV_CACHE=false` (dynamic KV via torch.cat)
- `OVERLAP_VAE_DIT=true`
- `ATTENTION_BACKEND=TE` (FA2 fallback on conda)
- `DISABLE_TORCH_COMPILE=false` (encoder compile)
- `DYNAMIC_CACHE_SCHEDULE=true` (TeaCache: 4/16 steps)
- `KV_INIT_CACHE_THRESH=0` (always run KV init)

## Attention Kernel Decision

### What works

| Approach | Speed | Works with torch.compile reduce-overhead? |
|---|---|---|
| FA2 + dynamic KV | **0.33s** | Yes |
| FA2 + static KV (buffer max_seqlen_k) | 0.36s | Yes (but wasted FA tiles) |
| FA3 (Hopper) + dynamic KV | 0.33s | Yes |

### FlashInfer status

| Claim | Verdict |
|---|---|
| FlashInfer has prefill kernels | True |
| FlashInfer supports manual CUDA graph usage | Often true, with restrictions |
| FlashInfer.plan() works inside torch.compile | **False** (pin_memory) |
| FlashInfer.run() always works inside torch.compile(reduce-overhead) | **False / not safe to assume** |
| Full _forward_blocks + FlashInfer + reduce-overhead works | **No** (empty graph warnings, eager fallback) |
| FA2 + torch.compile(reduce-overhead) works better | **Yes** (0.33s) |

### When to use FlashInfer

1. Isolated outside torch.compile (plan+run both in Python)
2. Manual CUDA graph capture (not via cudagraph_trees)
3. After upgrading FlashInfer (v0.6.8rc1+ may fix batch prefill + compile)
4. Inside a serving runtime (vLLM/SGLang) that manages graphs natively

### Why not FlashInfer for now

- `BatchPrefillWithRaggedKVCacheWrapper.begin_forward()` uses `pin_memory`
  which inductor cannot lower
- `run()` inside torch.compile(reduce-overhead) produces "empty CUDA graph"
  warnings — cudagraph_trees fails to capture the kernel, falls back to eager
- Manual CUDA graph capture of `run()` works (verified), but
  cudagraph_trees has a different capture protocol
- `use_cuda_graph=True` on the wrapper + pre-allocated indptr buffers
  did not resolve the empty graph issue

## Static KV Cache (vLLM-style)

### Architecture

Pre-allocated buffer at `max_attention_size + action_register_length`.
Sequential in-place writes. Action tokens written at `fill_level` position.
FA kernel masks via `cu_seqlens_k`. No torch.cat, no .clone(), no .item()
inside compiled function.

### Why it's not faster (currently)

1. **max_seqlen_k overhead**: FA2 allocates tiles based on `max_seqlen_k`.
   With buffer_size as upper bound, FA processes extra empty tiles (0.03s waste).
2. **max_seqlen_k hint recompilation**: Computing correct max_seqlen_k
   outside compile and passing as attribute causes dynamo to specialize
   on the int value → recompiles when it changes.
3. **K buffer sentinel (-1e4)**: Doesn't work because `is_causal=True`
   mask interacts differently with full buffer positions.

### What IS correct

- Buffer writes produce **bit-exact** results vs dynamic cache (verified)
- Action tokens at `fill_level` position are correctly included in `k_lens`
- The bug was: action tokens concatenated AFTER full buffer were beyond
  `k_lens` range → FA missed them (0.19 error). Fixed by writing into buffer.

## Profiling Breakdown (267ms kernel time per chunk)

| Category | Time | % |
|---|---|---|
| cuBLAS GEMM (linear/FFN) | 110.8ms | 41% |
| Flash Attention | 45.9ms | 17% |
| Triton (fused ops) | 37.2ms | 14% |
| VAE Convolution | 27.4ms | 10% |
| NCCL (SP all-to-all) | 24.5ms | 9% |
| Layout Transform | 7.3ms | 3% |
| Elementwise | 6.8ms | 3% |
| torch.cat (KV cache) | 5.9ms | 2% |

## Next steps for further optimization

1. **FP8 matmuls** — 2x throughput on H200, ~40ms savings (net of quant overhead)
2. **Fused QKV all-to-all** — 3→1 NCCL collective, ~15ms savings
3. **Fused QKV projection** — 3→1 GEMM per block, ~8ms savings
4. **Manual CUDA graph** for FlashInfer (bypass cudagraph_trees)
