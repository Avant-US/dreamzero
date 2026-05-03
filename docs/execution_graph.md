# Phase-Level Execution Graph Per Inference Call

> Measured on 8× H200, SP=4, TE, COMPILE_DIT=false (baseline), 4-frame chunk.
> Times in **steady state** (from `run_sp_h200.sh bench`, calls 4+).

## 1. The DAG

```
                                          prompt
                                            │
                                            ▼
                                       text_encoder       ← 10 ms (cached after
                                            │                       1st call)
                                            │
                                            ▼
                                       text_embeds ─────────────┐
                                                                │
  exterior_cam_0 ┐                                              │
  exterior_cam_1 ├──►   observation (images)                   │
      wrist_cam  ┘              │                              │
                                │                              │
                          ┌─────┴─────┐                        │
                          ▼           ▼                        │
                    image_encoder  vae_encoder                 │
                      (CLIP)      (Wan VAE)                    │
                     ~0 ms warm     ~46 ms                     │
                     ~113 ms cold     │                        │
                          │           │                        │
                          ▼           ▼                        │
                    clip_features  latents ─┐                  │
                          │                 │                  │
                          └────────┬────────┤                  │
                                   │        │                  │
  state  ─────────────────────────►│        │                  │
                                   ▼        │                  │
                              KV pre-pass ◄─┼──────────────────┤
                           (one DiT fwd     │                  │
                            at t=0)         │                  │
                            ~80 ms          │                  │
                                   │        │                  │
                                   ▼        │                  │
                           KV cache  +──────┘                  │
                           populated                           │
                                   │                           │
                                   ▼                           │
                           main diffusion loop ◄───────────────┘
                           (16 scheduled, ~4 compute           all 3 conditioning
                           after dynamic cache skip)            tensors feed here
                           40 layers × 4 steps
                                ~400 ms
                                   │
                                   ▼
                           ┌───────┴───────┐
                           ▼               ▼
                       video_pred       action_pred
                      (usually            │
                      discarded)          ▼
                                      action_head
                                         ~2 ms
                                         │
                                         ▼
                                      action vector
                                       (24, 8)
```

## 2. Steady-state timing (4-frame chunk, post warmup)

| Phase | Measured (s) | % of total | Notes |
|---|---:|---:|---|
| text_encoder | 0.010 | 2% | Re-used after first call |
| image_encoder | ~0.000 | 0% | Cached cross-call |
| vae_encoder | 0.046 | 9% | Encodes 4 frames × 3 cams |
| **kv_prepass** | **0.080** | **16%** | Misnamed — it's a full DiT fwd at t=0 |
| **main diffusion** | **0.400** | **80%** | 40 layers × 4 DiT steps |
| scheduler | 0.019 | 4% | CPU-side step selection |
| **TOTAL per call** | **~0.505** | 100% | |

## 3. Current execution reality (from trace)

- **All 8 ranks run every phase redundantly** (data-parallel replicas). 8 GPUs each run their own text/image/VAE encoders.
- **Only CFG (2-way) and SP (4-way) are real parallelism inside DiT.** Outside DiT, everything is serial on each rank.
- **Single CUDA stream (stream 7)** carries 28,580 kernels per call — zero kernel-level concurrency. Even NCCL collectives run on the same stream.

## 4. Parallelization opportunities

### 🟢 High-value

**(1) Cross-call pipelining (async double-buffering)** — *~45 ms saved*
- Start `image_enc + vae_enc` for call N+1 **while the diffusion loop for call N is running**.
- If the client allows it, this completely hides encoders' latency.
- Current flow: client sends obs_N+1 only after receiving response N → server idle between calls.
- Requires server-side queue + overlapping the socket read with the compute.
- Cost: moderate (server restructure).

### 🟡 Medium-value

**(2) `image_enc || vae_enc` on parallel streams** — *~46 ms saved (cold); ~0 ms steady*
- Both only read image frames; outputs are independent. Issue on separate streams + synchronize before KV prepass.
- Cost: low. ~10 lines to add a `torch.cuda.Stream()` block around VAE.

**(3) `text || image || vae` full parallel** — *~10 ms saved (warm); ~60 ms cold*
- Wider fan-out of the 3 input encoders.
- Marginal in steady state since text is cached.
- Cost: low, same pattern as (2).

**(4) Fan-out encoders across ranks instead of replicating** — *up to ~60 ms*
- Currently all 8 ranks redundantly compute text/image/VAE. These outputs get re-derived 8 times.
- Instead: rank 0 runs text, rank 1 runs image, ranks 2-3 run VAE, then `broadcast` to the rest.
- Effective wall time for encoder phases drops from sum to max.
- Cost: HIGH — restructure server-side dispatch + add broadcasts.
- Also has a broadcast cost (tens of MB per call).

**(5) Parallel Q/K/V all-to-all inside each DiT block** — *~5-10 ms saved*
- Currently the three `a2a(q); a2a(k); a2a(v)` issue sequentially per block; 40 blocks × 4 steps × 3 a2a = 480 a2a ops.
- Issue on 3 side streams, merge before attention kernel.
- Cost: moderate (stream mgmt, event sync). Risk of correctness subtleties.

### 🔴 Low-value / infeasible

**(6) Parallelize DiT layers** — impossible. Each layer reads previous layer's output.

**(7) Parallelize DiT steps** — impossible. Each diffusion step reads previous step's denoised latent.

**(8) CFG parallelism** — already active (2-way P2P exchange). No further gain.

## 5. Fundamental ceiling for phase-level parallelism

Even with **perfect** parallelization of every phase outside DiT:

- Critical path floor = max(text, image, vae, kv_prepass) + diffusion
- = max(10, 0, 46, 80) + 400
- = 80 + 400 = **480 ms**

vs current **505 ms**. Only **~25 ms** saveable at phase-level even in the best case.

**So phase-level parallelism gives at most ~5% improvement.** The real bottleneck is the 400 ms diffusion loop, which can only be attacked by:
- Kernel launch overhead reduction (CUDA graphs / compile)
- Shorter sequences (already compressed)
- Lower precision (FP8 — hurts at this seq length per README)
- Fewer DiT steps (already dynamic-cached down to 4)
- Fewer layers (architectural change)

## 6. Cross-GPU parallelism state

| Axis | Currently used? | Ranks | Purpose |
|---|---|---:|---|
| Data parallel (redundant) | N/A | N/A | Weights are replicated; ranks don't split batches |
| CFG parallel | ✅ | 2 | Conditional + unconditional denoising split |
| SP (Ulysses) | ✅ | 4 | Attention heads + sequence split per CFG group |
| Pipeline parallel | ❌ | — | Not applicable (single call, short sequence) |
| **Encoder parallelism across ranks** | ❌ | — | **Opportunity #4 above** |

## 7. Recommended action order (phase-level)

1. **Fix COMPILE_DIT** (current attempt: `mode="default"`) — tackles the 400 ms diffusion loop launch overhead, which dominates. Biggest single lever.
2. **Cross-call pipelining** (#1) — hides encoder phases entirely. ~45 ms / 9% saved.
3. **`image_enc || vae_enc` streams** (#2) — trivial change, ~46 ms cold-path savings.
4. Everything else (rank-fan-out, parallel Q/K/V a2a) — higher cost, lower return.
