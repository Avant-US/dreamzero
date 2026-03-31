# Sequence Parallelism for DreamZero Inference

## Motivation

The inference bottleneck is the **CausalWanModel** (14B parameter DiT) being called N times per denoising step (x2 for CFG). Currently, multi-GPU support is limited to **2-GPU CFG splitting** — one GPU runs the conditional pass, the other unconditional. Each GPU runs the full model on the full token sequence.

**Goal**: Add sequence parallelism (SP) so we can use 2+ GPUs per CFG branch. For example, 4 GPUs total = 2 CFG groups x 2 SP ranks. This splits the attention computation across GPUs within each CFG branch, reducing per-GPU compute and memory.

**No retraining required** — SP is mathematically identical to single-GPU execution. Same weights, same outputs (up to floating point nondeterminism).

## Approach: Ulysses-Style SP

We use **Ulysses SP** (all-to-all on attention heads) rather than ring attention. This avoids any changes to the complex blockwise causal attention masks, RoPE, or action/state register logic.

### How Ulysses SP works

For cheap ops (FFN, linear projections), each GPU holds a **chunk of the sequence**:
```
GPU 0: tokens [0..L/2]     (all heads)
GPU 1: tokens [L/2..L]     (all heads)
```

Before attention, an **all-to-all** swaps to full sequence / subset of heads:
```
GPU 0: tokens [0..L]       (heads 0..19)
GPU 1: tokens [0..L]       (heads 20..39)
```

Attention runs normally — each GPU sees all tokens but fewer heads. FlashAttention is head-parallel by nature, so causal masks, RoPE, and KV cache logic are **completely unchanged**.

After attention, another **all-to-all** swaps back to local sequence / all heads.

### Constraint

`num_heads % sp_size == 0`. The 14B model has 40 heads:
- SP=2: 20 heads/GPU
- SP=4: 10 heads/GPU
- SP=5: 8 heads/GPU
- SP=8: 5 heads/GPU

## Architecture

```
Example: 4 GPUs (ip=2, sp=2)

GPU 0 ─┐ SP Group A ─── conditional CFG branch
GPU 1 ─┘

GPU 2 ─┐ SP Group B ─── unconditional CFG branch
GPU 3 ─┘

Within each SP group:
  - Sequence is split across 2 GPUs
  - All-to-all before/after attention
  - After denoising, gather full output

Between CFG groups:
  - P2P exchange of predictions (existing logic, SP rank 0 only)
  - Broadcast result within SP group
```

## Implementation Plan

### Step 0: SP Communication Primitives (new file)

**Create**: `groot/vla/model/dreamzero/modules/sequence_parallel.py`

Contents:
- `SequenceParallelContext` dataclass — holds `sp_group`, `sp_rank`, `sp_size`
- `split_sequence(tensor, dim, sp_ctx)` — `torch.chunk` along dim, return local chunk
- `gather_sequence(tensor, dim, sp_ctx)` — `all_gather` along dim to reconstruct full tensor
- `all_to_all(tensor, scatter_dim, gather_dim, sp_ctx)` — head <-> sequence swap for Ulysses

**Dependencies**: None (standalone utility module).

---

### Step 1: 2D DeviceMesh

**Modify**: `socket_test_optimized_AR.py`

Change mesh from 1D `("ip",)` to 2D `("ip", "sp")`:

```python
# Before:
mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("ip",))

# After:
sp_size = args.sp_size  # new CLI arg, default 1
cfg_size = world_size // sp_size
mesh = init_device_mesh("cuda", mesh_shape=(cfg_size, sp_size), mesh_dim_names=("ip", "sp"))
```

Add `--sp-size` CLI argument. When `sp_size=1`, behavior is identical to current code.

**Dependencies**: None.

---

### Step 2: Extract SP Mesh in parallelize()

**Modify**: `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
**Method**: `WANPolicyHead.parallelize()` (line 1404)

```python
# Existing:
def parallelize(self, device_mesh):
    ip_mesh = device_mesh["ip"]
    self.ip_rank = ip_mesh.get_local_rank()
    self.ip_size = ip_mesh.size()
    self.ip_group = ip_mesh.get_group()

# Add:
    if "sp" in device_mesh.mesh_dim_names:
        sp_mesh = device_mesh["sp"]
        self.sp_ctx = SequenceParallelContext(
            sp_group=sp_mesh.get_group(),
            sp_rank=sp_mesh.get_local_rank(),
            sp_size=sp_mesh.size(),
        )
    else:
        self.sp_ctx = None
    self.model.set_sp_context(self.sp_ctx)
```

**Dependencies**: Steps 0, 1.

---

### Step 3: Propagate SP Context Through CausalWanModel

**Modify**: `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py`
**Class**: `CausalWanModel`

Add method:
```python
def set_sp_context(self, sp_ctx):
    self.sp_ctx = sp_ctx
    for block in self.blocks:
        block.self_attn.sp_ctx = sp_ctx
```

Cross-attention modules do NOT need SP context (see Step 6).

**Dependencies**: Step 0.

---

### Step 4: Split/Gather Sequence in _forward_blocks

**Modify**: `wan_video_dit_action_casual_chunk.py`
**Method**: `CausalWanModel._forward_blocks()` (line 1735)

After patch embedding + action register concat (line ~1765):
```python
if self.sp_ctx is not None and self.sp_ctx.sp_size > 1:
    x = split_sequence(x, dim=1, sp_ctx=self.sp_ctx)
    e0 = split_sequence(e0, dim=1, sp_ctx=self.sp_ctx)
```

After all transformer blocks, before extracting outputs (line ~1815):
```python
if self.sp_ctx is not None and self.sp_ctx.sp_size > 1:
    x = gather_sequence(x, dim=1, sp_ctx=self.sp_ctx)
```

**Padding**: If `total_seq_len % sp_size != 0`, pad to the nearest multiple before splitting, track padding amount, and trim after gathering.

**Dependencies**: Steps 0, 3.

---

### Step 5: Ulysses All-to-All in CausalWanSelfAttention (CORE CHANGE)

**Modify**: `wan_video_dit_action_casual_chunk.py`
**Method**: `CausalWanSelfAttention.forward()` (line 790)

After Q/K/V projection, before RoPE:
```python
if self.sp_ctx is not None and self.sp_ctx.sp_size > 1:
    q = all_to_all(q, scatter_dim=1, gather_dim=2, sp_ctx=self.sp_ctx)
    k = all_to_all(k, scatter_dim=1, gather_dim=2, sp_ctx=self.sp_ctx)
    v = all_to_all(v, scatter_dim=1, gather_dim=2, sp_ctx=self.sp_ctx)
    # Shape: [B, L_local, num_heads, head_dim] -> [B, L_full, num_heads/sp_size, head_dim]
```

After attention output, before output projection:
```python
if self.sp_ctx is not None and self.sp_ctx.sp_size > 1:
    x = all_to_all(x, scatter_dim=2, gather_dim=1, sp_ctx=self.sp_ctx)
    # Shape: [B, L_full, num_heads/sp_size, head_dim] -> [B, L_local, num_heads, head_dim]
```

**Why this works**: After the first all-to-all, each rank has the FULL sequence but only `num_heads/sp_size` heads. All existing attention logic (blockwise causal masks, RoPE, FlashAttention) operates on the full sequence with fewer heads — completely unchanged.

**KV cache inference path** (lines 1021-1091):
- New K/V are all-to-all'd before appending to cache
- Cache stores `num_heads/sp_size` heads per rank
- Attention runs on full sequence with sharded heads
- Output is all-to-all'd back

**Dependencies**: Steps 0, 3, 4.

---

### Step 6: Cross-Attention — No Changes Needed

**Files**: `groot/vla/model/dreamzero/modules/wan2_1_submodule.py`
**Classes**: `WanT2VCrossAttention`, `WanI2VCrossAttention`

Cross-attention Q comes from `x` (local sequence chunk, all heads). K/V come from `context` (text/CLIP embeddings, replicated on all ranks, small — ~512+257 tokens).

Each SP rank computes cross-attention for its local Q tokens against the full context using all heads. Output has shape `[B, L_local, C]` matching `x`. **No all-to-all needed.**

Cross-attention cache (`crossattn_cache`) stores K/V for replicated context — identical across SP ranks. No changes needed.

**Dependencies**: Verification only.

---

### Step 7: KV Cache Head-Sharding

**Modify**: `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py`
**Method**: `WANPolicyHead._create_kv_caches()` (line 484)

```python
# Before:
cache_shape = [..., num_heads, head_dim]

# After:
effective_num_heads = num_heads // (self.sp_ctx.sp_size if self.sp_ctx else 1)
cache_shape = [..., effective_num_heads, head_dim]
```

**Dependencies**: Steps 2, 5.

---

### Step 8: Update _exchange_predictions for SP

**Modify**: `wan_flow_matching_action_tf.py`
**Method**: `WANPolicyHead._exchange_predictions()` (line 918)

After gather, all SP ranks within a CFG group have identical predictions. Only SP rank 0 does the P2P exchange with the other CFG group, then broadcasts within its SP group:

```python
if self.sp_ctx is not None and self.sp_ctx.sp_size > 1:
    if self.sp_ctx.sp_rank == 0:
        # existing P2P exchange logic
        ...
    # broadcast result to other SP ranks
    dist.broadcast(other_predictions, src=sp_rank_0_global, group=self.sp_ctx.sp_group)
```

**Dependencies**: Steps 2, 4.

---

### Step 9: Inference Loop Verification

**Modify**: `wan_flow_matching_action_tf.py`
**Method**: `WANPolicyHead.lazy_joint_video_action()` (line 980)

Verify:
- **Noise generation**: Must be identical across SP ranks. Uses `self.seed` — deterministic. OK.
- **Encoders** (VAE, T5, CLIP): Run before diffusion loop. Run identically on all SP ranks (same inputs, deterministic). OK.
- **Scheduler step**: Runs on full prediction tensor after CFG combine + gather. Identical on all SP ranks. OK.

**Dependencies**: Steps 4, 8.

---

## Implementation Order

```
Step 0 (primitives) ───┐
                       v
Step 1 (2D mesh)  ──> Step 2 (parallelize) ──> Step 3 (propagate)
                                                      |
                                          +-----------+-----------+
                                          |           |           |
                                     Step 5       Step 4       Step 7
                                     (core)      (split)     (KV cache)
                                          |           |
                                     Step 6       Step 8
                                    (verify)    (exchange)
                                                      |
                                                  Step 9
                                                 (verify)
```

Steps 4, 5, 7 can be developed in parallel once Step 3 is done.

## Risk Areas

| Risk | Impact | Mitigation |
|------|--------|------------|
| Sequence length not divisible by `sp_size` | Crash or incorrect results | Pad to nearest multiple, mask after gather |
| `torch.compile` + all-to-all collectives | May fail with `fullgraph=True` | Use `allow_in_graph` or disable fullgraph for SP path |
| Noise divergence across SP ranks | Different predictions per rank | Verify `self.seed` produces identical noise; add assert |
| P2P group construction with 2D mesh | Wrong CFG peer pairing | Verify `device_mesh["ip"].get_group()` gives correct peers |
| Action/state register tokens in split | Uneven split boundaries | Validate `total_seq_len % sp_size == 0` at init; pad if needed |
| KV cache with sharded heads | Cache corruption | Unit test: compare SP=2 output vs SP=1 output |

## Key Files

| File | Changes |
|------|---------|
| `groot/vla/model/dreamzero/modules/sequence_parallel.py` | **NEW** — SP primitives |
| `groot/vla/model/dreamzero/modules/wan_video_dit_action_casual_chunk.py` | `CausalWanSelfAttention` all-to-all, `CausalWanModel._forward_blocks` split/gather, `set_sp_context()` |
| `groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py` | `parallelize()`, `_create_kv_caches()`, `_exchange_predictions()` |
| `groot/vla/model/dreamzero/modules/wan2_1_submodule.py` | Verify cross-attention (likely no changes) |
| `socket_test_optimized_AR.py` | 2D DeviceMesh, `--sp-size` CLI arg |

## Testing Strategy

1. **Correctness**: Run inference with SP=1 and SP=2, compare outputs. Should be identical (up to float nondeterminism).
2. **Latency**: Benchmark denoising loop time with SP=1 vs SP=2 vs SP=4.
3. **Memory**: Check per-GPU VRAM usage — should decrease with SP (smaller KV cache, smaller activations).
4. **Edge cases**: Test with action horizon and frame sizes that don't divide evenly by `sp_size`.
