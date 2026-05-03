"""Piecewise CUDA Graph runner for DiT inference (sglang/vLLM-style).

Restructures the DiT block loop into:
  - COMPUTE phases → captured as CUDA graphs (zero launch overhead)
  - NCCL phases (all_to_all) → run eagerly between graph replays

Each of the 40 DiT blocks has its compute captured as small graphs.
The block loop runs in Python, dispatching graph.replay() for compute
and dist.all_to_all() for communication.

This eliminates ~28,000 kernel launches per inference call, replacing
them with ~40 graph replays + ~160 NCCL calls = ~200 total launches.

Architecture follows sglang's piecewise approach:
  - Pre-allocate ALL intermediate buffers at max size
  - Warmup each phase eagerly (3 runs)
  - Capture each phase into its own CUDAGraph with a shared pool
  - On subsequent calls: .copy_() into static buffers → graph.replay()

Requirements:
  - STATIC_KV_CACHE=true (pre-allocated KV buffers)
  - All tensor inputs have stable shapes after warmup

Enable via: PIECEWISE_CUDA_GRAPH=true
"""
from __future__ import annotations

import gc
import os
from typing import Any

import torch
import torch.distributed as dist

# Number of eager warmup runs before capture (same as vLLM)
_NUM_WARMUP = 3


class PiecewiseDiTRunner:
    """Piecewise CUDA graph capture for the 40-layer DiT block loop.

    Instead of capturing the ENTIRE _forward_blocks (which contains NCCL ops
    that can't be captured), we capture each block's COMPUTE phases separately
    and run NCCL ops eagerly between replays.

    Phases per block:
      A. pre_self_attn: norm + modulation + QKV projection
      B. a2a_forward: all_to_all for q, k, v (NCCL — EAGER)
      C. attn_core: RoPE + attention + KV cache write
      D. a2a_reverse: all_to_all for attention output (NCCL — EAGER)
      E. post_attn: output projection + residual + cross-attn + FFN

    Phases A, C, E are captured as CUDA graphs. B, D run eagerly.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.pool = torch.cuda.graph_pool_handle()

        # Per-block captured graphs. Keyed by (block_idx, phase_name, shape_key).
        self._graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self._static_ios: dict[tuple, dict[str, torch.Tensor]] = {}
        self._warmup_counts: dict[tuple, int] = {}
        self._capture_stream = torch.cuda.Stream(device=device)

        self._num_blocks = len(model.blocks)
        print(f"[PiecewiseDiTRunner] Initialized for {self._num_blocks} blocks on {device}")

    # TODO: Implement phase-by-phase capture.
    #
    # This requires refactoring each CausalWanAttentionBlock.forward into
    # separate callable phases. The current block.forward interleaves compute
    # and NCCL (a2a inside self_attn). To capture compute-only phases, we need
    # to split self_attn into:
    #   1. qkv_projection(x, freqs) -> q, k, v  [COMPUTE]
    #   2. all_to_all(q, k, v)                   [NCCL]
    #   3. attention_core(q, k, v, kv_cache)      [COMPUTE]
    #   4. all_to_all(out)                        [NCCL]
    #   5. output_proj(out) + residual            [COMPUTE]
    #
    # Each compute phase is a small function that we can:
    #   - Warmup eagerly
    #   - Capture into its own CUDAGraph (with shared pool)
    #   - Replay on subsequent calls
    #
    # The restructuring work:
    #   a. Add methods to CausalWanSelfAttention:
    #      - qkv_project(x, freqs, ...) -> q, k, v, metadata
    #      - attention_and_cache(q, k, v, kv_cache, ...) -> out
    #      - output_project(out) -> y
    #   b. Add method to CausalWanAttentionBlock:
    #      - pre_self_attn(x, e) -> x_normed, e_parts
    #      - post_self_attn(y, x, e, context) -> x_new
    #   c. The runner calls these methods in order, capturing each as a graph.
    #
    # For now, this file serves as the DESIGN SPEC for the piecewise approach.
    # The next step is to refactor the block/self_attn modules to expose
    # these phase methods, then implement capture/replay here.
    #
    # Estimated effort: ~300 LOC refactor of block + self_attn + this runner.
    # Expected result: ~40 graph replays + ~160 NCCL calls per forward
    #                  = ~200 kernel launches (vs ~28,000 today)
    #                  = ~99.3% reduction in launch overhead
    #                  = ~0.15-0.25s per inference call (target: 0.2s)
