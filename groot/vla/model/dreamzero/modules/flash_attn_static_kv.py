"""Flash Attention with static KV cache for CUDA graph compatibility.

Uses flash_attn_varlen_func with cu_seqlens to skip padding positions
in the pre-allocated KV buffer. This gives both:
1. Static tensor shapes (CUDA graph safe)
2. O(Q × valid_len) compute (no wasted work on padding)

Compare with SDPA + mask approach which has static shapes but
O(Q × max_attn) compute (wastes work on all 7920 padded positions).
"""
from __future__ import annotations

import torch

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


def flash_attn_static_kv(
    q: torch.Tensor,           # [B, S_q, H, D]
    kv_cache_k: torch.Tensor,  # [B, max_attn, H, D] (pre-allocated, static shape)
    kv_cache_v: torch.Tensor,  # [B, max_attn, H, D]
    valid_len_t: torch.Tensor, # 0-d int64 GPU tensor: how many KV positions are valid
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run flash attention that only attends over valid_len positions of KV cache.

    All tensor shapes are static (CUDA graph safe). Only valid_len_t changes
    value between calls (via .fill_() before graph replay).

    Returns: [B, S_q, H, D]
    """
    assert FLASH_ATTN_AVAILABLE, "flash_attn not installed"

    B, S_q, H, D = q.shape
    max_attn = kv_cache_k.shape[1]

    # Flatten batch dim (B=1 for our inference)
    q_flat = q.reshape(-1, H, D)         # [B*S_q, H, D]
    k_flat = kv_cache_k.reshape(-1, H, D)  # [B*max_attn, H, D]
    v_flat = kv_cache_v.reshape(-1, H, D)

    # cu_seqlens: cumulative sequence lengths. For B=1:
    # cu_seqlens_q = [0, S_q]
    # cu_seqlens_k = [0, valid_len]
    # These are pre-allocated GPU tensors — just update the values.
    cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
    cu_seqlens_q[1] = S_q

    cu_seqlens_k = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
    cu_seqlens_k[1] = valid_len_t.to(torch.int32)

    # max_seqlen values: these are Python ints, baked into the graph.
    # max_seqlen_q = S_q (constant per shape key)
    # max_seqlen_k = max_attn (always the full buffer size — flash_attn
    #   reads up to cu_seqlens_k[1] but needs max_seqlen_k for indexing)
    out = flash_attn_varlen_func(
        q=q_flat,
        k=k_flat,
        v=v_flat,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=S_q,
        max_seqlen_k=max_attn,
        softmax_scale=softmax_scale,
        causal=False,  # We handle causality via KV cache structure
    )

    return out.reshape(B, S_q, H, D)
