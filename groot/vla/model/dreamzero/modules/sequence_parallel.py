"""Sequence parallelism (Ulysses-style) communication primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from groot.vla.model.dreamzero import perf_profile as _perf_profile


@dataclass
class SequenceParallelContext:
    """Holds the process group and rank info for one SP group."""
    sp_group: dist.ProcessGroup
    sp_rank: int
    sp_size: int
    # Set by _forward_blocks before the block loop so self-attention can trim padding
    original_seq_len: int | None = None
    # Optional graph-safe NCCL communicator (vLLM pattern: separate comm on current stream)
    pynccl_comm: object | None = None
    # Set to True ONLY inside torch.cuda.graph() capture context.
    # The pynccl comm must not be used outside graph capture (its internal state
    # must stay pristine between capture and replay — vLLM/NCCL graph requirement).
    _graph_capture_active: bool = False


def split_sequence(tensor: torch.Tensor, dim: int, sp_ctx: SequenceParallelContext,
                   alignment: int = 1) -> torch.Tensor:
    """Split *tensor* along *dim* across SP ranks, returning the local chunk.

    If the dimension is not evenly divisible by sp_size, the tensor is padded
    with zeros before splitting and the caller must track the original length
    for later trimming.

    *alignment*: each local chunk's size along *dim* will be a multiple of this
    value.  For FP8, pass alignment=8 so that te.Linear gets valid shapes.
    """
    length = tensor.shape[dim]
    # Pad to multiple of (sp_size * alignment) so each chunk is a multiple of alignment
    chunk_alignment = sp_ctx.sp_size * alignment
    pad_amount = (chunk_alignment - length % chunk_alignment) % chunk_alignment
    if pad_amount > 0:
        pad_sizes = [0] * (2 * tensor.ndim)
        # torch.nn.functional.pad uses (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
        pad_index = 2 * (tensor.ndim - 1 - dim) + 1
        pad_sizes[pad_index] = pad_amount
        # Reverse: pad expects trailing-dim-first ordering
        # Actually pad_sizes is already in trailing-first order by construction above
        # Let me just be explicit:
        pad_sizes_ordered = [0] * (2 * tensor.ndim)
        # For dimension `dim`, pad on the right
        # In torch.nn.functional.pad, dimensions are indexed from the last:
        # index 0,1 = last dim; 2,3 = second-to-last dim; etc.
        rev_dim = tensor.ndim - 1 - dim
        pad_sizes_ordered[2 * rev_dim + 1] = pad_amount
        tensor = torch.nn.functional.pad(tensor, pad_sizes_ordered)
    chunks = tensor.chunk(sp_ctx.sp_size, dim=dim)
    return chunks[sp_ctx.sp_rank].contiguous()


def gather_sequence(tensor: torch.Tensor, dim: int, sp_ctx: SequenceParallelContext,
                    original_length: int | None = None) -> torch.Tensor:
    """All-gather *tensor* along *dim* across SP ranks to reconstruct the full sequence.

    If *original_length* is provided, the result is trimmed to that length along *dim*.
    Uses pynccl during graph capture (same rationale as all_to_all).
    """
    _use_pynccl = (
        getattr(sp_ctx, "_graph_capture_active", False)
        and getattr(sp_ctx, "pynccl_comm", None) is not None
        and sp_ctx.pynccl_comm.available
    )

    if _use_pynccl:
        if _perf_profile.enabled():
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
            result = sp_ctx.pynccl_comm.all_gather(tensor, dim)
            e_ev.record()
            _perf_profile.record_comm("allgather_pynccl", s_ev, e_ev)
        else:
            result = sp_ctx.pynccl_comm.all_gather(tensor, dim)
    else:
        gathered = [torch.empty_like(tensor) for _ in range(sp_ctx.sp_size)]
        if _perf_profile.enabled():
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
            dist.all_gather(gathered, tensor, group=sp_ctx.sp_group)
            e_ev.record()
            _perf_profile.record_comm("allgather", s_ev, e_ev)
        else:
            dist.all_gather(gathered, tensor, group=sp_ctx.sp_group)
        result = torch.cat(gathered, dim=dim)

    if original_length is not None:
        result = result.narrow(dim, 0, original_length)
    return result


def all_to_all_fused_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         scatter_dim: int, gather_dim: int,
                         sp_ctx: SequenceParallelContext) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused QKV all-to-all (TRT-LLM pattern): 1 collective instead of 3.

    Stacks Q, K, V into [B, S, 3, H, D], performs one all_to_all_single on the
    5D tensor, then splits back. Cuts NCCL collectives per block from 4 to 2.

    Args:
        q, k, v: [B, S_local, H, D] tensors
        scatter_dim: dim to scatter (2 for heads in the 4D view = 3 in 5D)
        gather_dim: dim to gather (1 for seq)
    Returns:
        q, k, v after all-to-all: [B, S_full, H/sp, D]
    """
    sp_size = sp_ctx.sp_size
    if sp_size <= 1:
        return q, k, v

    # Stack into [B, S, 3, H, D]
    qkv = torch.stack([q, k, v], dim=2)  # [B, S_local, 3, H, D]

    # Use pynccl if in graph capture mode
    _use_pynccl = (
        getattr(sp_ctx, "_graph_capture_active", False)
        and getattr(sp_ctx, "pynccl_comm", None) is not None
        and sp_ctx.pynccl_comm.available
    )

    # 5D all-to-all: scatter heads (dim=3), gather seq (dim=1)
    # [B, S/P, 3, H, D] → [B, S, 3, H/P, D]
    b, seq, three, heads, head_dim = qkv.shape
    assert heads % sp_size == 0
    sharded_heads = heads // sp_size

    # Reshape: split heads into [sp_size, H/P]
    t = qkv.reshape(b, seq, three, sp_size, sharded_heads, head_dim)
    t = t.permute(3, 0, 1, 2, 4, 5).contiguous()  # [P, B, S/P, 3, H/P, D]

    if _use_pynccl:
        # pynccl path: use ncclSend/ncclRecv on current stream
        out_t = sp_ctx.pynccl_comm.all_to_all(t.flatten(1), scatter_dim=0, gather_dim=0)
        out_t = out_t.view_as(t)
    else:
        flat_in = t.flatten()
        flat_out = torch.empty_like(flat_in)

        if _perf_profile.enabled():
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
            dist.all_to_all_single(flat_out, flat_in, group=sp_ctx.sp_group)
            e_ev.record()
            _perf_profile.record_comm("a2a_fused_qkv", s_ev, e_ev)
        else:
            dist.all_to_all_single(flat_out, flat_in, group=sp_ctx.sp_group)
        out_t = flat_out.view_as(t)

    # [P, B, S/P, 3, H/P, D] → [B, P, S/P, 3, H/P, D] → [B, S, 3, H/P, D]
    out = out_t.permute(1, 0, 2, 3, 4, 5).contiguous()
    gathered_seq = seq * sp_size
    out = out.reshape(b, gathered_seq, three, sharded_heads, head_dim)

    # Split back to individual Q, K, V
    q_out, k_out, v_out = out.unbind(dim=2)
    return q_out, k_out, v_out


def all_to_all(tensor: torch.Tensor, scatter_dim: int, gather_dim: int,
               sp_ctx: SequenceParallelContext) -> torch.Tensor:
    """Ulysses all-to-all — graph-safe via pynccl, fallback to dist.all_to_all_single.

    When sp_ctx.pynccl_comm is set (separate NCCL communicator on current stream,
    vLLM pattern), uses it for CUDA-graph-safe all-to-all.  Otherwise falls back to
    dist.all_to_all_single (TRT-LLM flat-tensor pattern).
    """
    sp_size = sp_ctx.sp_size
    if sp_size <= 1:
        return tensor

    # ---------- pynccl path (graph-safe: separate comm, current stream) ----------
    # CRITICAL: only use pynccl inside graph capture. The pynccl comm's internal
    # NCCL state must be pristine between capture and replay — using it outside
    # the graph (warmup, eager) would corrupt counters and segfault on replay.
    _use_pynccl = (
        getattr(sp_ctx, "_graph_capture_active", False)
        and getattr(sp_ctx, "pynccl_comm", None) is not None
        and sp_ctx.pynccl_comm.available
    )
    if _use_pynccl:
        if _perf_profile.enabled():
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
            out = sp_ctx.pynccl_comm.all_to_all(tensor, scatter_dim, gather_dim)
            e_ev.record()
            _perf_profile.record_comm("a2a_pynccl", s_ev, e_ev)
        else:
            out = sp_ctx.pynccl_comm.all_to_all(tensor, scatter_dim, gather_dim)
        return out

    # ---------- fallback: dist.all_to_all_single (TRT-LLM pattern) ----------
    # tensor: [B, S, H, D] (4D)
    assert tensor.ndim == 4, f"Expected 4D, got {tensor.ndim}D"
    b, s, h, d = tensor.shape
    assert tensor.shape[scatter_dim] % sp_size == 0

    # Reshape: split scatter_dim into [sp_size, chunk], move sp_size to front
    if scatter_dim == 1:
        t = tensor.view(b, sp_size, s // sp_size, h, d)
        t = t.permute(1, 0, 2, 3, 4).contiguous()
    elif scatter_dim == 2:
        t = tensor.view(b, s, sp_size, h // sp_size, d)
        t = t.permute(2, 0, 1, 3, 4).contiguous()
    else:
        raise ValueError(f"scatter_dim must be 1 or 2, got {scatter_dim}")

    flat_in = t.flatten()
    flat_out = torch.empty_like(flat_in)

    if _perf_profile.enabled():
        s_ev = torch.cuda.Event(enable_timing=True)
        e_ev = torch.cuda.Event(enable_timing=True)
        s_ev.record()
        dist.all_to_all_single(flat_out, flat_in, group=sp_ctx.sp_group)
        e_ev.record()
        _perf_profile.record_comm("a2a", s_ev, e_ev)
    else:
        dist.all_to_all_single(flat_out, flat_in, group=sp_ctx.sp_group)

    out_t = flat_out.view_as(t)
    if gather_dim == 1:
        out = out_t.permute(1, 0, 2, 3, 4).contiguous()
        if scatter_dim == 2:
            out = out.view(b, s * sp_size, h // sp_size, d)
        else:
            out = out.view(b, s, h, d)
    elif gather_dim == 2:
        out = out_t.permute(1, 2, 0, 3, 4).contiguous()
        if scatter_dim == 1:
            out = out.view(b, s // sp_size, h * sp_size, d)
        else:
            out = out.view(b, s, h, d)
    else:
        raise ValueError(f"gather_dim must be 1 or 2, got {gather_dim}")

    return out


def all_to_all_async(tensor: torch.Tensor, scatter_dim: int, gather_dim: int,
                     sp_ctx: SequenceParallelContext) -> tuple[list, list, "dist.Work"]:
    """Non-blocking all-to-all. Returns (output_chunks, input_chunks, work).
    Call work.wait() then torch.cat(output_chunks, dim=gather_dim) to get result.
    """
    sp_size = sp_ctx.sp_size
    input_chunks = [c.contiguous() for c in tensor.chunk(sp_size, dim=scatter_dim)]
    output_chunks = [torch.empty_like(c) for c in input_chunks]
    work = dist.all_to_all(output_chunks, input_chunks, group=sp_ctx.sp_group, async_op=True)
    return output_chunks, gather_dim, work
