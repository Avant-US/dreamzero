"""Compile-safe wrappers for SP collective ops.

Registers all-to-all, split, and gather as torch.library.custom_op with
fake (meta) implementations so torch.compile can trace through them with
concrete output shapes. The actual NCCL calls happen at runtime only.

Uses pynccl (graph-safe ncclSend/ncclRecv on current stream) when inside
CUDA graph capture, falls back to dist.all_to_all_single for eager mode.

This avoids:
- InductorError: symbolic Mul shapes from @torch.compiler.disable
- Graph breaks from data-dependent control flow
- NCCL ops that the inductor can't codegen
"""

import torch
import torch.distributed as dist

from groot.vla.model.dreamzero import perf_profile as _perf_profile


# ---------------------------------------------------------------------------
# Global caches (custom ops can't take ProcessGroup / pynccl args)
# ---------------------------------------------------------------------------

_GROUP_CACHE: dict[str, dist.ProcessGroup] = {}
_PYNCCL_CACHE: dict[str, object] = {}  # name -> PyNcclAllToAll instance


def register_group(name: str, group: dist.ProcessGroup):
    """Register a process group by name for use in custom ops."""
    _GROUP_CACHE[name] = group


def register_pynccl(name: str, comm):
    """Register a pynccl communicator for graph-safe collectives."""
    _PYNCCL_CACHE[name] = comm


def _get_group(name: str) -> dist.ProcessGroup:
    return _GROUP_CACHE[name]


def _get_pynccl(name: str):
    return _PYNCCL_CACHE.get(name)


# ---------------------------------------------------------------------------
# all_to_all custom op: scatter one dim, gather another
# ---------------------------------------------------------------------------

@torch.library.custom_op("sp::all_to_all", mutates_args=())
def sp_all_to_all(
    tensor: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    sp_size: int,
    sp_group_name: str,
) -> torch.Tensor:
    """All-to-all: scatter along scatter_dim, gather along gather_dim.

    Uses pynccl (graph-safe) when available, else dist.all_to_all_single.
    """
    pynccl = _get_pynccl(sp_group_name)
    if pynccl is not None and getattr(pynccl, 'available', False):
        # Graph-safe path: ncclSend/ncclRecv on current stream
        return pynccl.all_to_all(tensor, scatter_dim, gather_dim)

    # Eager fallback: dist.all_to_all_single
    group = _get_group(sp_group_name)
    assert tensor.ndim == 4
    b, s, h, d = tensor.shape
    assert tensor.shape[scatter_dim] % sp_size == 0

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
        dist.all_to_all_single(flat_out, flat_in, group=group)
        e_ev.record()
        _perf_profile.record_comm("a2a", s_ev, e_ev)
    else:
        dist.all_to_all_single(flat_out, flat_in, group=group)

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


@sp_all_to_all.register_fake
def _sp_all_to_all_fake(
    tensor: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    sp_size: int,
    sp_group_name: str,
) -> torch.Tensor:
    """Fake impl: compute output shape without NCCL."""
    b, s, h, d = tensor.shape
    out_shape = list(tensor.shape)
    out_shape[scatter_dim] = out_shape[scatter_dim] // sp_size
    out_shape[gather_dim] = out_shape[gather_dim] * sp_size
    return tensor.new_empty(out_shape)


# ---------------------------------------------------------------------------
# split_sequence custom op
# ---------------------------------------------------------------------------

@torch.library.custom_op("sp::split_seq", mutates_args=())
def sp_split_seq(
    tensor: torch.Tensor,
    dim: int,
    sp_rank: int,
    sp_size: int,
) -> torch.Tensor:
    """Split tensor along dim, return local chunk for sp_rank."""
    # .contiguous().clone() required: custom_op outputs must not alias inputs
    # AND must have standard contiguous strides for CUDA graph stride assertions
    chunks = tensor.chunk(sp_size, dim=dim)
    return chunks[sp_rank].contiguous().clone()


@sp_split_seq.register_fake
def _sp_split_seq_fake(
    tensor: torch.Tensor,
    dim: int,
    sp_rank: int,
    sp_size: int,
) -> torch.Tensor:
    out_shape = list(tensor.shape)
    out_shape[dim] = out_shape[dim] // sp_size
    return tensor.new_empty(out_shape)


# ---------------------------------------------------------------------------
# gather_sequence custom op
# ---------------------------------------------------------------------------

@torch.library.custom_op("sp::gather_seq", mutates_args=())
def sp_gather_seq(
    tensor: torch.Tensor,
    dim: int,
    sp_size: int,
    sp_group_name: str,
    original_length: int,
) -> torch.Tensor:
    """All-gather tensor along dim, trim to original_length."""
    pynccl = _get_pynccl(sp_group_name)
    if pynccl is not None and getattr(pynccl, 'available', False):
        # Graph-safe path
        result = pynccl.all_gather(tensor, dim)
    else:
        # Eager fallback
        group = _get_group(sp_group_name)
        gathered = [torch.empty_like(tensor) for _ in range(sp_size)]

        if _perf_profile.enabled():
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
            dist.all_gather(gathered, tensor, group=group)
            e_ev.record()
            _perf_profile.record_comm("allgather", s_ev, e_ev)
        else:
            dist.all_gather(gathered, tensor, group=group)

        result = torch.cat(gathered, dim=dim)

    if original_length > 0:
        result = result.narrow(dim, 0, original_length)
    return result


@sp_gather_seq.register_fake
def _sp_gather_seq_fake(
    tensor: torch.Tensor,
    dim: int,
    sp_size: int,
    sp_group_name: str,
    original_length: int,
) -> torch.Tensor:
    out_shape = list(tensor.shape)
    out_shape[dim] = original_length if original_length > 0 else out_shape[dim] * sp_size
    return tensor.new_empty(out_shape)
