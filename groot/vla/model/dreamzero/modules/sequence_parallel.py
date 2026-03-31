"""Sequence parallelism (Ulysses-style) communication primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class SequenceParallelContext:
    """Holds the process group and rank info for one SP group."""
    sp_group: dist.ProcessGroup
    sp_rank: int
    sp_size: int
    # Set by _forward_blocks before the block loop so self-attention can trim padding
    original_seq_len: int | None = None


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
    """
    gathered = [torch.empty_like(tensor) for _ in range(sp_ctx.sp_size)]
    dist.all_gather(gathered, tensor, group=sp_ctx.sp_group)
    result = torch.cat(gathered, dim=dim)
    if original_length is not None:
        result = result.narrow(dim, 0, original_length)
    return result


def all_to_all(tensor: torch.Tensor, scatter_dim: int, gather_dim: int,
               sp_ctx: SequenceParallelContext) -> torch.Tensor:
    """Ulysses all-to-all: scatter along *scatter_dim*, gather along *gather_dim*.

    Used to swap between sequence-parallel layout (local sequence, all heads)
    and head-parallel layout (full sequence, local heads).

    Assumes both scatter_dim and gather_dim are evenly divisible by sp_size.
    """
    sp_size = sp_ctx.sp_size

    # Split along scatter_dim into sp_size chunks
    input_chunks = tensor.chunk(sp_size, dim=scatter_dim)
    # Each chunk: make contiguous for communication
    input_chunks = [c.contiguous() for c in input_chunks]

    output_chunks = [torch.empty_like(c) for c in input_chunks]
    dist.all_to_all(output_chunks, input_chunks, group=sp_ctx.sp_group)

    # Concatenate along gather_dim
    return torch.cat(output_chunks, dim=gather_dim)
