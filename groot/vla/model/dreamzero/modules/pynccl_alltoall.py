"""Pure-Python NCCL wrapper for graph-capturable all-to-all.

Adapted from vLLM's pynccl_wrapper.py. The key difference from
torch.distributed.all_to_all: this runs NCCL ops on the CURRENT CUDA
stream (not a side stream), making it capturable in CUDA graphs.

Usage:
    comm = PyNcclAllToAll(sp_group, device)  # once at init
    output = comm.all_to_all(tensor, scatter_dim, gather_dim)  # graph-safe
"""
from __future__ import annotations

import ctypes
import os
from typing import Optional

import torch
import torch.distributed as dist

# NCCL C types
ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p
cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p
ncclDataType_t = ctypes.c_int


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


# NCCL dtype map
_TORCH_TO_NCCL = {
    torch.float16: 6,
    torch.float32: 7,
    torch.float64: 8,
    torch.bfloat16: 9,
    torch.int8: 0,
    torch.int32: 2,
    torch.int64: 4,
}


def _find_nccl_library() -> str:
    """Find libnccl.so — check common locations."""
    candidates = [
        os.environ.get("NCCL_SO_PATH", ""),
        "libnccl.so.2",
        "libnccl.so",
    ]
    # Also check torch's bundled NCCL
    try:
        import torch.cuda
        torch_lib = os.path.join(os.path.dirname(torch.cuda.__file__), "..", "lib")
        for f in os.listdir(torch_lib):
            if f.startswith("libnccl"):
                candidates.append(os.path.join(torch_lib, f))
    except Exception:
        pass
    # Check nvidia pip package
    try:
        import nvidia.nccl
        nccl_dir = os.path.dirname(nvidia.nccl.__file__)
        for f in os.listdir(os.path.join(nccl_dir, "lib")):
            if f.startswith("libnccl"):
                candidates.append(os.path.join(nccl_dir, "lib", f))
    except Exception:
        pass

    for path in candidates:
        if not path:
            continue
        try:
            ctypes.CDLL(path)
            return path
        except OSError:
            continue
    raise RuntimeError("Cannot find libnccl.so. Set NCCL_SO_PATH env var.")


class PyNcclAllToAll:
    """Graph-capturable all-to-all via direct NCCL C calls.

    Creates a SEPARATE NCCL communicator (independent from torch.distributed)
    that runs on the CURRENT CUDA stream. This makes ncclSend/ncclRecv
    capturable in CUDA graphs.
    """

    def __init__(self, group: dist.ProcessGroup, device: torch.device):
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.device = device
        self.group = group

        if self.world_size <= 1:
            self.available = False
            return

        # Load NCCL library
        lib_path = _find_nccl_library()
        self.nccl = ctypes.CDLL(lib_path)

        # Get unique ID from rank 0, broadcast to all
        unique_id = ncclUniqueId()
        if self.rank == 0:
            self.nccl.ncclGetUniqueId(ctypes.byref(unique_id))

        # Sync all CUDA ops + dist before creating a second NCCL communicator.
        # Multiple comms CAN coexist but must not overlap with active ops.
        torch.cuda.synchronize()
        dist.barrier(group=group)

        # Broadcast unique ID via torch.distributed (init-time only, not graph-captured)
        id_tensor = torch.ByteTensor(list(unique_id.internal)).cuda(device)
        ranks = dist.get_process_group_ranks(group)
        dist.broadcast(id_tensor, src=ranks[0], group=group)
        for i, b in enumerate(id_tensor.cpu().tolist()):
            unique_id.internal[i] = b

        # Sync again before creating the new communicator
        torch.cuda.synchronize()
        dist.barrier(group=group)

        # Create SEPARATE NCCL communicator for graph-captured all-to-all
        self.comm = ncclComm_t()
        with torch.cuda.device(device):
            result = self.nccl.ncclCommInitRank(
                ctypes.byref(self.comm),
                ctypes.c_int(self.world_size),
                unique_id,
                ctypes.c_int(self.rank),
            )
            assert result == 0, f"ncclCommInitRank failed: {result}"

        # Sync after creation
        torch.cuda.synchronize()
        dist.barrier(group=group)

        self.available = True

        # NOTE: No warmup — keep the communicator pristine for graph capture.
        # Any use before graph capture would increment NCCL's internal counter,
        # potentially causing replay issues.  The first use will be inside
        # torch.cuda.graph() capture, exactly matching the replay sequence.

        if self.rank == 0:
            print(f"[PyNcclAllToAll] Initialized: rank={self.rank}, world={self.world_size}, device={device}")

    def all_to_all(self, tensor: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor:
        """Graph-capturable all-to-all using ncclSend/ncclRecv on current stream."""
        if not self.available:
            return tensor

        sp_size = self.world_size
        assert tensor.shape[scatter_dim] % sp_size == 0

        # Split into chunks along scatter_dim
        chunks = tensor.chunk(sp_size, dim=scatter_dim)
        chunks = [c.contiguous() for c in chunks]
        out_chunks = [torch.empty_like(c) for c in chunks]

        nccl_dtype = ncclDataType_t(_TORCH_TO_NCCL[tensor.dtype])
        stream = cudaStream_t(torch.cuda.current_stream(self.device).cuda_stream)
        count = ctypes.c_size_t(chunks[0].numel())

        # ncclGroupStart + ncclSend/ncclRecv + ncclGroupEnd
        # ALL on current stream → graph-capturable
        self.nccl.ncclGroupStart()
        for peer in range(sp_size):
            self.nccl.ncclSend(
                buffer_type(chunks[peer].data_ptr()),
                count,
                nccl_dtype,
                ctypes.c_int(peer),
                self.comm,
                stream,
            )
            self.nccl.ncclRecv(
                buffer_type(out_chunks[peer].data_ptr()),
                count,
                nccl_dtype,
                ctypes.c_int(peer),
                self.comm,
                stream,
            )
        self.nccl.ncclGroupEnd()

        return torch.cat(out_chunks, dim=gather_dim)

    def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        """Graph-capturable all-gather using ncclAllGather on current stream."""
        if not self.available:
            return tensor

        nccl_dtype = ncclDataType_t(_TORCH_TO_NCCL[tensor.dtype])
        stream = cudaStream_t(torch.cuda.current_stream(self.device).cuda_stream)
        count = ctypes.c_size_t(tensor.numel())

        out = torch.empty(
            *tensor.shape[:dim],
            tensor.shape[dim] * self.world_size,
            *tensor.shape[dim + 1:],
            dtype=tensor.dtype,
            device=tensor.device,
        )
        # ncclAllGather writes [rank0_chunk | rank1_chunk | ... ] contiguously.
        # We need to reshape afterwards to interleave along `dim`.
        # Simplest: allgather into flat buffer, then split and cat.
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]

        self.nccl.ncclGroupStart()
        for peer in range(self.world_size):
            # Broadcast our tensor to everyone
            self.nccl.ncclSend(
                buffer_type(tensor.data_ptr()),
                count,
                nccl_dtype,
                ctypes.c_int(peer),
                self.comm,
                stream,
            )
            # Receive peer's tensor
            self.nccl.ncclRecv(
                buffer_type(gathered[peer].data_ptr()),
                count,
                nccl_dtype,
                ctypes.c_int(peer),
                self.comm,
                stream,
            )
        self.nccl.ncclGroupEnd()

        return torch.cat(gathered, dim=dim)
