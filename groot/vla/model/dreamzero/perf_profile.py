"""Per-inference performance profiler for MFU + communication analysis.

Opt-in via env var `PROFILE_INFERENCE=true`. When enabled, each forward call
appends one JSON line to ${PROFILE_LOG_FILE:-/tmp/dreamzero_profile.jsonl}
containing phase breakdowns, communication totals, and an MFU estimate.

The profiler is intentionally kept to module-global state (single server
process per rank; rank 0 is the only writer). This avoids threading a context
object through dozens of call sites.

Optionally, a torch.profiler chrome trace can be captured for a single call
via `PROFILE_TRACE=true`. The trace lands at ${PROFILE_TRACE_FILE}.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import torch


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


# ── Module-global profiler state ─────────────────────────────────────────────

# Lists of (start_event, end_event) pairs, populated by the SP primitives
# during one inference call. Cleared via reset_call().
_comm_a2a_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
_comm_allgather_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

# These are set by the action-head code before it dumps (so we can compute
# MFU without re-deriving the model config here).
_active_model_cfg: dict[str, Any] = {}


def enabled() -> bool:
    return _env_true("PROFILE_INFERENCE")


def trace_enabled() -> bool:
    return _env_true("PROFILE_TRACE")


def reset_call() -> None:
    """Call at the start of each forward to clear per-call comm events."""
    _comm_a2a_events.clear()
    _comm_allgather_events.clear()


def record_comm(kind: str, start_event: torch.cuda.Event, end_event: torch.cuda.Event) -> None:
    """Called from within SP primitives. `kind` in {'a2a', 'allgather'}."""
    if not enabled():
        return
    if kind == "a2a":
        _comm_a2a_events.append((start_event, end_event))
    elif kind == "allgather":
        _comm_allgather_events.append((start_event, end_event))


def _sum_events_ms(pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> float:
    total_ms = 0.0
    for s, e in pairs:
        try:
            total_ms += s.elapsed_time(e)
        except Exception:
            # Events not fully recorded — skip.
            pass
    return total_ms


# ── MFU calculation ──────────────────────────────────────────────────────────

# H100/H200 BF16 dense peak. H200 SXM5 matches H100 on compute; the win is HBM.
# Source: NVIDIA datasheets. Adjust via PEAK_TFLOPS env var if needed.
_DEFAULT_PEAK_TFLOPS_BF16 = 989.0  # H100/H200 SXM5 dense BF16/FP16


def _peak_tflops() -> float:
    val = os.environ.get("PEAK_TFLOPS")
    if val:
        try:
            return float(val)
        except ValueError:
            pass
    return _DEFAULT_PEAK_TFLOPS_BF16


def _dit_forward_flops(
    *,
    dim: int,
    num_heads: int,
    head_dim: int | None,
    ffn_dim: int,
    seq_len: int,
    num_layers: int,
    sp_size: int,
    batch_size: int = 1,
) -> int:
    """Per-rank FLOPs for one full DiT forward (all layers, one denoising step).

    Accounts for sequence parallelism: attention projections/FFN run on seq_len/sp
    tokens per rank, while the attention kernel itself runs on seq_len tokens with
    num_heads/sp heads — the total compute divides cleanly by sp_size.

    FLOPs counted as 2 * (# multiply-adds). Returns per-rank flops for one step.
    """
    if head_dim is None:
        head_dim = dim // num_heads
    s = seq_len
    d = dim
    h = num_heads
    # Per-layer (forward only):
    #   Q/K/V projections: 3 × 2 × bs × s × d²
    #   Attention QK^T:    2 × bs × h × s² × head_dim  = 2 × bs × s² × d
    #   Attention × V:     2 × bs × h × s² × head_dim  = 2 × bs × s² × d
    #   Output proj:       2 × bs × s × d²
    #   FFN (up + down):   4 × bs × s × d × ffn_dim
    per_layer_total = (
        8 * batch_size * s * d * d
        + 4 * batch_size * s * s * d
        + 4 * batch_size * s * d * ffn_dim
    )
    # Divide by SP size — work is evenly split across SP ranks.
    per_layer_per_rank = per_layer_total // max(sp_size, 1)
    return per_layer_per_rank * num_layers


def compute_mfu_stats(
    *,
    diffusion_time_s: float,
    dit_compute_steps: int,
    num_dit_steps: int,
    model_cfg: dict[str, Any],
    sp_size: int,
) -> dict[str, Any]:
    """Compute achieved TFLOPs/s and MFU for the DiT denoising loop."""
    if diffusion_time_s <= 0 or dit_compute_steps <= 0:
        return {"achieved_tflops": 0.0, "mfu_pct": 0.0, "peak_tflops": _peak_tflops()}

    per_rank_per_step = _dit_forward_flops(
        dim=int(model_cfg.get("dim", 5120)),
        num_heads=int(model_cfg.get("num_heads", 40)),
        head_dim=model_cfg.get("head_dim"),
        ffn_dim=int(model_cfg.get("ffn_dim", 13824)),
        seq_len=int(model_cfg.get("seq_len", model_cfg.get("frame_seqlen", 880))),
        num_layers=int(model_cfg.get("num_layers", 40)),
        sp_size=sp_size,
        batch_size=int(model_cfg.get("batch_size", 1)),
    )
    # Total FLOPs actually executed for this call (some steps may be cache-skipped
    # by DYNAMIC_CACHE_SCHEDULE — use dit_compute_steps, not num_dit_steps).
    total_flops = per_rank_per_step * dit_compute_steps
    achieved_tflops = (total_flops / 1e12) / diffusion_time_s
    peak = _peak_tflops()
    return {
        "per_rank_flops_per_step": per_rank_per_step,
        "dit_compute_steps": dit_compute_steps,
        "num_dit_steps_scheduled": num_dit_steps,
        "total_flops_per_rank": total_flops,
        "achieved_tflops": achieved_tflops,
        "peak_tflops": peak,
        "mfu_pct": 100.0 * achieved_tflops / peak if peak > 0 else 0.0,
    }


# ── Dump ─────────────────────────────────────────────────────────────────────


def _log_path() -> str:
    return os.environ.get("PROFILE_LOG_FILE", "/tmp/dreamzero_profile.jsonl")


def finalize_and_dump(
    *,
    phases_s: dict[str, float],
    total_s: float,
    dit_compute_steps: int,
    num_dit_steps: int,
    model_cfg: dict[str, Any],
    sp_size: int,
    cfg_size: int,
    rank: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit one JSON profile record. Only rank 0 writes; other ranks no-op.

    `phases_s` should contain the per-phase seconds (text_encoder_s, image_encoder_s,
    vae_s, kv_creation_s, diffusion_s, scheduler_s, etc.).
    """
    if not enabled() or rank != 0:
        return

    a2a_ms = _sum_events_ms(_comm_a2a_events)
    allgather_ms = _sum_events_ms(_comm_allgather_events)

    comm = {
        "a2a_count": len(_comm_a2a_events),
        "a2a_total_s": a2a_ms / 1000.0,
        "allgather_count": len(_comm_allgather_events),
        "allgather_total_s": allgather_ms / 1000.0,
        "total_comm_s": (a2a_ms + allgather_ms) / 1000.0,
    }

    mfu = compute_mfu_stats(
        diffusion_time_s=phases_s.get("diffusion_s", 0.0),
        dit_compute_steps=dit_compute_steps,
        num_dit_steps=num_dit_steps,
        model_cfg=model_cfg,
        sp_size=sp_size,
    )

    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_s": total_s,
        "phases_s": phases_s,
        "communication": comm,
        "mfu": mfu,
        "sp_size": sp_size,
        "cfg_size": cfg_size,
        "model_cfg": model_cfg,
    }
    if extra:
        record["extra"] = extra

    path = _log_path()
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:  # pragma: no cover
        print(f"[perf_profile] failed to write {path}: {e}")


# ── Chrome/Perfetto trace (one-shot) ─────────────────────────────────────────


_trace_state: dict[str, Any] = {"done": False, "profiler": None}


def maybe_start_trace(rank: int = 0) -> None:
    """Called at start of one post-warmup inference call. Starts torch.profiler.

    Only global rank 0 records. Other ranks run unprofiled to keep overhead
    local to one process.

    Controlled by PROFILE_TRACE=true and PROFILE_TRACE_AFTER_CALL (skip this many
    warmup calls before capturing; default 5).
    """
    if rank != 0:
        return
    if not trace_enabled() or _trace_state["done"]:
        return
    skip = int(os.environ.get("PROFILE_TRACE_AFTER_CALL", "5"))
    cur = _trace_state.get("call_count", 0) + 1
    _trace_state["call_count"] = cur
    if cur <= skip:
        return
    if _trace_state["profiler"] is not None:
        return  # already recording

    try:
        from torch.profiler import profile, ProfilerActivity
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
            with_flops=True,
            with_modules=True,
        )
        prof.__enter__()
        _trace_state["profiler"] = prof
    except Exception as e:  # pragma: no cover
        print(f"[perf_profile] failed to start trace: {e}")


def maybe_stop_trace(rank: int) -> None:
    """Close trace capture and export chrome trace JSON (rank 0 only)."""
    prof = _trace_state.get("profiler")
    if prof is None or _trace_state["done"]:
        return
    try:
        prof.__exit__(None, None, None)
        if rank == 0:
            default_dir = "/mnt/localssd/dreamzero_traces"
            path = os.environ.get(
                "PROFILE_TRACE_FILE",
                f"{default_dir}/trace_{int(time.time())}.json",
            )
            # Kineto writes `{path}.tmp` then atomic-renames to `{path}`.
            # Both the target AND tmp parent must exist before export.
            parent = os.path.dirname(os.path.abspath(path)) or "."
            os.makedirs(parent, exist_ok=True)
            prof.export_chrome_trace(path)
            print(f"[perf_profile] chrome trace written: {path}")
    finally:
        _trace_state["done"] = True
        _trace_state["profiler"] = None
