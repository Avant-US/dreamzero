#!/usr/bin/env python3
"""Test client for AR_droid policy server using roboarena interface.

Sends real video frames from debug_image/ directory instead of zero dummy images.

Frame schedule (matching debug_inference.py):
  - Step 0 (initial): send frame [0]             (1 frame, H W 3)
  - Step 1: send frames [0, 7, 15, 23]           (4 frames, 4 H W 3)
  - Step 2: send frames [24, 31, 39, 47]         (4 frames)
  - Step 3: send frames [48, 55, 63, 71]         (4 frames)
  - ...

Expected server configuration:
    - image_resolution: (180, 320)
    - n_external_cameras: 2
    - needs_wrist_camera: True
    - action_space: "joint_position"

Usage:
    # Start server with roboarena interface:
    torchrun --nproc_per_node=8 socket_test_optimized_AR.py --port 8000

    # Run this test:
    python test_client_AR.py --host <server_host> --port 8000

    # Use zero images instead of real video (old behavior):
    python test_client_AR.py --host <server_host> --port 8000 --use-zero-images
"""

import argparse
import json
import logging
import os
import statistics
import time
import uuid

import cv2
import numpy as np

import eval_utils.policy_server as policy_server
from eval_utils.policy_client import WebsocketClientPolicy

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "debug_image")

# roboarena key -> video filename
CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}

# Frame schedule constants (matching debug_inference.py)
RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24


def load_all_frames(video_path: str) -> np.ndarray:
    """Load all frames from a video file. Returns (N, H, W, 3) uint8 array (RGB)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return np.stack(frames, axis=0)


def load_camera_frames() -> dict[str, np.ndarray]:
    """Load all video frames for each camera from the debug_image/ directory.

    Returns:
        Dict mapping roboarena camera keys to (N, H, W, 3) uint8 arrays.
    """
    camera_frames: dict[str, np.ndarray] = {}
    for cam_key, fname in CAMERA_FILES.items():
        path = os.path.join(VIDEO_DIR, fname)
        camera_frames[cam_key] = load_all_frames(path)
        logging.info(f"Loaded {cam_key}: {camera_frames[cam_key].shape}")
    return camera_frames


def build_frame_schedule(total_frames: int, num_chunks: int) -> list[list[int]]:
    """Build the frame index schedule for multi-frame chunks.

    Returns a list of frame-index lists. Each inner list has 4 indices.
    """
    chunks: list[list[int]] = []
    current_frame = 23  # first anchor frame
    for _ in range(num_chunks):
        indices = [max(current_frame + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            logging.info(
                f"Frame {indices[-1]} >= {total_frames}, stopping at {len(chunks)} chunks"
            )
            break
        chunks.append(indices)
        current_frame += ACTION_HORIZON
    return chunks


def _make_obs_from_video(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    prompt: str,
    session_id: str,
) -> dict:
    """Build an observation dict from real video frames.

    For 1 frame: each image key is (H, W, 3).
    For 4 frames: each image key is (4, H, W, 3).
    """
    obs: dict = {}
    for cam_key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]  # (T, H, W, 3)
        if len(frame_indices) == 1:
            selected = selected[0]  # (H, W, 3)
        obs[cam_key] = selected

    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs


def _make_zero_observation(
    server_config: policy_server.PolicyServerConfig,
    prompt: str = "pick up the object",
    session_id: str | None = None,
) -> dict:
    """Create a dummy observation matching AR_droid expectations.
    
    AR_droid expects:
        - 2 external cameras (exterior_image_0_left, exterior_image_1_left)
        - 1 wrist camera (wrist_image_left)
        - Image resolution: 180x320 (H x W)
        - joint_position: 7 DoF
        - gripper_position: 1 DoF
    """
    obs = {}
    
    # Determine image resolution
    if server_config.image_resolution is not None:
        h, w = server_config.image_resolution
    else:
        # Default for AR_droid
        h, w = 180, 320
    
    # External cameras (0-indexed in roboarena)
    for i in range(server_config.n_external_cameras):
        obs[f"observation/exterior_image_{i}_left"] = np.zeros((h, w, 3), dtype=np.uint8)
        if server_config.needs_stereo_camera:
            obs[f"observation/exterior_image_{i}_right"] = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Wrist camera
    if server_config.needs_wrist_camera:
        obs["observation/wrist_image_left"] = np.zeros((h, w, 3), dtype=np.uint8)
        if server_config.needs_stereo_camera:
            obs["observation/wrist_image_right"] = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Session ID - should be passed in to ensure consistency within a session
    if server_config.needs_session_id:
        import uuid
        # Generate unique session ID if not provided
        obs["session_id"] = session_id if session_id else str(uuid.uuid4())
    
    # State observations (AR_droid: 7 DoF arm + 1 gripper)
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    
    # Language prompt
    obs["prompt"] = prompt
    
    return obs


def _summarize_timings(label: str, values: list[float]) -> dict:
    """Return a stats dict for a list of per-call elapsed times (seconds)."""
    if not values:
        return {"label": label, "count": 0}
    sorted_v = sorted(values)
    n = len(sorted_v)
    def _pct(p: float) -> float:
        if n == 1:
            return sorted_v[0]
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        return sorted_v[f] + (sorted_v[c] - sorted_v[f]) * (k - f)
    return {
        "label": label,
        "count": n,
        "mean_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "stdev_s": statistics.pstdev(values) if n > 1 else 0.0,
        "min_s": min(values),
        "max_s": max(values),
        "p50_s": _pct(0.50),
        "p90_s": _pct(0.90),
        "p95_s": _pct(0.95),
        "p99_s": _pct(0.99),
        "raw_s": values,
    }


def _load_profile_records(path: str | None) -> list[dict]:
    """Read per-call server-side profile records (one JSON per line)."""
    if not path or not os.path.exists(path):
        return []
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return records


def _summarize_profile_records(records: list[dict], warmup: int) -> dict:
    """Roll per-call profile records (post-warmup) into summary means + MFU."""
    if not records:
        return {"count": 0}
    steady = records[warmup:] if warmup < len(records) else records[-1:]
    if not steady:
        return {"count": 0}

    def _mean(key_path: list):
        vals = []
        for r in steady:
            x = r
            for k in key_path:
                if not isinstance(x, dict) or k not in x:
                    x = None
                    break
                x = x[k]
            if isinstance(x, (int, float)):
                vals.append(float(x))
        return statistics.fmean(vals) if vals else None

    phase_keys = [
        "text_encoder_s", "image_encoder_s", "vae_s",
        "kv_creation_s", "diffusion_s", "scheduler_s",
    ]
    phases_mean = {k: _mean(["phases_s", k]) for k in phase_keys}
    comm_mean = {
        "a2a_total_s": _mean(["communication", "a2a_total_s"]),
        "a2a_count": _mean(["communication", "a2a_count"]),
        "allgather_total_s": _mean(["communication", "allgather_total_s"]),
        "allgather_count": _mean(["communication", "allgather_count"]),
        "total_comm_s": _mean(["communication", "total_comm_s"]),
    }
    mfu_mean = {
        "achieved_tflops": _mean(["mfu", "achieved_tflops"]),
        "peak_tflops": _mean(["mfu", "peak_tflops"]),
        "mfu_pct": _mean(["mfu", "mfu_pct"]),
        "dit_compute_steps": _mean(["mfu", "dit_compute_steps"]),
        "total_flops_per_rank": _mean(["mfu", "total_flops_per_rank"]),
    }
    # % of total time spent in comm vs compute (steady-state mean).
    total_mean = _mean(["total_s"]) or 0.0
    comm_pct = None
    if total_mean and comm_mean.get("total_comm_s") is not None:
        comm_pct = 100.0 * comm_mean["total_comm_s"] / total_mean
    return {
        "count": len(steady),
        "phases_mean_s": phases_mean,
        "communication_mean": comm_mean,
        "mfu_mean": mfu_mean,
        "comm_pct_of_total": comm_pct,
        "total_mean_s": total_mean,
    }


def test_ar_droid_policy_server(
    host: str = "localhost",
    port: int = 8000,
    num_chunks: int = 15,
    prompt: str = "Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
    use_zero_images: bool = False,
    report_path: str | None = None,
    skip_reset: bool = False,
    warmup_chunks: int = 0,
    profile_log: str | None = None,
):
    """Test the AR_droid policy server with roboarena interface.

    When use_zero_images is False (default), loads real video frames from
    debug_image/ and follows the frame schedule from debug_inference.py.
    """
    logging.info(f"Connecting to AR_droid server at {host}:{port}...")
    
    client = WebsocketClientPolicy(host=host, port=port)
    
    # Validate server metadata
    metadata = client.get_server_metadata()
    logging.info(f"Server metadata: {metadata}")
    assert isinstance(metadata, dict), "Metadata should be a dict"
    
    try:
        server_config = policy_server.PolicyServerConfig(**metadata)
    except Exception as e:
        logging.error(f"Error parsing metadata: {e}")
        raise e
    
    # Validate expected AR_droid configuration
    logging.info(f"Server config: {server_config}")
    assert server_config.n_external_cameras == 2, f"Expected 2 external cameras, got {server_config.n_external_cameras}"
    assert server_config.needs_wrist_camera, "Expected wrist camera to be enabled"
    assert server_config.action_space == "joint_position", f"Expected joint_position action space, got {server_config.action_space}"
    
    logging.info("Server configuration validated for AR_droid")
    
    # Generate unique session ID for this test run
    import uuid
    session_id = str(uuid.uuid4())
    logging.info(f"Session ID: {session_id}")

    # Per-chunk timings — index 0 is the initial 1-frame call.
    all_timings: list[float] = []

    # ── Zero-image fallback mode ──────────────────────────────────────
    if use_zero_images:
        logging.info("Using ZERO dummy images (legacy mode)")
        for i in range(num_chunks):
            obs = _make_zero_observation(server_config, prompt=prompt, session_id=session_id)
            logging.info(f"Inference {i + 1}/{num_chunks}: prompt='{prompt}'")
            t0 = time.time()
            actions = client.infer(obs)
            dt = time.time() - t0
            all_timings.append(dt)
            _log_action(actions, dt)
    else:
        # ── Real video frame mode ─────────────────────────────────────
        logging.info("Loading real video frames from debug_image/ directory")
        camera_frames = load_camera_frames()

        total_frames = min(v.shape[0] for v in camera_frames.values())
        logging.info(f"Total frames available: {total_frames}")

        chunks = build_frame_schedule(total_frames, num_chunks)

        logging.info("Frame schedule:")
        logging.info("  Initial: [0]")
        for i, indices in enumerate(chunks):
            logging.info(f"  Chunk {i}: {indices}")

        # Step 0: initial single frame
        logging.info("=== Initial: frame [0] ===")
        obs = _make_obs_from_video(camera_frames, [0], prompt, session_id)
        t0 = time.time()
        actions = client.infer(obs)
        dt = time.time() - t0
        all_timings.append(dt)
        _log_action(actions, dt)

        # Subsequent chunks: send 4 frames at a time
        for chunk_idx, frame_indices in enumerate(chunks):
            logging.info(f"=== Chunk {chunk_idx}: frames {frame_indices} ===")
            obs = _make_obs_from_video(camera_frames, frame_indices, prompt, session_id)
            t0 = time.time()
            actions = client.infer(obs)
            dt = time.time() - t0
            all_timings.append(dt)
            _log_action(actions, dt)

    # Reset triggers video save on the server — skip when benchmarking.
    if skip_reset:
        logging.info("Skipping reset (no server-side video save).")
    else:
        logging.info("Sending reset to save video...")
        client.reset({})

    # ── Stats ─────────────────────────────────────────────────────────
    all_stats = _summarize_timings("all", all_timings)
    warmup_count = max(0, min(warmup_chunks, len(all_timings) - 1))
    steady = all_timings[warmup_count:]
    steady_stats = _summarize_timings(f"steady (after {warmup_count} warmup)", steady)

    def _fmt(s: dict) -> str:
        if s["count"] == 0:
            return f"  {s['label']}: (no samples)"
        return (
            f"  {s['label']}: n={s['count']:<3d} "
            f"mean={s['mean_s']:.3f}s  med={s['median_s']:.3f}s  "
            f"p95={s['p95_s']:.3f}s  min={s['min_s']:.3f}s  max={s['max_s']:.3f}s"
        )

    logging.info("─── Performance summary ───")
    logging.info(_fmt(all_stats))
    logging.info(_fmt(steady_stats))

    # Per-call server-side profile records (MFU, phase breakdowns, comm timings).
    profile_records = _load_profile_records(profile_log)
    profile_summary = _summarize_profile_records(profile_records, warmup_count)

    if profile_summary.get("count"):
        mfu = profile_summary.get("mfu_mean") or {}
        phases = profile_summary.get("phases_mean_s") or {}
        comm = profile_summary.get("communication_mean") or {}
        logging.info("─── Server-side profile (steady-state mean) ───")
        if mfu.get("achieved_tflops") is not None:
            logging.info(
                f"  MFU: {mfu.get('mfu_pct', 0):.1f}%  "
                f"({mfu.get('achieved_tflops', 0):.1f} / {mfu.get('peak_tflops', 0):.0f} TFLOPs)"
            )
        logging.info(
            f"  Phases (s): text={phases.get('text_encoder_s') or 0:.3f}  "
            f"img={phases.get('image_encoder_s') or 0:.3f}  "
            f"vae={phases.get('vae_s') or 0:.3f}  "
            f"kv={phases.get('kv_creation_s') or 0:.3f}  "
            f"diffusion={phases.get('diffusion_s') or 0:.3f}  "
            f"sched={phases.get('scheduler_s') or 0:.3f}"
        )
        if comm.get("total_comm_s") is not None:
            logging.info(
                f"  Comm: total={comm['total_comm_s']:.3f}s  "
                f"a2a={comm.get('a2a_total_s') or 0:.3f}s (n={int(comm.get('a2a_count') or 0)})  "
                f"allgather={comm.get('allgather_total_s') or 0:.3f}s "
                f"(n={int(comm.get('allgather_count') or 0)})  "
                f"→ {profile_summary.get('comm_pct_of_total') or 0:.1f}% of total"
            )
    elif profile_log:
        logging.info(
            f"No profile records found at {profile_log} — "
            "did you start the server with PROFILE_INFERENCE=true?"
        )

    if report_path:
        report = {
            "host": host,
            "port": port,
            "prompt": prompt,
            "num_chunks_requested": num_chunks,
            "num_calls_measured": len(all_timings),
            "warmup_chunks_excluded": warmup_count,
            "use_zero_images": use_zero_images,
            "session_id": session_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "all": all_stats,
            "steady_state": steady_stats,
            # Attention/env knobs the server was started with (client view — informational only)
            "env": {
                k: os.environ[k]
                for k in (
                    "ATTENTION_BACKEND", "COMPILE_DIT", "CUDA_GRAPH_DIT",
                    "DYNAMIC_CACHE_SCHEDULE", "NUM_DIT_STEPS", "FP8_INFERENCE",
                    "SP_SIZE", "NUM_GPUS",
                )
                if k in os.environ
            },
            "profile_summary": profile_summary,
            "profile_calls": profile_records,
        }
        os.makedirs(os.path.dirname(os.path.abspath(report_path)) or ".", exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logging.info(f"Report written to: {report_path}")

    logging.info("Done.")


def _log_action(actions: np.ndarray, dt: float) -> None:
    """Pretty-print action shape, range, and timing."""
    assert isinstance(actions, np.ndarray), f"Expected numpy array, got {type(actions)}"
    assert actions.ndim == 2, f"Expected 2D array, got shape {actions.shape}"
    assert actions.shape[-1] == 8, (
        f"Expected 8 action dims (7 joints + 1 gripper), got {actions.shape[-1]}"
    )
    logging.info(
        f"  Action shape: {actions.shape}, "
        f"range: [{actions.min():.4f}, {actions.max():.4f}], "
        f"time: {dt:.2f}s"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test AR_droid policy server with real video frames from debug_image/"
    )
    parser.add_argument("--host", default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=15,
        help="Number of 4-frame chunks to send after the initial frame (default: 15)",
    )
    parser.add_argument(
        "--prompt",
        default="Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
        help="Language prompt for the policy",
    )
    parser.add_argument(
        "--use-zero-images",
        action="store_true",
        help="Use zero dummy images instead of real video frames (legacy mode)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path to write a JSON performance report (per-chunk timings + stats)",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Skip the trailing reset() call (which triggers server-side video save)",
    )
    parser.add_argument(
        "--warmup-chunks",
        type=int,
        default=0,
        help="Exclude the first N chunks from the steady-state stats (default: 0)",
    )
    parser.add_argument(
        "--profile-log",
        default=None,
        help="Path to server-side per-call profile JSONL (written when server was "
             "started with PROFILE_INFERENCE=true). Rolled into the --report output.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    test_ar_droid_policy_server(
        host=args.host,
        port=args.port,
        num_chunks=args.num_chunks,
        prompt=args.prompt,
        use_zero_images=args.use_zero_images,
        report_path=args.report,
        skip_reset=args.no_reset,
        warmup_chunks=args.warmup_chunks,
        profile_log=args.profile_log,
    )


if __name__ == "__main__":
    main()
