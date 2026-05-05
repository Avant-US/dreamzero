"""Test script for action-only optimization comparison.
Sends fixed observations and saves actions + timing for comparison."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
from eval_utils.policy_client import WebsocketClientPolicy
import numpy as np
import argparse

def run_test(port, output_file, num_chunks=10, session_id="action_only_test"):
    client = WebsocketClientPolicy(host="localhost", port=port)
    all_actions = []
    times = []
    for i in range(num_chunks):
        np.random.seed(42 + i)
        n = 1 if i == 0 else 4
        obs = {
            "observation/exterior_image_0_left": np.random.randint(0, 255, (n, 180, 320, 3), dtype=np.uint8),
            "observation/exterior_image_1_left": np.random.randint(0, 255, (n, 180, 320, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.random.randint(0, 255, (n, 180, 320, 3), dtype=np.uint8),
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
            "prompt": "Move the object forward and place it on the table in front of you carefully",
            "session_id": session_id,
        }
        t0 = time.time()
        result = client.infer(obs)
        dt = time.time() - t0
        times.append(dt)
        act = result if isinstance(result, np.ndarray) else np.array(result.get("action", result))
        all_actions.append(act.tolist())
        print(f"Chunk {i:2d}: {dt:.3f}s  range=[{act.min():.4f}, {act.max():.4f}]")

    with open(output_file, "w") as f:
        json.dump({"actions": all_actions, "times": times}, f)

    print(f"\nChunks 4-{num_chunks-1} (steady): {np.mean(times[4:]):.3f}s  (min={min(times[4:]):.3f}, max={max(times[4:]):.3f})")
    return all_actions, times

def compare(file_a, file_b):
    with open(file_a) as f: a = json.load(f)
    with open(file_b) as f: b = json.load(f)
    acts_a, acts_b = a["actions"], b["actions"]
    n = min(len(acts_a), len(acts_b))
    print(f"\nBit-exact comparison ({file_a} vs {file_b}):")
    for i in range(n):
        diff = np.abs(np.array(acts_a[i]) - np.array(acts_b[i])).max()
        print(f"  Chunk {i}: max_err={diff:.10f}")
    all_diff = np.abs(
        np.concatenate([np.array(acts_a[i]).flatten() for i in range(n)]) -
        np.concatenate([np.array(acts_b[i]).flatten() for i in range(n)])
    )
    if all_diff.max() == 0:
        print("  RESULT: BIT-EXACT!")
    else:
        print(f"  OVERALL: max_err={all_diff.max():.10f}")

    times_a = a.get("times", [0]*n)
    times_b = b.get("times", [0]*n)
    if len(times_a) > 4 and len(times_b) > 4:
        print(f"\nSpeed: A={min(times_a[4:]):.3f}s  B={min(times_b[4:]):.3f}s  diff={min(times_b[4:])-min(times_a[4:]):.3f}s")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5001)
    p.add_argument("--output", required=True)
    p.add_argument("--compare", nargs=2, metavar=("FILE_A", "FILE_B"))
    p.add_argument("--num-chunks", type=int, default=10)
    p.add_argument("--session-id", default="action_only_test")
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run_test(args.port, args.output, args.num_chunks, args.session_id)
