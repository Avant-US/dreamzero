"""Benchmark FastVideo's CausalWan inference for comparison with DreamZero."""
import time
import sys
sys.path.insert(0, "/mnt/localssd/FastVideo")

from fastvideo import VideoGenerator, SamplingParam

def main():
    print("Loading FastVideo CausalWan model...")
    generator = VideoGenerator.from_pretrained(
        "FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers",
        num_gpus=1,
        dit_precision="bf16",
        pin_cpu_memory=True,
    )
    
    sampling = SamplingParam.from_pretrained("FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers")
    sampling.num_frames = 9  # Small for quick benchmark
    sampling.width = 640
    sampling.height = 352
    sampling.seed = 42
    
    prompt = "A robot arm picks up an object from a table"
    
    # Warmup
    print("Warmup...")
    for _ in range(2):
        _ = generator.generate_video(prompt, save_video=False, sampling_param=sampling)
    
    # Benchmark
    print("Benchmarking...")
    times = []
    for i in range(5):
        t0 = time.perf_counter()
        _ = generator.generate_video(prompt, save_video=False, sampling_param=sampling)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"  Run {i}: {dt:.3f}s")
    
    import statistics
    print(f"\nFastVideo Results: mean={statistics.mean(times):.3f}s, "
          f"median={statistics.median(times):.3f}s, "
          f"min={min(times):.3f}s, max={max(times):.3f}s")

if __name__ == "__main__":
    main()
