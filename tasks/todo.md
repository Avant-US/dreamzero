# Benchmarking Plan: Paper Table 1 Step-by-Step on 2x H200

Goal: Measure each optimization step in isolation to compare against the paper's
cumulative speedups and identify where our implementation diverges.

Hardware: 2x H200 (141GB HBM3e each, 4.8 TB/s) via Docker
Paper reference: Table 1, 5.7s baseline, cumulative to 9.6x on H100

## Docker Setup

Start the container with both GPUs and /mnt/r mounted:
```bash
docker run --gpus all -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 5000:5000 -v ~/dreamzero:/workspace -v /mnt/r:/mnt/r -w /workspace --name dreamzero-bench dreamzero
```

One-time setup inside the container:
```bash
pip install --no-deps -e .
```

Download checkpoints (writes to /mnt/r via symlink):
```bash
HF_HOME=/mnt/r/huggingface_cache python -c "from huggingface_hub import snapshot_download; snapshot_download('GEAR-Dreams/DreamZero-DROID', local_dir='./huggingface_checkpoints')"
```

**Note:** All benchmark commands below require `HF_HOME=/mnt/r/huggingface_cache` prefix
to avoid downloading to the full root disk.

## Protocol

- Run 20 inference steps per config, discard first 5 (warmup), average the rest
- Record: total time, diffusion time, number of effective DiT steps
- Keep the same prompt/image across all runs for consistency
- Between steps: stop the server, `unset` changed env vars, restart
- Log GPU utilization + memory via `nvidia-smi` in a second shell:
  ```bash
  docker exec dreamzero-bench nvidia-smi
  ```

Test client (run from a second shell into the same container):
```bash
docker exec -it dreamzero-bench python test_client_AR.py --port 5000
```

---

## Step 1: CFG Parallelism Only

**What it measures:** 2-GPU CFG parallel vs paper's 1.9x claim
**Paper:** 5.7s -> 3.0s (1.9x) on H100

```bash
DYNAMIC_CACHE_SCHEDULE=false \
DISABLE_TORCH_COMPILE=true \
NUM_DIT_STEPS=16 \
ATTENTION_BACKEND=FA2 \
CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py \
  --port 5000 --enable-dit-cache --model-path ./huggingface_checkpoints
```

**Notes:**
- `DYNAMIC_CACHE_SCHEDULE=false` disables cosine-similarity step skipping
- `NUM_DIT_STEPS=16` runs all 16 diffusion steps (no fixed mask skipping)
- `--enable-dit-cache` is needed for KV cache reuse across AR steps (not the same as DiT caching/step skipping)
- FA2 backend to isolate from TE effects
- `DISABLE_TORCH_COMPILE=true` disables encoder/VAE/scheduler compilation for clean baseline

**Expected:** Roughly paper_baseline / 1.9 scaled by H200/H100 bandwidth ratio

- [x] Record: total ~2.88s, diffusion ~2.58s, effective steps: 16
- [x] Diffusion varies 2.32-2.82s in a repeating 4-call cycle (KV cache grows then resets)
- [x] Paper H100: 3.0s -- our 2.88s is slightly faster (H200 bandwidth advantage)

---

## Step 2: CFG Parallelism + DiT Caching

**What it measures:** Dynamic cache schedule contribution on top of CFG
**Paper:** 3.0s -> 1.04s (~2.9x from caching alone) on H100

```bash
DYNAMIC_CACHE_SCHEDULE=true \
DISABLE_TORCH_COMPILE=true \
NUM_DIT_STEPS=16 \
ATTENTION_BACKEND=FA2 \
CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py \
  --port 5000 --enable-dit-cache --model-path ./huggingface_checkpoints
```

**Notes:**
- Only change from Step 1: `DYNAMIC_CACHE_SCHEDULE=true`
- `DISABLE_TORCH_COMPILE=true` still set to keep compile off
- Should reduce effective steps from 16 to ~4-5 via cosine similarity
- The diffusion time should drop ~3x while non-diffusion time stays constant

- [x] Record: total ~1.03s, diffusion ~0.70s, effective steps: 4-6
- [x] Caching reduces diffusion from ~2.58s to ~0.70s (3.7x on diffusion alone)
- [x] Total 2.88s -> 1.03s = 2.8x speedup from caching
- [x] Paper H100: 1.04s -- our 1.03s matches almost exactly

---

## Step 3: CFG + DiT Caching + Torch Compile (partial)

**What it measures:** torch.compile on encoders/VAE/scheduler (DiT still uncompiled)
**Paper:** 1.04s -> 0.64s (1.6x from compile) on H100

```bash
DYNAMIC_CACHE_SCHEDULE=true \
DISABLE_TORCH_COMPILE=false \
NUM_DIT_STEPS=16 \
ATTENTION_BACKEND=FA2 \
CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py \
  --port 5000 --enable-dit-cache --model-path ./huggingface_checkpoints
```

**Notes:**
- Only change from Step 2: `DISABLE_TORCH_COMPILE=false` (enables encoder/VAE/scheduler compile)
- This is partial compile only -- DiT _forward_blocks still uncompiled (dynamic KV + complex RoPE)
- The paper's 1.6x includes DiT compile + CUDA graphs, so we expect less than 1.6x here
- Compare encoder/VAE/scheduler time breakdown vs Step 2 to see compile impact

- [x] Record: total ~0.95s, diffusion ~0.68s, effective steps: 4-5
- [x] First 2 calls are warmup (66s, 16.8s) due to torch.compile tracing
- [x] Compile shaves ~0.08s off total vs Step 2 (1.03s -> 0.95s = ~8% faster)
- [x] Breakdown: Text Enc 0.03->0.01s, Image Enc 0.30->0.18s, VAE 0.07->0.05s
- [x] Diffusion time unchanged (~0.68 vs ~0.70) -- confirms compile only helps encoders/VAE
- [x] Paper H100: 0.64s -- our 0.95s still 0.31s behind (missing DiT compile + CUDA graphs)

---

## Step 4: CFG + DiT Caching + Compile + TE Attention

**What it measures:** Transformer Engine kernels vs FA2
**Paper:** 0.64s -> 0.59s (~8% from TE) on H100

```bash
DYNAMIC_CACHE_SCHEDULE=true \
DISABLE_TORCH_COMPILE=false \
NUM_DIT_STEPS=16 \
ATTENTION_BACKEND=TE \
CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py \
  --port 5000 --enable-dit-cache --model-path ./huggingface_checkpoints
```

**Notes:**
- Only change from Step 3: `ATTENTION_BACKEND=TE`
- On H200, TE is ~10% faster than FA2 (confirmed in our Table 10b)
- On H100, TE is actually slower than FA2 (paper benchmarked TE on GB200)
- This should match our existing ~0.87s measurement

- [ ] Record: total ___s, diffusion ___s, effective steps: ___
- [ ] Confirm matches existing 0.87s data point

---

## Step 5: CFG + DiT Caching + Compile + TE + TRT FP8

**What it measures:** TensorRT FP8 quantization
**Paper:** GB200 only: 0.39s -> 0.34s (16.6x)

```bash
DYNAMIC_CACHE_SCHEDULE=true \
NUM_DIT_STEPS=16 \
ATTENTION_BACKEND=TE \
LOAD_TRT_ENGINE=./huggingface_checkpoints/tensorrt/wan/WanModel_nvfp4.trt \
CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py \
  --port 5000 --enable-dit-cache --model-path ./huggingface_checkpoints
```

**Notes:**
- `LOAD_TRT_ENGINE` auto-sets `ENABLE_TENSORRT=true` behavior
- torch.compile on encoders/VAE is disabled when TRT is active
- This should match our existing ~0.58s measurement

- [ ] Record: total ___s, diffusion ___s, effective steps: ___
- [ ] Confirm matches existing 0.58s data point

---

## Key Gaps to Investigate

1. **torch.compile on DiT _forward_blocks** -- the paper's biggest single-step gain
   (1.04s -> 0.64s = 1.6x). We can't reproduce this yet. Need to fix:
   - Dynamic KV cache shapes -> pre-allocate fixed-size cache
   - `view_as_complex` RoPE -> real-valued sin/cos RoPE
   - Shape-dependent branching -> masking

2. **CUDA Graphs** -- the paper bundles this with torch.compile. Even if we can't
   compile the DiT, we might capture CUDA graphs for the compiled portions.

3. **Encoder/VAE compile isolation** -- `DISABLE_TORCH_COMPILE` toggle now available
   to measure the compile speedup independently from CFG/caching.

## Results Summary

| Step | Config | Paper H100 | Our 2x H200 | Delta |
|------|--------|-----------|-------------|-------|
| 1. CFG Only | 2 GPU, no cache, no compile, FA2 | 3.0s | ~2.88s | -4% (H200 faster) |
| 2. + DiT Caching | + DYNAMIC_CACHE_SCHEDULE | 1.04s | ~1.03s | matches paper |
| 3. + Torch Compile | + DISABLE_TORCH_COMPILE=false | 0.64s | ~0.95s | +48% (missing DiT compile) |
| 4. + TE Attention | + ATTENTION_BACKEND=TE | 0.59s | ~0.87s | |
| 5. + TRT FP8 | + LOAD_TRT_ENGINE | 0.34s* | ~0.58s | |

*GB200 only
