# NVIDIA DreamZero: World Action Models Are Zero-Shot Policies
A research project from [NVIDIA GEAR Lab](https://research.nvidia.com/labs/gear/).

[![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?style=flat&logo=nvidia&logoColor=white)](https://www.nvidia.com) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) [![arXiv](https://img.shields.io/badge/arXiv-2602.15922-b31b1b.svg)](https://arxiv.org/abs/2602.15922)

[[Project Page](https://dreamzero0.github.io/)] [[Paper](https://arxiv.org/abs/2602.15922)]

DreamZero is a World Action Model that jointly predicts actions and videos, achieving strong zero-shot performance on unseen tasks. This release package contains everything needed to load a pretrained DreamZero model and run distributed inference via a WebSocket server.

## News

- **02/27:** DreamZero is **#1 on both [MolmoSpaces]([https://huggingface.co/spaces/ai2-adapt/MolmoSpaces](https://molmospaces.allen.ai/leaderboard)) and [RoboArena]([https://robo-arena.github.io/](https://robo-arena.github.io/leaderboard))**! DreamZero-DROID is trained *from scratch* using only the DROID dataset — no pretraining on large-scale robot data, unlike competing VLAs. This demonstrates the strength of video-model backbones for generalist robot policies (VAMs/WAMs).
- **02/27:** Released **DreamZero-AgiBot checkpoint** and **post-training code** for efficient few-shot adaptation. Post-train on just ~30 minutes of play data for your specific robot, and see the robot do basic language following and pick-and-place (see YAM experiments in our paper for more detail).
- **02/20:** Released the **full training codebase, preprocessed dataset, and guide for new embodiments** to replicate the DreamZero-DROID checkpoint and train on your own robot. See [Adding a New Embodiment to DreamZero](docs/DATASET_TO_GEAR_AND_TRAIN.md) for a step-by-step walkthrough.

## Features

**Available Now**
- Pretrained DreamZero-DROID model checkpoint [[Huggingface](https://huggingface.co/GEAR-Dreams/DreamZero-DROID)]
- Pretrained DreamZero-AgiBot checkpoint (for post-training on new embodiments) [[Huggingface](https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot)]
- Distributed WebSocket inference server (GB200, H100)
- DiT caching for optimized inference (~0.6s on GB200, ~3s on H100)
- DROID simulation evaluation support
- [RoboArena](https://robo-arena.github.io/) integration (DROID real)
- Video generation and saving (MP4)
- LoRA and full fine-tuning training scripts
- Training on new embodiments (AgiBot, YAM) — see [guide](docs/DATASET_TO_GEAR_AND_TRAIN.md)

**Coming Soon**
- [PolaRiS](https://polaris-evals.github.io/) simulation environment support
- [Genie 3.0](https://arxiv.org/abs/2601.02078) sim environment support for DreamZero-AgiBot

## Testing Out DreamZero in Simulation with API
We provide an inference script that directly evaluates a hosted DreamZero-DROID policy on [`sim_evals`](https://github.com/arhanjain/sim-evals). To test out the policy, first request access to the API via this form [link](https://forms.gle/zCj5zjDvHsoeuMXU7). Then, follow these instructions to install [`sim_evals`](https://github.com/arhanjain/sim-evals) and launch evaluation.

```bash
# Clone repository
git clone --recurse-submodules https://github.com/arhanjain/sim-evals.git
cd sim-evals

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Activate uv environment
uv sync
source .venv/bin/activate

# [Optional] update pytorch versions
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

# Download assets (may need to export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> first)
uvx hf download owhan/DROID-sim-environments --repo-type dataset --local-dir assets

# Run eval script
cd ..
python eval_utils/run_sim_eval.py --host <API_HOST> --port <API_PORT> 
```

The outputs are saved in `runs` directory.


## Quick Start

### Prerequisites

- **Python**: 3.11
- **Hardware**: Multi-GPU setup (tested on GB200, H100)
  - Minimum: 2 GPUs for distributed inference
- **CUDA**: Compatible GPU with CUDA 12.9+
- **Conda**
  ```bash
  # Miniconda Installation
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh
  ```

### Installation

1. **Create conda environment:**
```bash
git clone --recurse-submodules git@github.com:Avant-US/dreamzero.git
cd dreamzero
conda create -n dreamzero python=3.11
conda activate dreamzero
```

2. **Install dependencies (PyTorch 2.8+ with CUDA 12.9+):**
```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129
```

3. **Install flash attention:**
```bash
MAX_JOBS=8 pip install --no-build-isolation flash-attn
```

4. **[GB200 ONLY, SKIP FOR H100] Install Transformer Engine:**
```bash
pip install --no-build-isolation transformer_engine[pytorch]
```

5. **[GB200 ONLY FOR TENSORRT, SKIP FOR H100] Install Tensorrt:**
```bash
pip install tensorrt==10.13.2.6 tensorrt_cu13==10.13.2.6 tensorrt_cu13_libs==10.13.2.6 tensorrt_cu13_bindings==10.13.2.6 --no-deps
pip install transformer_engine==2.10.0 transformer_engine_cu12==2.10.0 transformer_engine_torch==2.10.0
```

## Downloading Pretrained Checkpoints

### DreamZero-DROID (for inference)

We release a 14B pretrained DROID checkpoint on [Huggingface](https://huggingface.co/GEAR-Dreams/DreamZero-DROID). To download the checkpoint, run

```bash
mkdir -p ~/dreamzero/huggingface_checkpoints
hf download GEAR-Dreams/DreamZero-DROID --repo-type model --local-dir ~/dreamzero/huggingface_checkpoints
```

### DreamZero-AgiBot (for fine-tuning on new embodiments)

To fine-tune DreamZero on a new embodiment (e.g. YAM, AgiBot), download the pretrained [DreamZero-AgiBot](https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot) checkpoint (~45GB) to `./checkpoints/DreamZero-AgiBot`:

```bash
git clone https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot ./checkpoints/DreamZero-AgiBot
```

Or with the Hugging Face CLI:

```bash
hf download GEAR-Dreams/DreamZero-AgiBot --repo-type model --local-dir ./checkpoints/DreamZero-AgiBot
```

The YAM and AgiBot training scripts use `pretrained_model_path=./checkpoints/DreamZero-AgiBot` by default. See the [new embodiment guide](docs/DATASET_TO_GEAR_AND_TRAIN.md) for usage.

## Running the Inference Server

### Command Overview

The inference server uses PyTorch distributed training utilities to parallelize the model across multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 socket_test_optimized_AR.py --port 5000 --enable-dit-cache --model-path <path/to/checkpoint>
```

(Optional only for GB200) Tensorrt enables faster generation
```bash
export LOAD_TRT_ENGINE=<path/to/checkpoint>/tensorrt/wan/WanModel_nvfp4.trt 
export DYNAMIC_CACHE_SCHEDULE=true 
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 /mnt/aws-lfs-02/shared/seonghyeony/dreamzero/socket_test_optimized_AR.py --port 8000 --enable-dit-cache --model-path <path/to/checkpoint>
```
To verify the server is working, run a test client. The first few inferences will take a few minutes to warm up. After warming up, inference takes ~0.6s on GB200 and ~3s on H100.

```
python test_client_AR.py --port 5000
```

### Command-line Arguments

- `--port`: Port number for the WebSocket server (default: 8000)
- `--model-path`: Path to the pretrained model checkpoint directory
- `--enable-dit-cache`: Enable caching in DiT layers for faster inference (recommended)
- `--max-chunk-size`: Override max_chunk_size for inference (optional)
- `--timeout-seconds`: Server timeout in seconds (default: 50000)
- `--index`: Index for output directory naming (default: 0)
### Output

The server saves:
- **Videos**: Generated video predictions as MP4 files in `{model_path}/real_world_eval_gen_{date}_{index}/{checkpoint_name}/`
- **Input observations**: Saved per message in `{output_dir}/inputs/{msg_index}_{timestamp}/`


## Inference using Nvidia RTX Pro 6000 (Avant)

The table below summarizes the inference optimizations from the dreamzero paper (Table 1) and their status running with the RTX6000 hardare. 
#### Hardware Tested 
Single NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB

#### Additional Optimizations
These are optimizations not in Table 1 that are active
- `DYNAMIC_CACHE_SCHEDULE=true` — skipping redundant diffusion steps via cosine similarity
- `NUM_DIT_STEPS=5` — reducing from 16 to 5 base steps

**Table 1: RTX Pro 6000 optimization status vs paper**

| Paper Optimization | Speedup | Status | Bash Cmds |
|---|---|---|---|
| CFG Parallelism | 1.9x | MISSING — needs 2 GPUs, you have 1 | |
| DiT Caching | 5.5x | ENABLED (`--enable-dit-cache`) | |
| Torch Compile + CUDA Graphs | 8.9x | ENABLED (auto, but using FA2 fallback instead of TE) | |
| Kernel & Scheduler Opts | 9.6x | PARTIAL — TE not installed, falling back to FA2 | (Total Inference: avg ~3.7s) <br> DYNAMIC_CACHE_SCHEDULE=true NUM_DIT_STEPS=5 CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 socket_test_optimized_AR.py --port 5000 --enable-dit-cache --model-path ./huggingface_checkpoints  |
| Quantization (NVFP4) | 16.6x | MISSING — TRT engine incompatible with your GPU | |
| DreamZero-Flash | 38x | MISSING — requires specially trained checkpoint | |

Both the Transformer Engine and TensorRT upgrades require system-level CUDA development headers (cudnn.h, nccl.h) that aren't installed on this machine. These are typically available in NVIDIA Docker containers or data center setups but not on workstation installs.
 
Your current setup with FA2 + DiT caching + dynamic cache scheduling + reduced steps is already getting you ~3.7s per inference. The remaining optimizations (TE, TensorRT) would need either:                                                                                                                                                                                                         
  1. Install CUDA dev headers: sudo apt-get install libcudnn9-dev-cuda-12 libnccl-dev — then rebuild                                                                                     
  2. Use an NVIDIA container (like nvcr.io/nvidia/pytorch) which has everything pre-installed
  3. Ask your team if they have a working Docker image from the GB200 setup   

### Performance Summary

Measured on RTX PRO 6000 Blackwell (single GPU, no TensorRT) with DiT caching, dynamic cache scheduling, and `NUM_DIT_STEPS=5`:

**Table 2: RTX Pro 6000 inference breakdown**

| Component | Time | Notes |
|---|---|---|
| **Total inference** | **3.1 – 4.5s** (avg ~3.7s) | End-to-end per action chunk |
| Diffusion | 2.3 – 3.7s | Dominant bottleneck (~75% of total) |
| DIT Compute Steps | 4–6 steps | Dynamic cache skips redundant steps (vs 16 baseline) |
| KV Cache Creation | 0.35 – 0.68s | |
| Image Encoder | 0.38s first call, 0.00s cached | |
| Text Encoder | 0.05s | |
| VAE | 0.00 – 0.10s | |

For reference, the paper reports ~0.6s on GB200 (with TensorRT + NVFP4) and ~3s on H100. The pre-built TensorRT engine is not compatible with the RTX PRO 6000 and must be rebuilt for that platform.

## Inference using Single Nvidia H100 (Avant)

The table below summarizes the inference optimizations from the dreamzero paper (Table 1) and their status running with the H100 hardware.
#### Hardware Tested
Single NVIDIA H100 PCIe, 81559 MiB

#### Additional Optimizations
These are optimizations not in Table 1 that are active
- `DYNAMIC_CACHE_SCHEDULE=true` — skipping redundant diffusion steps via cosine similarity (default 16 base steps, dynamic cache reduces to 4-5 effective steps)

**Table 3: H100 optimization status vs paper**

| Paper Optimization | Speedup | Status | Bash Cmds |
|---|---|---|---|
| CFG Parallelism | 1.9x | MISSING — needs 2 GPUs, you have 1 | |
| DiT Caching | 5.5x | ENABLED (`--enable-dit-cache`) | |
| Torch Compile + CUDA Graphs | 8.9x | ENABLED (auto, but using FA2 fallback instead of TE) | |
| Kernel & Scheduler Opts | 9.6x | FA2 recommended over TE on H100 (see below) | (Total Inference: avg ~2.6s) <br> ATTENTION_BACKEND=FA2 DYNAMIC_CACHE_SCHEDULE=true CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 socket_test_optimized_AR.py --port 5000 --enable-dit-cache --model-path ./huggingface_checkpoints |
| Quantization (NVFP4/FP8) | 16.6x | NOT VIABLE on single 80GB GPU (see below) | |
| DreamZero-Flash | 38x | MISSING — requires specially trained checkpoint | |

### FA2 vs Transformer Engine on H100

The paper benchmarks TE (cuDNN fused attention) as faster than FA2 (Table 1: 9.6x vs 8.9x), but **those benchmarks were measured on GB200**. On H100, the results are reversed:

**Table 4: H100 attention backend comparison**

| Attention Backend | Avg Inference | Diffusion Time | Notes |
|---|---|---|---|
| **FA2 (FlashAttention2)** | **~2.6s** | 1.7 – 2.6s | Recommended for H100 |
| TE (cuDNN fused attn) | ~4.5s | 3.0 – 4.4s | ~60% slower on H100 |

FA2 is extremely well-optimized for H100/Hopper, while TE's cuDNN attention kernels are tuned for GB200/Blackwell. Use `ATTENTION_BACKEND=FA2` on H100 (set via env var before launch). The server defaults to TE if unset.

**Recommendation:** Use **FA2 on H100**, **TE on GB200/Blackwell**.

### TensorRT Quantization on H100 (NVFP4/FP8)

We attempted to build and run TensorRT engines with FP8 quantization on H100 GPUs (80GB each). The engine builds successfully and the diffusion step drops from ~2.6s to ~0.3-0.5s, but **the engine cannot run reliably due to GPU memory constraints — even with 2 GPUs**.

**What works:**
- FP8 TRT engine builds successfully (15.5GB engine file)
- Diffusion inference via TRT: ~0.85s per step (vs ~2.6s with PyTorch FA2)
- First 1-2 inference calls complete

**What fails:**
- OOM on the 3rd+ inference call as KV cache accumulates
- The TRT engine (15.5GB runtime) + PyTorch DiT (~28GB, needed for KV cache creation) + KV cache + other models exceeds 80GB
- The TRT engine only handles the diffusion-only path; KV cache creation still requires the full PyTorch DiT model on GPU

**Why this doesn't work even with 2x H100:**
CFG parallelism **replicates** the full model on each GPU (not splits it), so each GPU needs: TRT engine (15.5GB) + PyTorch DiT (~28GB for KV cache creation) + KV cache + other models = ~75GB+. As KV cache grows, it exceeds 80GB per GPU. The paper's GB200 likely had more per-GPU memory (96GB+) or a different KV cache strategy.

**Recommendation:** Use **FA2 + CFG parallelism** (~1.0s on 2x H100) for production. TRT on H100 is only useful as a benchmark for the diffusion step in isolation.

**To build and test the FP8 TRT engine yourself (Docker required):**
```bash
# Install ONNX export dependencies (inside the Docker container)
pip install --no-deps -e .
pip install onnxconverter-common onnx onnxruntime onnxslim
pip install onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
pip install numpy==1.26.4

# Build the FP8 engine (~10-15 min)
bash scripts/inference/build_trt_engine.sh \
    --model-path ./huggingface_checkpoints \
    --tensorrt fp8 \
    --cuda-device 0

# Run inference with TRT (will OOM after 2-3 calls on single 80GB GPU)
LOAD_TRT_ENGINE=./huggingface_checkpoints/tensorrt/wan/WanModel_fp8.trt \
    DYNAMIC_CACHE_SCHEDULE=true CUDA_VISIBLE_DEVICES=0 \
    torchrun --standalone --nproc_per_node=1 socket_test_optimized_AR.py \
    --port 5000 --enable-dit-cache --model-path ./huggingface_checkpoints
```

**To make TRT viable on H100:** Add a second GPU for CFG parallelism, which splits the memory load and is how the paper achieved ~0.6s on GB200.

### Performance Summary

Measured on H100 PCIe (single GPU, no TensorRT) with FA2, DiT caching, and dynamic cache scheduling (default 16 base steps):

**Table 5: H100 inference breakdown**

| Component | Time | Notes |
|---|---|---|
| **Total inference** | **2.3 – 3.1s** (avg ~2.6s) | End-to-end per action chunk |
| Warmup (1st call) | ~13s | Torch compile + cache initialization |
| Warmup (2nd call) | ~4s | VAE compile + scheduling warmup |
| Diffusion | 1.7 – 2.6s | Dominant bottleneck (~75% of total) |
| DIT Compute Steps | 4–5 steps | Dynamic cache skips redundant steps (from 16 base) |
| KV Cache Creation | 0.25 – 0.53s | |
| Image Encoder | 0.30s first call, 0.00s cached | |
| Text Encoder | 0.03s | |
| VAE | 0.00 – 0.08s | |

Note: Using `NUM_DIT_STEPS=5` (explicit 5 base steps) is slightly slower (~2.8s avg) because the dynamic cache has fewer steps to evaluate and skip. Letting the default 16 base steps run with `DYNAMIC_CACHE_SCHEDULE=true` gives the cache more room to optimize, landing at 4-5 effective steps with better skip decisions.

For reference, the paper reports ~0.6s on GB200 (with TensorRT + NVFP4) and ~3s on H100. The H100 PCIe with FA2 + DiT caching + dynamic scheduling achieves ~2.6s avg, better than the paper's baseline H100 figure.

### Analysis: Our Results vs Paper's Table 1

The paper reports a 5.7s baseline on a single GPU and claims cumulative speedups up to 9.6x on H100 (sub-0.6s). Here's how those claims map to implied latency:

**Table 6: Paper's Table 1 cumulative speedups (starting from 5.7s baseline)**

| Optimization | H100 Speedup | Implied Time |
|---|---|---|
| Baseline (single GPU) | 1x | 5.7s |
| + CFG Parallelism (2 GPUs) | 1.9x | 3.0s |
| + DiT Caching | 5.5x | 1.04s |
| + Torch Compile + CUDA Graphs | 8.9x | 0.64s |
| + Kernel & Scheduler Opts | 9.6x | 0.59s |

**Our setup:** Single H100 PCIe, FA2, DiT caching, dynamic cache scheduling → **~2.6s avg**

Despite having DiT caching, torch compile, and CUDA graphs all enabled, we see ~2.6s rather than the sub-1s the table might suggest. The key reasons:

1. **CFG Parallelism requires 2 GPUs (missing ~1.9x).** Table 1 is cumulative — every row after CFG Parallelism assumes 2 GPUs are already in use. Without a second GPU, the conditional and unconditional CFG passes run sequentially, so every diffusion step takes ~2x longer than what the paper assumes. This single factor accounts for most of the gap.

2. **H100 PCIe vs SXM.** The paper almost certainly benchmarked on H100 SXM, which has higher memory bandwidth (3.35 TB/s vs 2.0 TB/s) and TDP. PCIe variants are ~20-30% slower for memory-bound workloads like diffusion.

3. **DiT caching effectiveness.** The dynamic cache reduces 16 base steps to 4-5 effective steps via cosine similarity skipping. This is a large multiplier but doesn't reach the paper's ideal because the paper's benchmarks assume CFG parallelism is already active.

**Table 7: Adjusted expectations for single H100 PCIe**

| Configuration | Adjusted Speedup | Expected | Actual |
|---|---|---|---|
| Baseline (1 GPU, no opts) | 1x | 5.7s | — |
| + DiT Caching  | 5.5x / 1.9x = 2.9x | ~1.9s | — |
| + Torch Compile + CUDA Graphs | 8.9x / 1.9x = 4.7x | ~1.2s | — |
| + FA2 (instead of TE kernel opts) | 9.6x / 1.9x = 5.1x | ~1.1s | **~2.6s** |

The remaining gap between ~1.1s expected and ~2.6s actual is likely the PCIe bandwidth penalty (vs SXM).

**To get closer to the paper's performance:**
1. **Add a second GPU** for CFG parallelism — expected ~1.9x improvement (bringing ~2.6s → ~1.4s)
2. **Use H100 SXM** instead of PCIe — expected ~20-30% gain on top of that

**Bottom line:** Our ~2.6s on a single H100 PCIe is reasonable given the constraints. The paper's sub-1s H100 numbers assume 2x SXM GPUs with CFG parallelism active.

## Inference using Double Nvidia H100 (Avant)

With 2x H100 GPUs, CFG parallelism is enabled — the conditional and unconditional diffusion passes run in parallel across GPUs, cutting diffusion time roughly in half.

#### Hardware Tested
2x NVIDIA H100 80GB HBM3 (GCP a3-highgpu, 8x H100 node, using 2 GPUs)

#### Setup (Docker)

1. **Install Docker + NVIDIA Container Toolkit** (if not already):
```bash
sudo apt-get install -y docker.io
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sudo bash -c 'echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/\$(ARCH) /" > /etc/apt/sources.list.d/nvidia-container-toolkit.list'
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

2. **Build the Docker image** (requires `.dockerignore` to exclude checkpoints):
```bash
cd ~/dreamzero
sudo docker build -t dreamzero .
```

3. **Download checkpoint** (if not already present):
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('GEAR-Dreams/DreamZero-DROID', local_dir='./huggingface_checkpoints')"
```

4. **Run container with 2 GPUs** (adjust device IDs as needed):
```bash
sudo docker run --gpus '"device=6,7"' -it --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 5000:5000 \
  -v ~/dreamzero:/workspace \
  -v /mnt/localssd/huggingface_cache:/root/.cache/huggingface \
  -w /workspace \
  dreamzero
```
Note: Mount the HF cache to a disk with sufficient space (~30GB for Wan2.1 base model auto-download).

5. **Inside the container — launch 2-GPU inference:**
```bash
pip install --no-deps -e .
ATTENTION_BACKEND=FA2 DYNAMIC_CACHE_SCHEDULE=true \
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py --port 5000 --enable-dit-cache \
  --model-path ./huggingface_checkpoints
```
Note: `CUDA_VISIBLE_DEVICES=0,1` inside the container because Docker's `--gpus "device=6,7"` maps host GPUs 6,7 to container GPUs 0,1.

6. **Test from a second shell:**
```bash
sudo docker exec -it $(sudo docker ps -q) python test_client_AR.py --port 5000
```

### Performance Summary

Measured on 2x H100 80GB HBM3 (GCP) with FA2, DiT caching, dynamic cache scheduling, and CFG parallelism:

**Table 8: 2x H100 inference breakdown**

| Component | Time | Notes |
|---|---|---|
| **Total inference** | **0.91 – 1.23s** (avg ~1.0s) | End-to-end per action chunk |
| Warmup (1st call) | ~98s | Torch compile + cache initialization |
| Warmup (2nd call) | ~19s | VAE compile + scheduling warmup |
| Diffusion | 0.62 – 0.93s | CFG parallelism splits across 2 GPUs |
| DIT Compute Steps | 4–5 steps | Dynamic cache skips redundant steps (from 16 base) |
| KV Cache Creation | 0.10 – 0.19s | |
| Image Encoder | 0.20s first call, 0.00s cached | |
| Text Encoder | 0.01s | |
| VAE | 0.00 – 0.05s | |

**Table 9: Single vs Double H100 comparison**

| Setup | Avg Inference | Diffusion | Speedup |
|---|---|---|---|
| Single H100 PCIe (FA2) | ~2.6s | 1.7 – 2.6s | 1x |
| **2x H100 (FA2 + CFG)** | **~1.0s** | **0.6 – 0.9s** | **2.6x** |
| Paper (GB200 + TRT + NVFP4) | ~0.6s | — | — |

The 2x H100 setup achieves ~1.0s without TRT, already close to the paper's 0.6s GB200 result. TRT is not viable on H100 even with 2 GPUs — CFG parallelism replicates the full model per GPU, so each GPU still needs TRT + PyTorch DiT + KV cache > 80GB. See the [TensorRT section above](#tensorrt-quantization-on-h100-nvfp4fp8) for details.

## Inference using Double Nvidia H200 (Avant)

The H200 has 141GB HBM3e per GPU (vs H100's 80GB), providing enough memory for TRT + PyTorch DiT + KV cache to coexist on each GPU. This enables TensorRT FP8 quantization with CFG parallelism.

#### Hardware Tested
2x NVIDIA H200 141GB HBM3e (GCP, 8x H200 node, using 2 GPUs)

#### Setup (Docker)

1. **Install Docker + NVIDIA Container Toolkit** (if not already):
```bash
sudo apt-get update && sudo apt-get install -y docker.io
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sudo bash -c 'echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/\$(ARCH) /" > /etc/apt/sources.list.d/nvidia-container-toolkit.list'
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

2. **Clone and build:**
```bash
git clone --recurse-submodules https://github.com/Avant-US/dreamzero.git
cd dreamzero
docker build -t dreamzero .
```

3. **Run container with 2 GPUs:**
```bash
mkdir -p /dev/shm/huggingface_cache
docker run --gpus '"device=0,1"' -it --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 5000:5000 \
  -v $(pwd):/workspace \
  -v /dev/shm/huggingface_cache:/root/.cache/huggingface \
  -w /workspace \
  dreamzero
```
Note: HF cache is mounted to `/dev/shm` (RAM-backed tmpfs, ~1.5TB available). Fast but lost on reboot.

4. **Inside the container — download checkpoint and run:**
```bash
pip install --no-deps -e .

# Download checkpoint (~45GB, saves to RAM-backed cache)
python -c "from huggingface_hub import snapshot_download; snapshot_download('GEAR-Dreams/DreamZero-DROID', local_dir='/root/.cache/huggingface/dreamzero-checkpoints')"

# Run 2-GPU inference with FA2 + CFG parallelism
ATTENTION_BACKEND=FA2 DYNAMIC_CACHE_SCHEDULE=true \
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py --port 5000 --enable-dit-cache \
  --model-path /root/.cache/huggingface/dreamzero-checkpoints
```

5. **Test from a second shell:**
```bash
docker exec -it $(docker ps -q) python test_client_AR.py --port 5000
```

6. **(Optional) Build and run with TRT FP8 for maximum speed:**
```bash
# Install ONNX deps
pip install onnxconverter-common onnx onnxruntime onnxslim numpy==1.26.4
pip install onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com

# Build FP8 engine (~10-15 min)
bash scripts/inference/build_trt_engine.sh \
    --model-path /root/.cache/huggingface/dreamzero-checkpoints \
    --tensorrt fp8 --cuda-device 0

# Run with TRT + CFG parallelism
LOAD_TRT_ENGINE=/root/.cache/huggingface/dreamzero-checkpoints/tensorrt/wan/WanModel_fp8.trt \
  DYNAMIC_CACHE_SCHEDULE=true CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 socket_test_optimized_AR.py \
  --port 5000 --enable-dit-cache \
  --model-path /root/.cache/huggingface/dreamzero-checkpoints
```

### Performance Summary (FA2 + CFG, no TRT)

Measured on 2x H200 141GB HBM3e (GCP) with FA2, DiT caching, dynamic cache scheduling, and CFG parallelism:

**Table 10: 2x H200 FA2 inference breakdown**

| Component | Time | Notes |
|---|---|---|
| **Total inference** | **0.85 – 1.15s** (avg ~0.93s) | End-to-end per action chunk |
| Warmup (1st call) | ~90s | Torch compile + cache initialization |
| Warmup (2nd call) | ~17s | VAE compile + scheduling warmup |
| Diffusion | 0.58 – 0.88s | CFG parallelism splits across 2 GPUs |
| DIT Compute Steps | 4–5 steps | Dynamic cache skips redundant steps (from 16 base) |
| KV Cache Creation | 0.09 – 0.18s | |
| Image Encoder | 0.18-0.24s first call, 0.00s cached | |
| Text Encoder | 0.01s | |
| VAE | 0.00 – 0.05s | |

### Performance Summary (TRT FP8 + CFG)

Measured on 2x H200 141GB HBM3e (GCP) with TRT FP8, DiT caching, dynamic cache scheduling, and CFG parallelism. **No OOM** — H200's 141GB per GPU fits TRT engine + PyTorch DiT + KV cache simultaneously.

**Table 11: 2x H200 TRT FP8 inference breakdown**

| Component | Time | Notes |
|---|---|---|
| **Total inference** | **0.53 – 0.64s** (avg ~0.58s) | End-to-end per action chunk |
| Warmup (1st call) | ~15s | TRT engine initialization |
| Warmup (2nd call) | ~3s | VAE compile |
| Diffusion | 0.27 – 0.38s | TRT FP8 + CFG parallelism |
| DIT Compute Steps | 4–5 steps | Dynamic cache skips redundant steps (from 16 base) |
| KV Cache Creation | 0.08 – 0.19s | PyTorch DiT stays on GPU (no offloading needed) |
| Image Encoder | 0.18-0.23s first call, 0.00s cached | |
| Text Encoder | 0.01s | |
| VAE | 0.00 – 0.06s | |

Note: TRT on H200 requires disabling the CPU offloading code (designed for H100's 80GB limit). Comment out `self.model.cpu()` and `torch.cuda.empty_cache()` in `wan_flow_matching_action_tf.py` lines ~908 and ~1392.

**Table 12: All hardware comparison**

| Setup | Avg Inference | Diffusion | VRAM/GPU | No OOM |
|---|---|---|---|---|
| Single H100 PCIe (FA2) | ~2.6s | 1.7 – 2.6s | 80GB | Yes |
| 2x H100 (FA2 + CFG) | ~1.0s | 0.6 – 0.9s | 80GB | Yes |
| 2x H100 (TRT + CFG) | OOM | — | 80GB | No |
| 2x H200 (FA2 + CFG) | ~0.93s | 0.58 – 0.88s | 141GB | Yes |
| **2x H200 (TRT FP8 + CFG)** | **~0.58s** | **0.27 – 0.38s** | **141GB** | **Yes** |
| Paper (GB200 + TRT + NVFP4) | ~0.6s | — | 192GB | Yes |

**2x H200 with TRT FP8 matches the paper's GB200 result (~0.6s) and is actually slightly faster (~0.58s).** The key enabler is H200's 141GB VRAM — enough for TRT engine (15.5GB) + PyTorch DiT (~28GB) + KV cache + other models to coexist on each GPU without offloading.


## Training

> **Training on a new embodiment?** See [Adding a New Embodiment to DreamZero](docs/DATASET_TO_GEAR_AND_TRAIN.md) for a complete guide on converting your dataset, configuring modalities, and launching training. <em>Make sure to align the 3 camera view order to ensure positive transfer.</em>

### Downloading Pretrained Base Model Weights

DreamZero is built on top of [Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) and uses the [umt5-xxl](https://huggingface.co/google/umt5-xxl) tokenizer. Download both before training:

```bash
pip install "huggingface_hub[cli]"

# You may need to set your HuggingFace token:
# export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>

# Download Wan2.1 model weights (~28GB)
hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P

# Download umt5-xxl tokenizer
hf download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl
```

> **Note:** The training script will auto-download these if they are not found at the configured paths, but pre-downloading is recommended to avoid delays at launch.

### DROID Dataset

We release the preprocessed DROID dataset used to train DreamZero on HuggingFace: [GEAR-Dreams/DreamZero-DROID-Data](https://huggingface.co/datasets/GEAR-Dreams/DreamZero-DROID-Data).

This dataset is derived from the [DROID 1.0.1](https://droid-dataset.github.io/) dataset with the following modifications:
- Converted from RLDS/TFDS format to [LeRobot](https://github.com/huggingface/lerobot) v2.0 format
- Idle frames removed using [Physical Intelligence's idle frame detector](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md#data-filtering) (`droid_sample_ranges_v1_0_1.json`)
- Episodes without language annotations are filtered out
- Successful episodes only (episodes with non-zero reward)
- 3 camera views: `exterior_image_1_left`, `exterior_image_2_left`, `wrist_image_left`

**To download the preprocessed dataset (~131GB):**

```bash
huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir ./data/droid_lerobot
```

If you want to reproduce the dataset conversion from raw DROID 1.0.1 yourself (or modify the filtering), see [docs/DROID_CONVERSION.md](docs/DROID_CONVERSION.md).

### Running Training

```bash
# Configure paths (override defaults as needed)
export DROID_DATA_ROOT="./data/droid_lerobot"
export OUTPUT_DIR="./checkpoints/dreamzero_droid"
export NUM_GPUS=4

# Point to your downloaded model weights (if not using default paths)
export WAN_CKPT_DIR="./checkpoints/Wan2.1-I2V-14B-480P"
export TOKENIZER_DIR="./checkpoints/umt5-xxl"

# Launch training
bash scripts/train/droid_training.sh
```

**Using Wan2.2-TI2V-5B backbone (5B params, lower VRAM):** To train with the smaller Wan2.2-TI2V-5B model instead of Wan2.1-I2V-14B, see [docs/WAN22_BACKBONE.md](docs/WAN22_BACKBONE.md) and run `bash scripts/train/droid_training_wan22.sh`.

### Training Configuration

The training script uses Hydra for configuration and DeepSpeed ZeRO Stage 2 for distributed training. Key defaults:

**Table 8: Training configuration defaults**

| Parameter | Default | Description |
|---|---|---|
| `NUM_GPUS` | 4 | Number of GPUs |
| `per_device_train_batch_size` | 1 | Batch size per GPU |
| `learning_rate` | 1e-5 | Learning rate |
| `max_steps` | 10 | Max training steps (increase for full training) |
| `warmup_ratio` | 0.05 | Warmup ratio |
| `weight_decay` | 1e-5 | Weight decay |
| `image_resolution_width` | 320 | Image width |
| `image_resolution_height` | 176 | Image height |
| `num_frames` | 33 | Number of video frames |
| `action_horizon` | 24 | Action prediction horizon |
| `save_lora_only` | true | Only save LoRA weights |
| `bf16` | true | Use bfloat16 precision |

> **Note:** `max_steps=10` is set for a quick sanity check. For full training, increase this to your desired number of steps and configure `save_steps` / `save_strategy` accordingly.


## Citation

If you use DreamZero in your research, please cite:

```bibtex
@misc{ye2026worldactionmodelszeroshot,
      title={World Action Models are Zero-shot Policies}, 
      author={Seonghyeon Ye and Yunhao Ge and Kaiyuan Zheng and Shenyuan Gao and Sihyun Yu and George Kurian and Suneel Indupuru and You Liang Tan and Chuning Zhu and Jiannan Xiang and Ayaan Malik and Kyungmin Lee and William Liang and Nadun Ranawaka and Jiasheng Gu and Yinzhen Xu and Guanzhi Wang and Fengyuan Hu and Avnish Narayan and Johan Bjorck and Jing Wang and Gwanghyun Kim and Dantong Niu and Ruijie Zheng and Yuqi Xie and Jimmy Wu and Qi Wang and Ryan Julian and Danfei Xu and Yilun Du and Yevgen Chebotar and Scott Reed and Jan Kautz and Yuke Zhu and Linxi "Jim" Fan and Joel Jang},
      year={2026},
      eprint={2602.15922},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.15922}, 
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Support

For issues and questions:
- Check the troubleshooting section above
- Review server logs for detailed error messages
- Verify your checkpoint is compatible with this release

[![Star History Chart](https://api.star-history.com/svg?repos=dreamzero0/dreamzero&type=Date)](https://star-history.com/#dreamzero0/dreamzero&Date)
