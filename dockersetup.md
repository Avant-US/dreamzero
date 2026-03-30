# DreamZero Docker Setup

This guide walks through building and running DreamZero using NVIDIA's PyTorch Docker container. Tested on H100 and H200 GPUs.

---

## Prerequisites

- NVIDIA GPU (H100 or H200)
- Ubuntu (tested on 22.04)
- NVIDIA driver installed (verify with `nvidia-smi`)
- Docker installed

Check:
```bash
docker --version
nvidia-smi
```

## 1. Install NVIDIA Container Toolkit
This enables Docker to access GPUs.
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 2. Verify GPU Access in Docker
```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```
You should see your GPU(s) listed inside the container.

## 3. Login to NVIDIA NGC
NGC requires API key authentication.

Create API Key:
Go to: https://ngc.nvidia.com/
Generate API key

Login:
```bash
export NGC_API_KEY='your_key_here'
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

## 4. Build the DreamZero Docker Image
From the repo root (where the `Dockerfile` lives):
```bash
cd ~/dreamzero
docker build -t dreamzero .
```
This extends the NVIDIA PyTorch container (`nvcr.io/nvidia/pytorch:26.02-py3`) with:
- System libraries for OpenCV/GUI
- All Python dependencies from `pyproject.toml` (including PyTorch with CUDA 12.9)
- Flash Attention

## 5. Run the Container
```bash
docker run --gpus all -it --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 5000:5000 \
  -v ~/dreamzero:/workspace \
  -w /workspace \
  dreamzero
```
Notes:
- `--gpus all` -- exposes all GPUs (use `--gpus '"device=0,1"'` to select specific ones)
- `--ipc=host` -- required for PyTorch shared memory
- `--ulimit` -- avoids memory issues
- `-p 5000:5000` -- exposes the inference server port so clients can connect from the host
- `-v ~/dreamzero:/workspace` -- mounts your code so edits on host reflect inside the container
- Dependencies are baked into the image, so nothing is lost when the container exits

To keep the container around after exiting (instead of `--rm`):
```bash
docker run --gpus all -it \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 5000:5000 \
  -v ~/dreamzero:/workspace \
  -w /workspace \
  --name dreamzero-dev \
  dreamzero
```
Then restart later with: `docker start -ai dreamzero-dev`

## 6. Verify Transformer Engine
Inside the container:
```bash
python - <<'PY'
import transformer_engine.pytorch as te

print("Linear exists:", hasattr(te, "Linear"))
print("fp8_autocast exists:", hasattr(te, "fp8_autocast"))
PY
```

## 7. Test FP8 Execution
```bash
python - <<'PY'
import torch
import transformer_engine.pytorch as te

print("GPU:", torch.cuda.get_device_name(0))

layer = te.Linear(1024, 1024).cuda()
x = torch.randn(8, 1024, device="cuda")

with te.fp8_autocast(enabled=True):
    y = layer(x)

print("Output shape:", y.shape)
PY
```

## 8. Install Editable Package

The volume mount overlays the container's `/workspace`, so the editable install from the Dockerfile is lost. Reinstall after starting the container (required once per fresh container):
```bash
pip install --no-deps -e .
```

## 9. Launch Inference Server

### 2x H200 (recommended)

Use TE attention backend on H200 for best performance (~0.87s avg). H200's 4.8 TB/s HBM3e bandwidth favors TE's memory-bound kernels over FA2.

```bash
DYNAMIC_CACHE_SCHEDULE=true \
ATTENTION_BACKEND=TE \
CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py \
  --port 5000 --enable-dit-cache --model-path ./huggingface_checkpoints
```

### 2x H100

Use FA2 attention backend on H100 (~1.0s avg). TE's cuDNN kernels are optimized for Blackwell and are slower than FlashAttention2 on Hopper.

```bash
DYNAMIC_CACHE_SCHEDULE=true \
ATTENTION_BACKEND=FA2 \
CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py \
  --port 5000 --enable-dit-cache --model-path ./huggingface_checkpoints
```

### Single GPU (H100 or H200)

```bash
DYNAMIC_CACHE_SCHEDULE=true \
ATTENTION_BACKEND=FA2 \
CUDA_VISIBLE_DEVICES=0 \
  torchrun --standalone --nproc_per_node=1 \
  socket_test_optimized_AR.py \
  --port 5000 --enable-dit-cache --model-path ./huggingface_checkpoints
```

### Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `ATTENTION_BACKEND` | `TE`, `FA2`, `FA3`, `torch` | `TE` | Attention kernel backend. Use TE on H200/GB200, FA2 on H100 |
| `DYNAMIC_CACHE_SCHEDULE` | `true`, `false` | `false` | Cosine-similarity step skipping (16 -> ~4-5 effective steps) |
| `DISABLE_TORCH_COMPILE` | `true`, `false` | `false` | Disable torch.compile on encoders/VAE/scheduler |
| `NUM_DIT_STEPS` | `5`,`6`,`7`,`8`,`16` | `8` | Base diffusion steps (before dynamic cache skipping) |
| `LOAD_TRT_ENGINE` | path | unset | Load TensorRT FP8 engine for quantized diffusion |
| `ENABLE_TENSORRT` | `true`, `false` | `false` | Enable TRT mode (disables torch.compile + TE) |

### Testing

Test from a second shell (or from the host if port is exposed):
```bash
docker exec -it dreamzero-dev python test_client_AR.py --port 5000
```

## 10. (Optional) Build TensorRT FP8 Engine

TensorRT quantization reduces diffusion time significantly but requires sufficient GPU memory. On a single 80GB H100 it OOMs after 2-3 inference calls. This optimization is viable with 2x H200 (141GB each) or 2+ GPUs with sufficient memory.

To build the engine (~10-15 min):
```bash
# Install ONNX export dependencies
pip install onnxconverter-common onnx onnxruntime onnxslim
pip install onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
pip install numpy==1.26.4

# Build FP8 engine
bash scripts/inference/build_trt_engine.sh \
    --model-path ./huggingface_checkpoints \
    --tensorrt fp8 \
    --cuda-device 0

# Run with TRT engine (2x H200)
LOAD_TRT_ENGINE=./huggingface_checkpoints/tensorrt/wan/WanModel_fp8.trt \
DYNAMIC_CACHE_SCHEDULE=true \
CUDA_VISIBLE_DEVICES=0,1 \
  torchrun --standalone --nproc_per_node=2 \
  socket_test_optimized_AR.py \
  --port 5000 --enable-dit-cache --model-path ./huggingface_checkpoints
```
