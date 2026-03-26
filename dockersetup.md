# DreamZero Docker Setup (H100)

This guide walks through building and running DreamZero using NVIDIA's PyTorch Docker container on an H100 GPU. The Dockerfile installs all dependencies into the image so they persist across container restarts.

---

## Prerequisites

- NVIDIA H100 GPU
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
You should see your H100 listed inside the container.

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
sudo docker run --gpus all -it --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v ~/dreamzero:/workspace \
  -w /workspace \
  dreamzero
```
Notes:
- `--ipc=host` -- required for PyTorch shared memory
- `--ulimit` -- avoids memory issues
- `-v ~/dreamzero:/workspace` -- mounts your code so edits on host reflect inside the container
- Dependencies are baked into the image, so nothing is lost when the container exits

To keep the container around after exiting (instead of `--rm`):
```bash
sudo docker run --gpus all -it \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
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

## 7. Test FP8 Execution (H100)
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
