FROM nvcr.io/nvidia/pytorch:26.02-py3

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies, preserving the container's native
# PyTorch/torchvision/torchaudio (required for Transformer Engine compatibility)
COPY pyproject.toml .
RUN pip install --no-cache-dir --no-deps -e . && \
    pip install --no-cache-dir -r /dev/stdin --extra-index-url https://download.pytorch.org/whl/cu129 <<'EOF'
av==15.0.0
pyttsx3==2.90
scipy==1.15.3
numpy==1.26.4
matplotlib
hydra-core
ray[default]==2.47.1
click
gymnasium
mujoco
termcolor
flask
python-socketio>=5.13.0
flask_socketio
loguru
lmdb
meshcat
meshcat-shapes
rerun-sdk==0.21.0
pygame
sshkeyboard
msgpack
msgpack-numpy
peft==0.5.0
pyzmq
PyQt6
pin
pin-pink
timm
tyro
redis
lark
datasets==3.6.0
pandas
evdev
pybullet
gear
multi-storage-client[boto3,msal,observability-otel]==0.33.0
dm_tree
openai
transformers==4.51.3
albumentations==1.4.18
einops==0.8.1
tianshou==0.5.1
imageio==2.34.2
imageio-ffmpeg
wandb
opencv-python==4.8.0.74
diffusers==0.30.2
ftfy
nvidia-modelopt
nvidia-modelopt-core
tensorrt
openpi-client==0.1.1
huggingface_hub
decord2
deepspeed
tiktoken
sentencepiece
pytest>=7.0.0
black>=23.0.0
isort>=5.12.0
EOF

# Install flash attention
RUN MAX_JOBS=8 pip install --no-cache-dir --no-build-isolation flash-attn

# Default working directory for mounted code
WORKDIR /workspace
