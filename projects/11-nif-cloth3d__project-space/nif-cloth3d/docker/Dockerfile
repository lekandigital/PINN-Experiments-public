# NIF-Cloth3D-Interactive Docker Image
# Base: NVIDIA CUDA 12.1 with cuDNN 8 for PyTorch 2.1+
# Includes: PyTorch, Blender (headless), training dependencies

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    unzip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Blender dependencies
    libxi6 \
    libxxf86vm1 \
    libxfixes3 \
    libxkbcommon0 \
    libgl1-mesa-glx \
    libegl1-mesa \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Blender 4.0 (LTS) for headless operation
RUN wget -q https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz \
    && tar -xf blender-4.0.2-linux-x64.tar.xz \
    && mv blender-4.0.2-linux-x64 /opt/blender \
    && ln -s /opt/blender/blender /usr/local/bin/blender \
    && rm blender-4.0.2-linux-x64.tar.xz

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch 2.1+ with CUDA 12.1 support
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional ML/visualization tools
RUN pip3 install --no-cache-dir \
    tensorboard \
    wandb \
    tqdm

# Copy project files
COPY src/ /workspace/src/
COPY configs/ /workspace/configs/
COPY tests/ /workspace/tests/
COPY scripts/ /workspace/scripts/

# Create output directories
RUN mkdir -p /workspace/checkpoints /workspace/logs /workspace/data

# Set Python path
ENV PYTHONPATH="/workspace/src:${PYTHONPATH}"

# Expose TensorBoard port
EXPOSE 6006

# Default command: show help
CMD ["python3", "src/train.py", "--help"]

# Usage:
# Build: docker build -t nif-cloth3d .
# Run training: docker run --gpus all -v $(pwd)/data:/workspace/data nif-cloth3d python3 src/train.py --config configs/l40s.yaml
# Interactive: docker run --gpus all -it nif-cloth3d bash
