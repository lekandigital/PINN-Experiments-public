# Dockerfile for HGNN-NIF-Cloth
# 
# Build: docker build -t hgnn-nif-cloth -f docker/Dockerfile .
# Run:   docker run --gpus all -v $(pwd)/outputs:/workspace/hgnn-nif-cloth/outputs hgnn-nif-cloth
#
# For interactive use:
#   docker run --gpus all -it -v $(pwd):/workspace/hgnn-nif-cloth hgnn-nif-cloth bash

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /workspace/hgnn-nif-cloth

# Copy requirements first (for layer caching)
COPY docker/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set Python path
ENV PYTHONPATH=/workspace/hgnn-nif-cloth
ENV PYTHONUNBUFFERED=1

# Set CUDA environment
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create output directories
RUN mkdir -p outputs data

# Default command: run training with test mode
CMD ["python", "scripts/train_local.py", "--test_mode"]

# Alternative commands:
# Full training:
#   CMD ["python", "scripts/train_local.py", "--epochs", "100", "--batch_size", "8"]
#
# Run tests:
#   CMD ["python", "-m", "pytest", "tests/", "-v"]
#
# Interactive:
#   CMD ["bash"]
