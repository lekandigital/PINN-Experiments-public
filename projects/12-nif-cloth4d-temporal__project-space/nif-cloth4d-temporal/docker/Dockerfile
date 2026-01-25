# NIF-Cloth4D-Temporal Docker Image
# Base: NVIDIA CUDA 11.8 with cuDNN 8
# Target: NVIDIA L40S GPU for ≥3 fps inference

FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu22.04

LABEL maintainer="NIF-Cloth4D-Temporal"
LABEL description="Neural Implicit Field for Continuous-Time Cloth Simulation"

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Create conda environment
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Activate environment by default
ENV CONDA_DEFAULT_ENV=nif_cloth4d
ENV PATH=$CONDA_DIR/envs/$CONDA_DEFAULT_ENV/bin:$PATH

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Ensure CUDA is visible
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Enable cuDNN autotuning for performance
ENV TORCH_CUDNN_V8_API_ENABLED=1

# Default command
CMD ["python", "scripts/train.py", "--help"]
