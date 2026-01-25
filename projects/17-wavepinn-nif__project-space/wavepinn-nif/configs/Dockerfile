# WavePINN-NIF-Scalar Docker Image
# Build: docker build -t wavepinn-nif-scalar -f configs/Dockerfile .
# Run:   docker run --gpus all -it wavepinn-nif-scalar

FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
# JAX with CUDA 12 support
RUN pip install --no-cache-dir "jax[cuda12]>=0.4.14"
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package in development mode
RUN pip install -e .

# Default command: run tests
CMD ["python", "-m", "pytest", "tests/", "-v"]
