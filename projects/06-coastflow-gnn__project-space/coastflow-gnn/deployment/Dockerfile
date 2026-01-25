# CoastFlow-GNN Docker Image
# Multi-stage build for optimized production image

# ==============================================================================
# Stage 1: Builder with CUDA and PyTorch
# ==============================================================================
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install PyTorch Geometric and dependencies
RUN pip install --no-cache-dir \
    torch-geometric>=2.4.0 \
    torch-scatter \
    torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Copy requirements and install dependencies
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 2: Production image
# ==============================================================================
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash coastflow

# Create working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /opt/conda /opt/conda

# Copy application code
COPY --chown=coastflow:coastflow . .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/checkpoints \
    && chown -R coastflow:coastflow /app

# Switch to non-root user
USER coastflow

# Expose Jupyter port
EXPOSE 8888

# Default command: run training
CMD ["python", "src/training/train_single.py", "--epochs", "50", "--output-dir", "/app/outputs"]

# ==============================================================================
# Usage:
# ==============================================================================
# Build:
#   docker build -t coastflow-gnn:latest -f deployment/Dockerfile .
#
# Run training:
#   docker run --gpus all -v $(pwd)/outputs:/app/outputs coastflow-gnn:latest
#
# Run Jupyter:
#   docker run --gpus all -p 8888:8888 coastflow-gnn:latest \
#     jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
#
# Interactive shell:
#   docker run --gpus all -it coastflow-gnn:latest /bin/bash
# ==============================================================================
