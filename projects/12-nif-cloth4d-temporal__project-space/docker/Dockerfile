FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget \
    libhdf5-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt /workspace/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code
COPY . /workspace/

CMD ["bash"]
