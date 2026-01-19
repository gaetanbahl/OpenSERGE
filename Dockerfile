# OpenSERGE Docker Image
# Based on PyTorch 2.4.0 with CUDA 12.4 and cuDNN 9
# Includes all dependencies for training, inference, and evaluation (TOPO, APLS metrics)

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    # Graphics and visualization
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # For OpenCV
    libopencv-dev \
    # Spatial indexing (for TOPO metrics)
    libspatialindex-dev \
    # Go language (for APLS metrics)
    golang-go \
    # Parallel processing
    parallel \
    # Text processing utilities
    vim \
    less \
    htop \
    tmux \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify CUDA installation
RUN nvcc --version && \
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Set working directory
WORKDIR /workspace/openserge

# Copy requirements first for better caching
COPY requirements.txt .
COPY metrics/topo/requirements.txt ./metrics/topo/
COPY metrics/apls/requirements.txt ./metrics/apls/

# Install Python dependencies
# Install base requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install TOPO metric dependencies
RUN pip install --no-cache-dir -r metrics/topo/requirements.txt

# Install APLS metric dependencies (minimal, mostly uses Go)
RUN pip install --no-cache-dir -r metrics/apls/requirements.txt || true

# Verify critical packages
RUN python -c "import torch; import torchvision; import timm; import cv2; import numpy; import scipy; import sklearn; print('All critical packages imported successfully')"

# Copy the entire project
COPY . .

# Install OpenSERGE in development mode
RUN pip install -e .

# Initialize Go modules for APLS metric
WORKDIR /workspace/openserge/metrics/apls
RUN go mod init apls || true && \
    go mod tidy || true && \
    go get github.com/dhconnelly/rtreego || true

# Verify Go installation
RUN go version

# Build APLS metric (optional pre-compilation)
RUN go build -o apls_metric main.go || echo "APLS will be built on first run"

# Return to main workspace
WORKDIR /workspace/openserge

# Create directories for data, checkpoints, and results
RUN mkdir -p \
    data \
    checkpoints \
    logs \
    evaluation_results \
    roadtracer_evaluation_results \
    globalscale_evaluation_results

# Make evaluation scripts executable
RUN chmod +x scripts/*.sh

# Configure GNU parallel to not show citation notice
RUN mkdir -p ~/.parallel && touch ~/.parallel/will-cite

# Set up Jupyter notebook extensions (optional)
RUN jupyter labextension list || true

# Expose ports
# 8888 for Jupyter Lab
# 6006 for TensorBoard
EXPOSE 8888 6006

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command: start Jupyter Lab
# Users can override with training/inference commands
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

# Usage examples (add to README):
#
# Build:
#   docker build -t openserge:latest .
#
# Run with GPU for training:
#   docker run --gpus all -v $(pwd)/data:/workspace/openserge/data \
#              -v $(pwd)/checkpoints:/workspace/openserge/checkpoints \
#              -p 8888:8888 -p 6006:6006 \
#              openserge:latest
#
# Run training:
#   docker run --gpus all -v $(pwd)/data:/workspace/openserge/data \
#              -v $(pwd)/checkpoints:/workspace/openserge/checkpoints \
#              openserge:latest \
#              python -m openserge.train --config configs/cityscale.json
#
# Run inference:
#   docker run --gpus all -v $(pwd)/data:/workspace/openserge/data \
#              -v $(pwd)/checkpoints:/workspace/openserge/checkpoints \
#              openserge:latest \
#              python -m openserge.infer --weights checkpoints/best_model.pt \
#              --image data/test_image.png --output results/graph.json
#
# Run evaluation (CityScale):
#   docker run --gpus all -v $(pwd)/data:/workspace/openserge/data \
#              -v $(pwd)/checkpoints:/workspace/openserge/checkpoints \
#              -v $(pwd)/evaluation_results:/workspace/openserge/evaluation_results \
#              openserge:latest \
#              bash scripts/run_cityscale_evaluation.sh checkpoints/best_model.pt 8 test
#
# Run evaluation (RoadTracer):
#   docker run --gpus all -v $(pwd)/data:/workspace/openserge/data \
#              -v $(pwd)/checkpoints:/workspace/openserge/checkpoints \
#              -v $(pwd)/roadtracer_evaluation_results:/workspace/openserge/roadtracer_evaluation_results \
#              openserge:latest \
#              bash scripts/run_roadtracer_evaluation.sh checkpoints/best_model.pt 8
#
# Interactive shell:
#   docker run --gpus all -it -v $(pwd)/data:/workspace/openserge/data \
#              openserge:latest bash
