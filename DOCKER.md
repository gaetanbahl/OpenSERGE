# OpenSERGE Docker Guide

This guide explains how to use Docker for training, inference, and evaluation with OpenSERGE.

## Prerequisites

- **Docker**: Version 20.10 or later ([Install Docker](https://docs.docker.com/get-docker/))
- **NVIDIA Docker Runtime**: For GPU support ([Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- **Docker Compose**: Version 2.0 or later (usually included with Docker Desktop)

### Verify GPU Support

```bash
# Test NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

If this command shows your GPU(s), you're ready to use OpenSERGE with Docker!

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up -d

# Access Jupyter Lab at http://localhost:8888

# Execute training
docker-compose exec openserge python -m openserge.train --config configs/cityscale.json

# Stop the container
docker-compose down
```

### Option 2: Docker CLI

```bash
# Build the image
docker build -t openserge:latest .

# Run Jupyter Lab
docker run --gpus all -p 8888:8888 \
  -v $(pwd)/data:/workspace/openserge/data:ro \
  -v $(pwd)/checkpoints:/workspace/openserge/checkpoints \
  openserge:latest

# Or run training directly
docker run --gpus all \
  -v $(pwd)/data:/workspace/openserge/data:ro \
  -v $(pwd)/checkpoints:/workspace/openserge/checkpoints \
  openserge:latest \
  python -m openserge.train --config configs/cityscale.json
```

## Image Details

**Base Image**: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`

**Includes**:
- PyTorch 2.4.0 with CUDA 12.4 and cuDNN 9
- All Python dependencies from `requirements.txt`
- TOPO metric dependencies (Python: numpy, scipy, hopcroftkarp, rtree, svgwrite)
- APLS metric dependencies (Go 1.21+, rtreego library)
- GNU parallel for batch evaluation
- Jupyter Lab for interactive development
- OpenCV, timm, and all ML dependencies

**Default Command**: Starts Jupyter Lab on port 8888

**Working Directory**: `/workspace/openserge`

## Common Use Cases

### 1. Training

#### CityScale Dataset

```bash
docker-compose exec openserge python -m openserge.train \
  --config configs/cityscale.json
```

#### RoadTracer Dataset

```bash
docker-compose exec openserge python -m openserge.train \
  --config configs/roadtracer.json
```

#### With Custom Settings

```bash
docker-compose exec openserge python -m openserge.train \
  --config configs/cityscale.json \
  --batch_size 8 \
  --lr 0.0005 \
  --stage1_epochs 500
```

### 2. Inference

#### Single Image

```bash
docker-compose exec openserge python -m openserge.infer \
  --weights checkpoints/best_model.pt \
  --image data/test_image.png \
  --output results/graph.json \
  --output_image results/visualization.png
```

#### Large Image with Sliding Window

```bash
docker-compose exec openserge python -m openserge.infer \
  --weights checkpoints/best_model.pt \
  --image data/large_region.png \
  --output results/graph.json \
  --img_size 512 \
  --stride 448 \
  --merge_threshold 16.0
```

### 3. Evaluation

#### CityScale/Sat2Graph Evaluation

```bash
# Evaluate on test split
docker-compose exec openserge bash scripts/run_cityscale_evaluation.sh \
  checkpoints/best_model.pt 8 test

# Evaluate on all splits (train + valid + test)
docker-compose exec openserge bash scripts/run_cityscale_evaluation.sh \
  checkpoints/best_model.pt 8 all
```

#### RoadTracer Evaluation

```bash
docker-compose exec openserge bash scripts/run_roadtracer_evaluation.sh \
  checkpoints/best_model.pt 8
```

#### GlobalScale Evaluation

```bash
docker-compose exec openserge bash scripts/run_globalscale_evaluation.sh \
  checkpoints/best_model.pt 8 in-domain-test
```

### 4. Interactive Development

#### Jupyter Lab

```bash
# Start container with Jupyter Lab
docker-compose up -d

# Access at http://localhost:8888
```

#### Interactive Shell

```bash
# Open bash shell in running container
docker-compose exec openserge bash

# Or start new container with shell
docker run --gpus all -it \
  -v $(pwd)/data:/workspace/openserge/data \
  openserge:latest bash
```

### 5. TensorBoard

```bash
# Start with TensorBoard profile
docker-compose --profile tensorboard up -d

# Access at http://localhost:6007
```

### 6. Dataset Visualization

```bash
# Run visualization notebook
docker-compose exec openserge jupyter nbconvert \
  --execute --to notebook \
  notebooks/dataset_visualization.ipynb
```

## Volume Mounts

The Docker setup uses several volume mounts for data persistence:

### Required Volumes

```yaml
# Data directory (mounted read-only for safety)
-v ./data:/workspace/openserge/data:ro

# Checkpoints (read-write)
-v ./checkpoints:/workspace/openserge/checkpoints

# Logs (read-write)
-v ./logs:/workspace/openserge/logs
```

### Optional Volumes

```yaml
# Evaluation results
-v ./evaluation_results:/workspace/openserge/evaluation_results
-v ./roadtracer_evaluation_results:/workspace/openserge/roadtracer_evaluation_results

# Notebooks
-v ./notebooks:/workspace/openserge/notebooks

# Configs (read-only)
-v ./configs:/workspace/openserge/configs:ro
```

## Building Custom Images

### Build with Different Base

```dockerfile
# Edit Dockerfile to use different PyTorch version
FROM pytorch/pytorch:2.5.0-cuda12.1-cudnn9-devel
```

### Build with Additional Dependencies

```dockerfile
# Add to Dockerfile before final pip install
RUN pip install wandb pillow-simd
```

### Rebuild Image

```bash
# Full rebuild
docker-compose build --no-cache

# Quick rebuild
docker-compose build
```

## Resource Management

### Memory and CPU Limits

Edit `docker-compose.yml`:

```yaml
services:
  openserge:
    mem_limit: 64g
    cpus: 16
    shm_size: '16gb'  # Important for PyTorch DataLoaders
```

### GPU Selection

```bash
# Use specific GPU
docker run --gpus '"device=0"' openserge:latest ...

# Use multiple GPUs
docker run --gpus '"device=0,1"' openserge:latest ...

# Use all GPUs
docker run --gpus all openserge:latest ...
```

Or set in environment:

```bash
CUDA_VISIBLE_DEVICES=0 docker-compose up
```

## Troubleshooting

### GPU Not Available

**Problem**: `torch.cuda.is_available()` returns `False`

**Solution**:
```bash
# Verify NVIDIA runtime is installed
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Check Docker daemon configuration
cat /etc/docker/daemon.json
# Should include: "default-runtime": "nvidia"

# Restart Docker daemon
sudo systemctl restart docker
```

### Out of Memory

**Problem**: CUDA out of memory during training

**Solution**:
```bash
# Reduce batch size in config
{
  "batch_size": 2,  # Reduce from 4
  ...
}

# Increase shared memory
docker run --shm-size=32g ...
```

### Permission Errors

**Problem**: Cannot write to mounted volumes

**Solution**:
```bash
# Run as current user
docker run --user $(id -u):$(id -g) ...

# Or fix permissions on host
sudo chown -R $USER:$USER checkpoints/ logs/
```

### APLS Metric Fails

**Problem**: Go compilation errors in APLS metric

**Solution**:
```bash
# Rebuild Go modules inside container
docker-compose exec openserge bash -c "cd metrics/apls && go mod tidy && go build main.go"
```

### Slow Data Loading

**Problem**: Training is slow due to I/O

**Solution**:
```bash
# Use faster storage (SSD) for data volumes
# Enable preloading in config:
{
  "preload": true,  # Load entire dataset into memory
  ...
}

# Increase shared memory size
docker run --shm-size=32g ...
```

## Performance Tips

### 1. Use Local SSD for Volumes

Mount data on fast local storage (NVMe SSD) rather than network drives.

### 2. Enable Preloading

For small datasets, enable preloading in config:

```json
{
  "preload": true
}
```

### 3. Persistent Workers

Already enabled in code via `persistent_workers=True`.

### 4. Mixed Precision Training

Edit config to use automatic mixed precision:

```json
{
  "use_amp": true
}
```

### 5. Increase Shared Memory

Important for DataLoader workers:

```bash
docker run --shm-size=16gb ...
```

## Multi-GPU Training

### DataParallel (Single Node)

```bash
# Automatically uses all available GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 docker-compose exec openserge \
  python -m openserge.train --config configs/cityscale.json
```

### Manual GPU Selection

```bash
# Use only GPU 0 and 1
CUDA_VISIBLE_DEVICES=0,1 docker-compose exec openserge \
  python -m openserge.train --config configs/cityscale.json
```

## Maintenance

### Clean Up

```bash
# Remove stopped containers
docker-compose down

# Remove images
docker rmi openserge:latest

# Clean up volumes (CAUTION: deletes data!)
docker-compose down -v

# Clean up everything Docker (nuclear option)
docker system prune -a --volumes
```

### Update Image

```bash
# Pull latest base image
docker pull pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Rebuild
docker-compose build --pull
```

### View Logs

```bash
# Follow logs
docker-compose logs -f openserge

# Last 100 lines
docker-compose logs --tail=100 openserge
```

## Production Deployment

### Multi-Stage Build (Smaller Image)

Create `Dockerfile.prod`:

```dockerfile
# Builder stage
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel as builder
WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Runtime stage
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*
...
```

### Health Checks

Already included in Dockerfile:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1
```

### Security

```bash
# Run as non-root user
docker run --user 1000:1000 ...

# Read-only filesystem (where possible)
docker run --read-only -v /tmp:/tmp ...

# Drop capabilities
docker run --cap-drop=ALL --cap-add=SYS_NICE ...
```

## FAQ

**Q: Can I use CPU-only?**

A: Yes, but it will be very slow. Remove `--gpus all` flag and use CPU-based PyTorch image.

**Q: How much disk space do I need?**

A: ~20GB for image, plus dataset size and checkpoints (varies by dataset).

**Q: Can I use this on Windows?**

A: Yes, with Docker Desktop and WSL2. GPU support requires Windows 11 + WSL2 + NVIDIA drivers.

**Q: Can I use this on macOS?**

A: Yes for CPU/MPS, but CUDA is not available. Use `--device mps` for Apple Silicon GPUs.

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)
- [OpenSERGE GitHub](https://github.com/yourusername/OpenSERGE)
