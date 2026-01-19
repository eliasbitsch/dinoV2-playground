#!/bin/bash

# DINOv2 Playground Docker Run Script

IMAGE_NAME="dinov2-playground"
CONTAINER_NAME="dinov2-playground-container"

# Get the absolute path of the current directory
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==================================="
echo "DINOv2 Playground Docker Setup"
echo "==================================="
echo "Workspace: $WORKSPACE_DIR"
echo ""

# Ensure required folders exist
mkdir -p "$WORKSPACE_DIR/workspace"
mkdir -p "$WORKSPACE_DIR/models"
mkdir -p "$WORKSPACE_DIR/images"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Build Docker image
echo "📦 Building Docker image..."
docker build -t $IMAGE_NAME "$WORKSPACE_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Failed to build Docker image"
    exit 1
fi

echo "✓ Docker image built successfully"
echo ""

# Stop and remove existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🧹 Removing existing container..."
    docker rm -f $CONTAINER_NAME
fi

# Check if NVIDIA GPU is available
GPU_FLAGS=""
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 NVIDIA GPU detected - enabling GPU support"
    GPU_FLAGS="--gpus all"
else
    echo "⚠️  No NVIDIA GPU detected - running in CPU mode"
    echo "   (CUDA libraries in image won't be used)"
fi

# Run the container
echo "🚀 Starting container..."
docker run -d \
    --name $CONTAINER_NAME \
    $GPU_FLAGS \
    -p 8888:8888 \
    -v "$WORKSPACE_DIR:/app" \
    -v "$WORKSPACE_DIR/workspace:/app/workspace" \
    -v "$WORKSPACE_DIR/images:/app/images" \
    -v "$WORKSPACE_DIR/models:/app/models" \
    $IMAGE_NAME

if [ $? -ne 0 ]; then
    echo "❌ Failed to start container"
    exit 1
fi

echo ""
echo "==================================="
echo "✓ Container started successfully!"
echo "==================================="
echo ""
echo "📓 Jupyter Notebook is running at:"
echo "   http://localhost:8888"
echo ""
echo "📂 Mounted folders:"
echo "   /app          → Project root"
echo "   /app/workspace → Outputs and data"
echo "   /app/images   → Images (persistent)"
echo "   /app/models   → DINOv2 models (cached)"
echo ""
echo "💡 CPU-only users: The image works fine without GPU!"
echo "💾 Models will be downloaded once and cached in models/"
echo ""
echo "Useful commands:"
echo "  View logs:    docker logs -f $CONTAINER_NAME"
echo "  Stop:         docker stop $CONTAINER_NAME"
echo "  Start again:  docker start $CONTAINER_NAME"
echo "  Shell access: docker exec -it $CONTAINER_NAME /bin/bash"
echo "  Remove:       docker rm -f $CONTAINER_NAME"
echo ""
