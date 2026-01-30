#!/bin/bash

# DINOv2 Playground - Auto-detects GPU and uses appropriate image

set -e

CONTAINER_NAME="dinov2-playground"

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "GPU detected - using CUDA image"
    PROFILE="gpu"
else
    echo "No GPU detected - using lightweight CPU image"
    PROFILE="cpu"
fi

# Run with docker compose
docker compose --profile "$PROFILE" up --build
