#!/bin/bash
# Start Gazebo Harmonic simulation container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Allow X11 connections
xhost +local:docker 2>/dev/null || true

# Check if container is already running
if docker compose ps 2>/dev/null | grep -q "gazebo-sim.*Up"; then
    echo "Container already running. Entering..."
    docker compose exec gazebo-sim bash
    exit 0
fi

# Build if image doesn't exist
if ! docker images | grep -q "gazebo-harmonic-sim"; then
    echo "Building Docker image..."
    docker compose build
fi

# Start container and enter
echo "Starting Gazebo simulation container..."
docker compose up -d
docker compose exec gazebo-sim bash
