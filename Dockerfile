# GPU version with CUDA support
# For CPU-only, use Dockerfile.cpu instead
FROM python:3.11-slim-bookworm

WORKDIR /app

# Set environment variables for persistent model caching
ENV TORCH_HOME=/app/models
ENV HF_HOME=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for user data and model cache
RUN mkdir -p /app/workspace /app/models /app/images

# Expose Jupyter port
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
