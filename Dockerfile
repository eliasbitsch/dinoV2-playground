# Use PyTorch base image (works for both CPU and GPU)
# GPU support is enabled via --gpus flag at runtime
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Set environment variables for persistent model caching
ENV TORCH_HOME=/app/models
ENV HF_HOME=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for user data and model cache
RUN mkdir -p /app/workspace /app/models /app/images

# Expose Jupyter port
EXPOSE 8888

# Set up Jupyter to accept connections from any IP
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
