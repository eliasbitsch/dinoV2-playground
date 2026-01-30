# New Gazebo (Harmonic) with ROS2 Jazzy
FROM osrf/ros:jazzy-desktop-full

ENV DEBIAN_FRONTEND=noninteractive

# Install Gazebo Harmonic and ROS2-Gazebo bridge
RUN apt-get update && apt-get install -y \
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-ros-gz-image \
    ros-jazzy-ros-gz-sim \
    ros-jazzy-ros-gz-sim-demos \
    # Utilities
    wget \
    curl \
    git \
    vim \
    x11-apps \
    mesa-utils \
    # GPU/Graphics support
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-dri \
    sudo \
    # Python packages
    python3-pip \
    python3-opencv \
    ros-jazzy-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip3 install --break-system-packages \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# Install other ML dependencies
RUN pip3 install --break-system-packages \
    matplotlib \
    pillow

# Create workspace directory
WORKDIR /workspace

# Set up environment in global bashrc
RUN echo "source /opt/ros/jazzy/setup.bash" >> /etc/bash.bashrc \
    && echo "export GZ_SIM_RESOURCE_PATH=/workspace/models:/workspace/worlds" >> /etc/bash.bashrc

ENV GZ_SIM_RESOURCE_PATH=/workspace/models:/workspace/worlds

# Entrypoint
RUN echo '#!/bin/bash\n\
set -e\n\
source /opt/ros/jazzy/setup.bash\n\
if [ -f /workspace/install/setup.bash ]; then\n\
    source /workspace/install/setup.bash\n\
fi\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
