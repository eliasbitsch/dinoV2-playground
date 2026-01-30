# ROBLL-SIM: Gazebo Simulation with DINOv2 Target Tracking

A Gazebo Harmonic simulation environment with ROS2 Jazzy for testing DINOv2-based visual object tracking and following.

## Prerequisites

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with CUDA support
- X11 display server (for GUI)

## Quick Start

### 1. Start the Simulation Container

```bash
./start.sh
```

This script will:
- Build the Docker image if it doesn't exist
- Start the container with GPU and display access
- Drop you into an interactive shell inside the container

### 2. Build the ROS2 Workspace (first time only)

Inside the container:

```bash
cd /workspace
colcon build
source install/setup.bash
```

### 3. Launch the Full System

Run the complete simulation with target tracking and following:

```bash
ros2 launch jet_commander target_follow.launch.py
```

This launches:
- **Gazebo Harmonic** with the Sonoma raceway world
- **ROS-Gazebo bridge** for camera and velocity topics
- **DINOv2 tracker** - tracks reference object in camera feed
- **Target follower** - controls the chasing jet based on tracking
- **Target mover** - makes the target jet zig-zag

## Launch Parameters

Customize the follower behavior:

```bash
ros2 launch jet_commander target_follow.launch.py linear_speed:=3.0 kp:=0.005 ki:=0.001
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `linear_speed` | 2.0 | Forward speed of the follower jet |
| `kp` | 0.005 | Proportional gain for steering |
| `ki` | 0.001 | Integral gain for steering |
| `world` | `/workspace/world/sonoma.sdf` | World file path |

## ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/front_camera` | `sensor_msgs/Image` | Camera feed from follower jet |
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity commands for follower |
| `/target_cmd_vel` | `geometry_msgs/Twist` | Velocity commands for target |
| `/tracked_position` | `geometry_msgs/Point` | Tracked object position (x, y, confidence) |
| `/dino_debug` | `sensor_msgs/Image` | Debug visualization with heatmap |
| `/dino_reference` | `sensor_msgs/Image` | Reference image being tracked |

## Project Structure

```
ROBLL-SIM/
├── start.sh              # Container startup script
├── docker-compose.yml    # Docker configuration
├── Dockerfile            # ROS2 Jazzy + Gazebo Harmonic image
└── workspace/
    ├── world/            # Gazebo world files
    │   └── sonoma.sdf
    ├── models/           # 3D models (F-14 Tomcat)
    └── src/
        └── jet_commander/
            ├── launch/           # ROS2 launch files
            ├── jet_commander/    # Python nodes
            │   ├── dino_tracker.py     # DINOv2 tracking
            │   ├── target_follower.py  # Following controller
            │   └── target_mover.py     # Target movement
            └── images/           # Reference images
```

## Troubleshooting

### Display Issues

If Gazebo doesn't show, ensure X11 forwarding is enabled:

```bash
xhost +local:docker
```

### GPU Not Detected

Verify NVIDIA container toolkit:

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Rebuilding After Changes

```bash
cd /workspace
colcon build --packages-select jet_commander
source install/setup.bash
```
