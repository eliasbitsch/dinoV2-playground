from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'linear_speed',
            default_value='2.0',
            description='Forward speed of the jet'
        ),
        DeclareLaunchArgument(
            'kp',
            default_value='0.005',
            description='Proportional gain for PI angular control'
        ),
        DeclareLaunchArgument(
            'ki',
            default_value='0.001',
            description='Integral gain for PI angular control'
        ),
        DeclareLaunchArgument(
            'world',
            default_value='/workspace/world/sonoma.sdf',
            description='Path to world file'
        ),
        DeclareLaunchArgument(
            'gui_config',
            default_value='/root/.gz/sim/8/gui.config',
            description='Path to Gazebo GUI config file'
        ),

        # Launch Gazebo with saved GUI config
        ExecuteProcess(
            cmd=[
                'gz', 'sim',
                '--gui-config', LaunchConfiguration('gui_config'),
                LaunchConfiguration('world')
            ],
            output='screen'
        ),

        # ROS-Gazebo bridge for cmd_vel and camera
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='ros_gz_bridge',
            arguments=[
                '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                '/target_cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                '/front_camera@sensor_msgs/msg/Image@gz.msgs.Image',
                # ROS to Gazebo bridges for visualization
                '/dino_debug@sensor_msgs/msg/Image]gz.msgs.Image',
                '/dino_reference@sensor_msgs/msg/Image]gz.msgs.Image',
            ],
            output='screen'
        ),

        # DINOv2 tracker
        Node(
            package='jet_commander',
            executable='dino_tracker',
            name='dino_tracker',
            output='screen',
        ),

        # Target follower controller
        Node(
            package='jet_commander',
            executable='target_follower',
            name='target_follower',
            output='screen',
            parameters=[{
                'linear_speed': LaunchConfiguration('linear_speed'),
                'kp': LaunchConfiguration('kp'),
                'ki': LaunchConfiguration('ki'),
                'image_width': 320,  # Must match camera width in world file
            }]
        ),

        # Target mover - makes target jet zig-zag
        Node(
            package='jet_commander',
            executable='target_mover',
            name='target_mover',
            output='screen',
            parameters=[{
                'linear_speed': 1.5,
                'angular_amplitude': 0.3,
                'zigzag_period': 4.0,
            }]
        ),
    ])
