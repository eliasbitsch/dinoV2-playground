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
            'angular_speed',
            default_value='0.5',
            description='Angular velocity (smaller = larger circle)'
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
                '/dino_debug@sensor_msgs/msg/Image]gz.msgs.Image',
                '/dino_reference@sensor_msgs/msg/Image]gz.msgs.Image',
            ],
            output='screen'
        ),

        # Launch circle flight node
        Node(
            package='jet_commander',
            executable='circle_flight',
            name='circle_flight',
            output='screen',
            parameters=[{
                'linear_speed': LaunchConfiguration('linear_speed'),
                'angular_speed': LaunchConfiguration('angular_speed'),
            }]
        ),
    ])
