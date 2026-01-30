#!/usr/bin/env python3
"""
Target follower node - steers the jet to follow a tracked target.
Uses tracked_position from dino_tracker to control the jet.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point


class TargetFollower(Node):
    def __init__(self):
        super().__init__('target_follower')

        # Parameters
        self.declare_parameter('linear_speed', 3.0)
        self.declare_parameter('kp', 0.0003)  # Proportional gain
        self.declare_parameter('max_angular', 0.1)  # Max angular velocity
        self.declare_parameter('image_width', 320)  # Must match camera width in world file
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('timeout', 1.0)  # seconds without tracking before stopping

        # Force float conversion (LaunchConfiguration passes strings)
        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.kp = float(self.get_parameter('kp').value)
        self.max_angular = float(self.get_parameter('max_angular').value)
        self.image_width = float(self.get_parameter('image_width').value)
        publish_rate = float(self.get_parameter('publish_rate').value)
        self.timeout = float(self.get_parameter('timeout').value)

        # State
        self.last_target_x = None
        self.last_tracking_time = None
        self.msg_count = 0

        # Publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.position_sub = self.create_subscription(
            Point,
            '/tracked_position',
            self.position_callback,
            10
        )

        # Control loop timer
        self.timer = self.create_timer(1.0 / publish_rate, self.control_loop)

        self.get_logger().info(
            f'Target follower started: kp={self.kp}, max_angular={self.max_angular}'
        )

    def position_callback(self, msg):
        """Receive tracked position from dino_tracker."""
        self.last_target_x = msg.x
        self.last_tracking_time = self.get_clock().now()
        self.msg_count += 1
        if self.msg_count % 10 == 1:  # Log every 10th message
            self.get_logger().info(f'Received position: x={msg.x:.1f}, conf={msg.z:.3f}')

    def control_loop(self):
        """Main control loop - compute and publish velocity commands."""
        msg = Twist()

        # Check if we have recent tracking data
        if self.last_tracking_time is not None:
            time_since_tracking = (
                self.get_clock().now() - self.last_tracking_time
            ).nanoseconds / 1e9

            if time_since_tracking < self.timeout and self.last_target_x is not None:
                # Calculate error from center of image
                image_center = self.image_width / 2.0
                error = image_center - self.last_target_x

                # P control: angular velocity proportional to error
                angular = self.kp * error

                # Clamp to max angular velocity
                angular = max(-self.max_angular, min(self.max_angular, angular))

                msg.linear.x = self.linear_speed
                msg.angular.z = angular

                self.get_logger().info(f'P control: error={error:.1f}, angular={angular:.3f}')
            else:
                # No recent tracking - go straight
                msg.linear.x = self.linear_speed
                msg.angular.z = 0.0
                self.get_logger().info(f'Lost target - going straight')
        else:
            # No tracking data yet - just move forward slowly
            msg.linear.x = self.linear_speed * 0.3
            msg.angular.z = 0.0
            self.get_logger().info('No tracking data received yet')

        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TargetFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot
        stop_msg = Twist()
        node.cmd_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
