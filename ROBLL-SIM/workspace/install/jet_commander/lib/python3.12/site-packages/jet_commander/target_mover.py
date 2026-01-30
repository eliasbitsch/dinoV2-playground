#!/usr/bin/env python3
"""
Target mover node - moves the target jet in a zig-zag pattern.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math


class TargetMover(Node):
    def __init__(self):
        super().__init__('target_mover')

        # Parameters
        self.declare_parameter('linear_speed', 3.0)
        self.declare_parameter('turn_amplitude', 0.1)  # Max turn rate for S-curve
        self.declare_parameter('s_curve_period', 10.0)  # Seconds for one full S-curve
        self.declare_parameter('publish_rate', 10.0)

        self.linear_speed = float(self.get_parameter('linear_speed').value)
        self.turn_amplitude = float(self.get_parameter('turn_amplitude').value)
        self.s_curve_period = float(self.get_parameter('s_curve_period').value)
        publish_rate = float(self.get_parameter('publish_rate').value)

        self.start_time = None

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/target_cmd_vel', 10)

        # Timer for control loop
        self.timer = self.create_timer(1.0 / publish_rate, self.control_loop)

        self.get_logger().info(
            f'Target mover started: speed={self.linear_speed}, '
            f'amplitude={self.turn_amplitude}, period={self.s_curve_period}s'
        )

    def control_loop(self):
        """Publish velocity commands: smooth S-curve pattern."""
        now = self.get_clock().now()

        # Initialize start time
        if self.start_time is None:
            self.start_time = now

        # Time since start
        elapsed = (now - self.start_time).nanoseconds / 1e9

        # Sinusoidal angular velocity for smooth S-curve
        angular = self.turn_amplitude * math.sin(2 * math.pi * elapsed / self.s_curve_period)

        msg = Twist()
        msg.linear.x = self.linear_speed
        msg.angular.z = angular

        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TargetMover()

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
