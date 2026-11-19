#!/usr/bin/env python3
"""Print logical-joint targets published by the SuperMimic ROS2 node."""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory


class Monitor(Node):
    def __init__(self):
        super().__init__("supermimic_command_monitor")
        self.count = 0
        self.create_subscription(
            JointTrajectory,
            "/whole_body_controller/joint_trajectory",
            self.on_command,
            10,
        )

    def on_command(self, msg):
        if not msg.points:
            return
        self.count += 1
        if self.count % 50 == 0:
            point = msg.points[0]
            self.get_logger().info(
                f"commands={self.count} joints={len(msg.joint_names)} "
                f"range=({min(point.positions):.3f},{max(point.positions):.3f})"
            )


def main():
    rclpy.init()
    node = Monitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
