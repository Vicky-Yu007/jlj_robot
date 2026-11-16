#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from supermimic_deploy.ros2_core import JOINTS, SuperMimicCore, quat_inverse_rotate


class SuperMimicRos2Node(Node):
    def __init__(self):
        super().__init__("supermimic_deploy_node")
        self.declare_parameter("asset_root", str(Path.cwd()))
        self.declare_parameter("shadow_mode", True)
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("imu_topic", "/imu/data")
        self.declare_parameter(
            "command_topic",
            "/whole_body_controller/joint_trajectory",
        )

        root = self.get_parameter("asset_root").value
        self.shadow_mode = bool(self.get_parameter("shadow_mode").value)
        self.core = SuperMimicCore(root)

        self.lock = threading.Lock()
        self.joint_pos = None
        self.joint_vel = None
        self.imu = None
        self.initialized = False
        self.last_log_ns = 0

        self.create_subscription(
            JointState,
            self.get_parameter("joint_state_topic").value,
            self.on_joint_state,
            10,
        )
        self.create_subscription(
            Imu,
            self.get_parameter("imu_topic").value,
            self.on_imu,
            10,
        )
        self.command_pub = self.create_publisher(
            JointTrajectory,
            self.get_parameter("command_topic").value,
            10,
        )
        self.create_timer(0.02, self.control_step)
        self.get_logger().info(
            f"SuperMimic node started, shadow_mode={self.shadow_mode}"
        )

    def on_joint_state(self, msg: JointState):
        pos = dict(zip(msg.name, msg.position))
        vel = dict(zip(msg.name, msg.velocity))
        if not all(name in pos for name in JOINTS):
            return
        with self.lock:
            self.joint_pos = np.array([pos[name] for name in JOINTS], dtype=np.float32)
            self.joint_vel = np.array([vel.get(name, 0.0) for name in JOINTS], dtype=np.float32)

    def on_imu(self, msg: Imu):
        with self.lock:
            self.imu = msg

    def control_step(self):
        with self.lock:
            if self.joint_pos is None or self.joint_vel is None or self.imu is None:
                return
            joint_pos = self.joint_pos.copy()
            joint_vel = self.joint_vel.copy()
            imu = self.imu

        q = np.array(
            [
                imu.orientation.w,
                imu.orientation.x,
                imu.orientation.y,
                imu.orientation.z,
            ],
            dtype=np.float32,
        )
        gravity = quat_inverse_rotate(q, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        ang_vel_b = quat_inverse_rotate(
            q,
            np.array(
                [
                    imu.angular_velocity.x,
                    imu.angular_velocity.y,
                    imu.angular_velocity.z,
                ],
                dtype=np.float32,
            ),
        )

        if not self.initialized:
            self.core.reset(joint_pos, joint_vel, ang_vel_b, gravity)
            self.initialized = True

        action, target = self.core.step(
            joint_pos,
            joint_vel,
            ang_vel_b,
            gravity,
        )

        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_log_ns > 1_000_000_000:
            self.get_logger().info(
                f"frame={self.core.frame} action_norm={np.linalg.norm(action):.3f} "
                f"target_range=({target.min():.3f},{target.max():.3f})"
            )
            self.last_log_ns = now_ns

        if self.shadow_mode:
            return

        command = JointTrajectory()
        command.joint_names = list(JOINTS)
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in target]
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 20_000_000
        command.points = [point]
        self.command_pub.publish(command)


def main():
    rclpy.init()
    node = SuperMimicRos2Node()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
