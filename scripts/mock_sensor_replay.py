#!/usr/bin/env python3
"""Replay walk2_subject3 as ROS2 joint and IMU sensor topics.

This is a no-hardware test source. It publishes the same logical joint
names and body-frame IMU quantities expected by the deployment node.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState


JOINTS = [
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll", "left_hip_pitch",
    "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch",
    "left_ankle_roll", "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_yaw", "left_wrist_roll", "left_wrist_pitch",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_yaw", "right_wrist_roll", "right_wrist_pitch",
]


def rotate_inverse(q, v):
    w, x, y, z = q
    qc = np.array([w, -x, -y, -z], dtype=np.float32)
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)

    def mul(a, b):
        return np.array([
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ], dtype=np.float32)

    return mul(mul(qc, vq), q)[1:]


class ReplayNode(Node):
    def __init__(self, asset_root: Path):
        super().__init__("supermimic_mock_sensor_replay")
        self.motion = np.load(asset_root / "assets/walk2_subject3.npz")
        names = [str(x) for x in self.motion["joint_names"]]
        if names != [f"{name}_joint" for name in JOINTS]:
            raise RuntimeError("trajectory joint order does not match deployment order")
        self.body_names = [str(x) for x in self.motion["body_names"]]
        self.pelvis = self.body_names.index("pelvis")
        self.frame = 0
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.imu_pub = self.create_publisher(Imu, "/imu/data", 10)
        self.create_timer(0.02, self.publish_frame)
        self.get_logger().info("Mock sensor replay started at 50 Hz")

    def publish_frame(self):
        i = min(self.frame, len(self.motion["joint_pos"]) - 1)
        q = self.motion["body_quat_w"][i, self.pelvis].astype(np.float32)
        ang_b = rotate_inverse(q, self.motion["body_ang_vel_w"][i, self.pelvis])
        gravity = rotate_inverse(q, np.array([0.0, 0.0, -1.0], dtype=np.float32))

        joint = JointState()
        joint.header.stamp = self.get_clock().now().to_msg()
        joint.name = list(JOINTS)
        joint.position = [float(x) for x in self.motion["joint_pos"][i]]
        joint.velocity = [float(x) for x in self.motion["joint_vel"][i]]
        self.joint_pub.publish(joint)

        imu = Imu()
        imu.header = joint.header
        imu.orientation.w = float(q[0])
        imu.orientation.x = float(q[1])
        imu.orientation.y = float(q[2])
        imu.orientation.z = float(q[3])
        imu.angular_velocity.x = float(ang_b[0])
        imu.angular_velocity.y = float(ang_b[1])
        imu.angular_velocity.z = float(ang_b[2])
        imu.linear_acceleration.x = float(gravity[0])
        imu.linear_acceleration.y = float(gravity[1])
        imu.linear_acceleration.z = float(gravity[2])
        self.imu_pub.publish(imu)

        self.frame += 1
        if self.frame >= len(self.motion["joint_pos"]):
            self.frame = 0


def main():
    rclpy.init()
    root = Path.cwd()
    node = ReplayNode(root)
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
