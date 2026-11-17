#!/usr/bin/env python3

from __future__ import annotations

import time
from pathlib import Path
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from trajectory_msgs.msg import (
    JointTrajectory,
    JointTrajectoryPoint,
)

from supermimic_deploy.ros2_core import (
    JOINTS,
    SuperMimicCore,
)


class SuperMimicRos2Node(Node):
    def __init__(self):
        super().__init__(
            "supermimic_deploy_node"
        )

        self.declare_parameter(
            "asset_root",
            str(Path.cwd()),
        )

        self.declare_parameter(
            "shadow_mode",
            True,
        )

        self.declare_parameter(
            "joint_state_topic",
            "/joint_states",
        )

        self.declare_parameter(
            "imu_topic",
            "/imu/data",
        )

        self.declare_parameter(
            "command_topic",
            "/whole_body_controller/"
            "joint_trajectory",
        )

        self.declare_parameter(
            "sensor_timeout_s",
            0.1,
        )

        self.declare_parameter(
            "default_joint_pos_path",
            "assets/default_joint_pos.npy",
        )

        self.declare_parameter(
            "goals_path",
            "assets/walk2_subject3_goals.npz",
        )

        self.declare_parameter(
            "policy_path",
            "assets/policy.onnx",
        )

        self.declare_parameter(
            "joint_limit_xml",
            "assets/"
            "A00_robot_mujoco_actual_used.xml",
        )

        root = self.get_parameter(
            "asset_root"
        ).value

        self.shadow_mode = bool(
            self.get_parameter(
                "shadow_mode"
            ).value
        )

        self.sensor_timeout_ns = int(
            float(
                self.get_parameter(
                    "sensor_timeout_s"
                ).value
            ) * 1_000_000_000
        )

        self.core = SuperMimicCore(
            root=root,
            default_joint_pos_path=(
                self.get_parameter(
                    "default_joint_pos_path"
                ).value
            ),
            goals_path=(
                self.get_parameter(
                    "goals_path"
                ).value
            ),
            policy_path=(
                self.get_parameter(
                    "policy_path"
                ).value
            ),
            joint_limit_xml=(
                self.get_parameter(
                    "joint_limit_xml"
                ).value
            ),
        )

        self.lock = threading.Lock()

        self.joint_pos = None
        self.joint_vel = None
        self.imu = None

        self.joint_rx_ns = None
        self.imu_rx_ns = None

        self.initialized = False
        self.last_log_ns = 0
        self.last_stale_log_ns = 0

        self.create_subscription(
            JointState,
            self.get_parameter(
                "joint_state_topic"
            ).value,
            self.on_joint_state,
            10,
        )

        self.create_subscription(
            Imu,
            self.get_parameter(
                "imu_topic"
            ).value,
            self.on_imu,
            10,
        )

        self.command_pub = self.create_publisher(
            JointTrajectory,
            self.get_parameter(
                "command_topic"
            ).value,
            10,
        )

        self.create_timer(
            0.02,
            self.control_step,
        )

        self.get_logger().info(
            "SuperMimic node started, "
            f"shadow_mode={self.shadow_mode}, "
            f"sensor_timeout_ms="
            f"{self.sensor_timeout_ns / 1e6:.1f}"
        )

    def on_joint_state(self, msg: JointState):
        if len(msg.name) != len(msg.position):
            return

        if len(msg.name) != len(msg.velocity):
            return

        if len(set(msg.name)) != len(msg.name):
            return

        positions = np.asarray(
            msg.position,
            dtype=np.float32,
        )

        velocities = np.asarray(
            msg.velocity,
            dtype=np.float32,
        )

        if not np.isfinite(positions).all():
            return

        if not np.isfinite(velocities).all():
            return

        state = dict(
            zip(msg.name, msg.position)
        )

        velocity_state = dict(
            zip(msg.name, msg.velocity)
        )

        if not all(
            name in state
            for name in JOINTS
        ):
            return

        joint_pos = np.array(
            [
                state[name]
                for name in JOINTS
            ],
            dtype=np.float32,
        )

        joint_vel = np.array(
            [
                velocity_state.get(name, 0.0)
                for name in JOINTS
            ],
            dtype=np.float32,
        )

        with self.lock:
            self.joint_pos = joint_pos
            self.joint_vel = joint_vel
            self.joint_rx_ns = time.monotonic_ns()

    def on_imu(self, msg: Imu):
        q = np.array(
            [
                msg.orientation.w,
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
            ],
            dtype=np.float32,
        )

        gyro = np.array(
            [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ],
            dtype=np.float32,
        )

        if not np.isfinite(q).all():
            return

        if not np.isfinite(gyro).all():
            return

        if abs(float(np.linalg.norm(q)) - 1.0) > 1e-3:
            return

        with self.lock:
            self.imu = msg
            self.imu_rx_ns = time.monotonic_ns()

    def inputs_are_fresh(self):
        now_ns = time.monotonic_ns()

        joint_fresh = (
            self.joint_rx_ns is not None
            and now_ns - self.joint_rx_ns
            <= self.sensor_timeout_ns
        )

        imu_fresh = (
            self.imu_rx_ns is not None
            and now_ns - self.imu_rx_ns
            <= self.sensor_timeout_ns
        )

        return joint_fresh and imu_fresh

    def control_step(self):
        if not self.inputs_are_fresh():
            self.initialized = False

            now_ns = time.monotonic_ns()

            if (
                now_ns - self.last_stale_log_ns
                > 1_000_000_000
            ):
                self.get_logger().warning(
                    "Sensor timeout. "
                    "Policy inference and command "
                    "publishing are stopped."
                )
                self.last_stale_log_ns = now_ns

            return

        with self.lock:
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

        # ROS sensor_msgs/Imu angular_velocity 已经在
        # IMU 坐标系中表达。这里不再重复旋转。
        ang_vel_b = np.array(
            [
                imu.angular_velocity.x,
                imu.angular_velocity.y,
                imu.angular_velocity.z,
            ],
            dtype=np.float32,
        )

        from supermimic_deploy.ros2_core import (
            quat_inverse_rotate,
        )

        gravity = quat_inverse_rotate(
            q,
            np.array(
                [0.0, 0.0, -1.0],
                dtype=np.float32,
            ),
        )

        if not self.initialized:
            self.core.previous_action.fill(0.0)
            self.core.reset(
                joint_pos,
                joint_vel,
                ang_vel_b,
                gravity,
            )
            self.initialized = True

        action, target, unclipped_target = (
            self.core.step(
                joint_pos,
                joint_vel,
                ang_vel_b,
                gravity,
            )
        )

        if not np.isfinite(target).all():
            self.initialized = False
            self.get_logger().error(
                "Target contains NaN or Inf. "
                "Commands stopped."
            )
            return

        now_ns = self.get_clock().now().nanoseconds

        if (
            now_ns - self.last_log_ns
            > 1_000_000_000
        ):
            target_correction = float(
                np.max(
                    np.abs(
                        target - unclipped_target
                    )
                )
            )

            self.get_logger().info(
                f"frame={self.core.frame} "
                f"action_norm="
                f"{np.linalg.norm(action):.3f} "
                f"target_range="
                f"({target.min():.3f},"
                f"{target.max():.3f}) "
                f"limit_correction="
                f"{target_correction:.3f}"
            )

            self.last_log_ns = now_ns

        if self.shadow_mode:
            return

        command = JointTrajectory()
        command.joint_names = list(JOINTS)

        point = JointTrajectoryPoint()
        point.positions = [
            float(value)
            for value in target
        ]

        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 20_000_000

        command.points = [point]
        self.command_pub.publish(command)


def main():
    rclpy.init()
    node = SuperMimicRos2Node()

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
