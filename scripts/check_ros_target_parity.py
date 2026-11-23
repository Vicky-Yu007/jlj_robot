"""Offline Humble command-path check using cloud observations (no robot topics)."""
from pathlib import Path
import sys

import numpy as np
import rclpy
from sensor_msgs.msg import Imu, JointState

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from supermimic_deploy.ros2_core import JOINTS, SuperMimicCore
from supermimic_ros2_node import SuperMimicRos2Node


class Recorder:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def main(path):
    sim = SuperMimicCore(ROOT, use_raw_policy_action=True)
    with np.load(path) as trace:
        obs, actions, goals = (trace[k] for k in ("policy_obs", "action", "goal"))
    count = len(obs)
    if not count or obs.shape != (count, 930) or actions.shape != (count, 29):
        raise ValueError("Expected a continuous trace from reset with 930 observations and 29 actions")
    np.testing.assert_allclose(goals, sim.goals[1:count + 1], atol=1e-5, rtol=0)
    rclpy.init(args=["--ros-args", "-p", f"asset_root:={ROOT}",
                     "-p", "use_raw_policy_action:=true", "-p", "shadow_mode:=false",
                     "-p", "command_topic:=/offline_parity/unused",
                     "-p", "imu_topic:=/offline_parity/imu",
                     "-p", "joint_state_topic:=/offline_parity/joints"])
    node = SuperMimicRos2Node()
    record = Recorder()
    node.command_pub = record  # Exercise message assembly without sending to DDS.
    worst_pair = worst_cloud = 0.
    try:
        for i, row in enumerate(obs):
            q = row[261:290] + sim.default
            v, gyro, gravity = row[551:580], row[607:610], row[637:640]
            if i == 0:
                sim.reset(q, v, gyro, gravity)
            _, target, _ = sim.step(q, v, gyro, gravity)
            joints = JointState()
            joints.name = list(reversed(JOINTS))
            joints.position = q[::-1].astype(float).tolist()
            joints.velocity = v[::-1].astype(float).tolist()
            imu = Imu()
            # Reconstruct any orientation with this body-frame gravity (yaw is unobserved).
            g = gravity.astype(float) / np.linalg.norm(gravity)
            quat = np.array([1 - g[2], -g[1], g[0], 0.])
            if np.linalg.norm(quat) < 1e-10:
                quat = np.array([0., 1., 0., 0.])
            quat /= np.linalg.norm(quat)
            imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z = quat.tolist()
            imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = gyro.astype(float).tolist()
            node.on_joint_state(joints)
            node.on_imu(imu)
            node.control_step()
            if len(record.messages) != i + 1:
                raise RuntimeError(f"Missing command at step {i}")
            msg = record.messages[-1]
            if list(msg.joint_names) != JOINTS:
                raise ValueError("Command joint order differs from policy order")
            actual = np.array(msg.points[0].positions)
            expected = np.clip(sim.default + sim.action_scale * actions[i], sim.lower, sim.upper)
            for reference in (target, expected):
                np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=0, equal_nan=False)
            worst_pair = max(worst_pair, np.max(abs(actual - target)))
            worst_cloud = max(worst_cloud, np.max(abs(actual - expected)))
        print(f"PASS: {count} x 29 targets, ROS vs simulation {worst_pair:.3g} rad, "
              f"ROS vs cloud-derived targets {worst_cloud:.3g} rad")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
