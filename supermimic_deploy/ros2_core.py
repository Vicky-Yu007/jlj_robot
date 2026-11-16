#!/usr/bin/env python3
"""SuperMimic ROS2 logical-joint deployment node.

Default mode is shadow mode. It computes and logs targets but does not publish
commands. The hardware stack remains the sibling repository's ros2_control and
parallel-mechanism adapter.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


JOINTS = [
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll", "left_hip_pitch",
    "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch",
    "left_ankle_roll", "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_yaw", "left_wrist_roll", "left_wrist_pitch",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_yaw", "right_wrist_roll",
    "right_wrist_pitch",
]


def quat_inverse_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    qc = np.array([w, -x, -y, -z], dtype=np.float32)
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)

    def mul(a, b):
        return np.array([
            a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
            a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
            a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
            a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
        ], dtype=np.float32)

    return mul(mul(qc, vq), q)[1:]


class SuperMimicCore:
    def __init__(self, root: str | Path):
        root = Path(root)
        self.default = np.load(root / "assets/default_joint_pos.npy").astype(np.float32)
        self.goals = np.load(root / "assets/walk2_subject3_goals.npz")["goal"]
        import onnxruntime as ort
        self.session = ort.InferenceSession(
            str(root / "assets/policy.onnx"),
            providers=["CPUExecutionProvider"],
        )
        inputs = {x.shape[-1]: x.name for x in self.session.get_inputs()}
        self.obs_name = inputs[930]
        self.goal_name = inputs[680]
        self.output_name = self.session.get_outputs()[0].name
        self.history = np.zeros((10, 93), dtype=np.float32)
        self.previous_action = np.zeros(29, dtype=np.float32)
        self.frame = 0

    def reset(self, joint_pos, joint_vel, ang_vel_b, gravity):
        frame = np.concatenate((
            joint_pos - self.default,
            joint_vel,
            ang_vel_b,
            gravity,
            self.previous_action,
        )).astype(np.float32)
        self.history[:] = frame
        self.frame = 0

    def step(self, joint_pos, joint_vel, ang_vel_b, gravity):
        frame = np.concatenate((
            joint_pos - self.default,
            joint_vel,
            ang_vel_b,
            gravity,
            self.previous_action,
        )).astype(np.float32)
        self.history[:-1] = self.history[1:]
        self.history[-1] = frame

        chunks = []
        start = 0
        for size in (29, 29, 3, 3, 29):
            chunks.append(self.history[:, start:start + size].reshape(-1))
            start += size
        obs = np.concatenate(chunks).astype(np.float32)
        goal = self.goals[min(self.frame, len(self.goals) - 1)]
        action = self.session.run(
            [self.output_name],
            {self.obs_name: obs[None], self.goal_name: goal[None]},
        )[0][0].astype(np.float32)
        action = np.clip(action, -5.0, 5.0)
        target = self.default + 0.18 * action
        self.previous_action = action
        self.frame += 1
        return action, target
