#!/usr/bin/env python3

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


JOINTS = [
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_yaw",
    "left_wrist_roll",
    "left_wrist_pitch",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_yaw",
    "right_wrist_roll",
    "right_wrist_pitch",
]


def quat_inverse_rotate(
    q: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    w, x, y, z = q

    qc = np.array(
        [w, -x, -y, -z],
        dtype=np.float32,
    )

    vq = np.array(
        [0.0, v[0], v[1], v[2]],
        dtype=np.float32,
    )

    def mul(a, b):
        return np.array(
            [
                a[0] * b[0]
                - a[1] * b[1]
                - a[2] * b[2]
                - a[3] * b[3],
                a[0] * b[1]
                + a[1] * b[0]
                + a[2] * b[3]
                - a[3] * b[2],
                a[0] * b[2]
                - a[1] * b[3]
                + a[2] * b[0]
                + a[3] * b[1],
                a[0] * b[3]
                + a[1] * b[2]
                - a[2] * b[1]
                + a[3] * b[0],
            ],
            dtype=np.float32,
        )

    return mul(mul(qc, vq), q)[1:]


def load_joint_limits(
    xml_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if not xml_path.exists():
        raise FileNotFoundError(
            f"A00 joint-limit XML not found: {xml_path}"
        )

    root = ET.parse(xml_path).getroot()

    limits = {}

    for joint in root.iter("joint"):
        name = joint.attrib.get("name")
        joint_range = joint.attrib.get("range")

        if name is None or joint_range is None:
            continue

        logical_name = name.removesuffix("_joint")
        lower, upper = [
            float(value)
            for value in joint_range.split()
        ]

        limits[logical_name] = (lower, upper)

    missing = [
        name
        for name in JOINTS
        if name not in limits
    ]

    if missing:
        raise RuntimeError(
            f"Missing A00 joint limits for: {missing}"
        )

    lower = np.array(
        [limits[name][0] for name in JOINTS],
        dtype=np.float32,
    )

    upper = np.array(
        [limits[name][1] for name in JOINTS],
        dtype=np.float32,
    )

    return lower, upper


class SuperMimicCore:
    def __init__(
        self,
        root: str | Path,
        default_joint_pos_path: str | Path = (
            "assets/default_joint_pos.npy"
        ),
        goals_path: str | Path = (
            "assets/walk2_subject3_goals.npz"
        ),
        policy_path: str | Path = (
            "assets/policy.onnx"
        ),
        joint_limit_xml: str | Path = (
            "assets/A00_robot_mujoco_actual_used.xml"
        ),
        action_scale: float = 0.18,
        action_clip: float = 5.0,
    ):
        root = Path(root)

        default_path = Path(default_joint_pos_path)
        if not default_path.is_absolute():
            default_path = root / default_path

        goals_file = Path(goals_path)
        if not goals_file.is_absolute():
            goals_file = root / goals_file

        policy_file = Path(policy_path)
        if not policy_file.is_absolute():
            policy_file = root / policy_file

        xml_file = Path(joint_limit_xml)
        if not xml_file.is_absolute():
            xml_file = root / xml_file

        self.default = np.load(default_path).astype(
            np.float32
        )

        if self.default.shape != (29,):
            raise ValueError(
                f"default_joint_pos must have shape (29,), "
                f"got {self.default.shape}"
            )

        with np.load(goals_file) as data:
            self.goals = data["goal"].astype(
                np.float32
            )

        if self.goals.ndim != 2:
            raise ValueError(
                f"goals must be 2D, got {self.goals.shape}"
            )

        if self.goals.shape[1] != 680:
            raise ValueError(
                f"goals must have final dimension 680, "
                f"got {self.goals.shape}"
            )

        self.lower, self.upper = load_joint_limits(
            xml_file
        )

        self.action_scale = float(action_scale)
        self.action_clip = float(action_clip)

        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(policy_file),
            providers=["CPUExecutionProvider"],
        )

        inputs = {
            item.shape[-1]: item.name
            for item in self.session.get_inputs()
        }

        if 930 not in inputs:
            raise RuntimeError(
                f"ONNX policy has no 930-dim input: {inputs}"
            )

        if 680 not in inputs:
            raise RuntimeError(
                f"ONNX policy has no 680-dim input: {inputs}"
            )

        output_shape = self.session.get_outputs()[0].shape

        if output_shape[-1] != 29:
            raise RuntimeError(
                f"ONNX policy output must be 29, "
                f"got {output_shape}"
            )

        self.obs_name = inputs[930]
        self.goal_name = inputs[680]
        self.output_name = (
            self.session.get_outputs()[0].name
        )

        self.history = np.zeros(
            (10, 93),
            dtype=np.float32,
        )

        self.previous_action = np.zeros(
            29,
            dtype=np.float32,
        )

        self.frame = 0

    def reset(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        ang_vel_b: np.ndarray,
        gravity: np.ndarray,
    ):
        frame = np.concatenate(
            (
                joint_pos - self.default,
                joint_vel,
                ang_vel_b,
                gravity,
                self.previous_action,
            )
        ).astype(np.float32)

        if frame.shape != (93,):
            raise ValueError(
                f"single observation must have shape (93,), "
                f"got {frame.shape}"
            )

        self.history[:] = frame
        self.frame = 0

    def step(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        ang_vel_b: np.ndarray,
        gravity: np.ndarray,
    ):
        frame = np.concatenate(
            (
                joint_pos - self.default,
                joint_vel,
                ang_vel_b,
                gravity,
                self.previous_action,
            )
        ).astype(np.float32)

        if frame.shape != (93,):
            raise ValueError(
                f"single observation must have shape (93,), "
                f"got {frame.shape}"
            )

        self.history[:-1] = self.history[1:]
        self.history[-1] = frame

        chunks = []
        start = 0

        for size in (29, 29, 3, 3, 29):
            chunks.append(
                self.history[
                    :,
                    start:start + size,
                ].reshape(-1)
            )
            start += size

        obs = np.concatenate(chunks).astype(
            np.float32
        )

        goal = self.goals[
            min(self.frame, len(self.goals) - 1)
        ].astype(np.float32)

        result = self.session.run(
            [self.output_name],
            {
                self.obs_name: obs[None],
                self.goal_name: goal[None],
            },
        )

        action = result[0][0].astype(
            np.float32
        )

        if not np.isfinite(action).all():
            raise RuntimeError(
                f"Policy output contains NaN or Inf "
                f"at frame {self.frame}"
            )

        action = np.clip(
            action,
            -self.action_clip,
            self.action_clip,
        )

        target = (
            self.default
            + self.action_scale * action
        )

        unclipped_target = target.copy()

        target = np.clip(
            target,
            self.lower,
            self.upper,
        )

        self.previous_action = action.copy()
        self.frame += 1

        return action, target, unclipped_target
