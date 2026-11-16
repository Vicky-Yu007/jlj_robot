#!/usr/bin/env python3
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supermimic_deploy.goal_builder import build_goals
from supermimic_deploy.observation import ObservationHistory
from supermimic_deploy.onnx_policy import SuperMimicONNXPolicy


def quat_rotate_inverse(q, v):
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    qc = q.copy()
    qc[1:] *= -1.0
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    def mul(a, b):
        return np.array([
            a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
            a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
            a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
            a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0],
        ])
    return mul(mul(qc, vq), q)[1:]


def main():
    motion = np.load(ROOT / "assets/walk2_subject3.npz")
    default = np.load(ROOT / "assets/default_joint_pos.npy").astype(np.float32)
    goals = np.load(ROOT / "assets/walk2_subject3_goals.npz")["goal"]

    body_names = [str(x) for x in motion["body_names"]]
    pelvis = body_names.index("pelvis")

    history = ObservationHistory(default)
    obs = history.update(
        motion["joint_pos"][0],
        motion["joint_vel"][0],
        motion["body_ang_vel_w"][0, pelvis],
        quat_rotate_inverse(motion["body_quat_w"][0, pelvis], np.array([0.0, 0.0, -1.0], dtype=np.float32)),
        np.zeros(29, dtype=np.float32),
    )

    policy = SuperMimicONNXPolicy(ROOT / "assets/policy.onnx")
    action = policy(obs, goals[0])

    print("obs shape:", obs.shape)
    print("goal shape:", goals[0].shape)
    print("action shape:", action.shape)
    print("obs finite:", np.isfinite(obs).all())
    print("goal finite:", np.isfinite(goals[0]).all())
    print("action finite:", np.isfinite(action).all())
    print("action min/max:", float(action.min()), float(action.max()))
    print("action norm:", float(np.linalg.norm(action)))
    print()
    print("index, joint_name, raw_action, target_offset")
    for i, (name, value) in enumerate(zip(motion["joint_names"], action)):
        print(f"{i:02d}, {name}, {value:+.6f}, {0.18 * value:+.6f}")


if __name__ == "__main__":
    main()
