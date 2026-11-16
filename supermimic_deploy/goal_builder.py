from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (aw * bw - ax * bx - ay * by - az * bz,
         aw * bx + ax * bw + ay * bz - az * by,
         aw * by - ax * bz + ay * bw + az * bx,
         aw * bz + ax * by - ay * bx + az * bw),
        axis=-1,
    )


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    vq = np.concatenate((np.zeros((*v.shape[:-1], 1), dtype=v.dtype), v), axis=-1)
    return quat_multiply(quat_multiply(quat_conjugate(q), vq), q)[..., 1:]


def yaw_quaternion(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.moveaxis(q, -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    return np.stack((np.cos(half), np.zeros_like(half), np.zeros_like(half), np.sin(half)), axis=-1)


def projected_gravity(q: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse(q, np.broadcast_to(np.array([0.0, 0.0, -1.0], dtype=q.dtype), q.shape[:-1] + (3,)))


def build_goals(npz_path: str | Path, default_joint_pos_path: str | Path, future_frames: int = 10) -> np.ndarray:
    data = np.load(npz_path)
    names = [str(x) for x in data["body_names"]]
    root = names.index("pelvis")
    q = data["body_quat_w"][:, root].astype(np.float32)
    root_lin = data["body_lin_vel_w"][:, root].astype(np.float32)
    root_ang = data["body_ang_vel_w"][:, root].astype(np.float32)
    joint_pos = data["joint_pos"].astype(np.float32)
    joint_vel = data["joint_vel"].astype(np.float32)
    default = np.load(default_joint_pos_path).astype(np.float32).reshape(1, -1)
    if joint_pos.shape[1] != 29 or default.shape[1] != 29:
        raise ValueError("expected 29 robot joints")

    body_gravity = projected_gravity(q)
    ang_body = quat_rotate_inverse(q, root_ang)
    horizontal = quat_rotate_inverse(yaw_quaternion(q), root_lin)
    height = data["body_pos_w"][:, root, 2:3].astype(np.float32)
    per_frame = np.concatenate(
        (joint_pos - default, joint_vel, body_gravity, ang_body, horizontal, height), axis=-1
    )
    if per_frame.shape[1] != 68:
        raise ValueError(f"expected 68 goal features, got {per_frame.shape[1]}")

    n = len(per_frame)
    goals = np.empty((n, future_frames * 68), dtype=np.float32)
    for i in range(n):
        ids = np.minimum(np.arange(i, i + future_frames), n - 1)
        goals[i] = per_frame[ids].reshape(-1)
    return goals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("default_joint_pos", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    goals = build_goals(args.trajectory, args.default_joint_pos)
    np.savez_compressed(args.output, goal=goals, fps=np.array([50.0], dtype=np.float32))
    print(f"wrote {args.output}: {goals.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
