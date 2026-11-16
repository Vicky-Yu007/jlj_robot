#!/usr/bin/env python3
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supermimic_deploy.onnx_policy import SuperMimicONNXPolicy
from supermimic_deploy.observation import ObservationHistory


def quat_inverse_rotate(q, v):
    q = np.asarray(q, dtype=np.float32)
    qc = q.copy()
    qc[1:] *= -1
    vq = np.array([0, v[0], v[1], v[2]], dtype=np.float32)
    def mul(a, b):
        return np.array([
            a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3],
            a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
            a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1],
            a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0],
        ])
    return mul(mul(qc, vq), q)[1:]


def main():
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/A00_robot_mujoco_supermimic.xml"))
    data = mujoco.MjData(model)
    motion = np.load(ROOT / "assets/walk2_subject3.npz")
    goals = np.load(ROOT / "assets/walk2_subject3_goals.npz")["goal"]
    default = np.load(ROOT / "assets/default_joint_pos.npy").astype(np.float32)
    names = [str(x) for x in motion["joint_names"]]
    pelvis = [str(x) for x in motion["body_names"]].index("pelvis")

    qpos_ids, qvel_ids, actuator_ids = [], [], []
    for name in names:
        name = name.removesuffix("_joint")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_servo")
        qpos_ids.append(model.jnt_qposadr[jid])
        qvel_ids.append(model.jnt_dofadr[jid])
        actuator_ids.append(aid)

    frame = 9
    data.qpos[:3] = motion["body_pos_w"][frame, pelvis]
    data.qpos[3:7] = motion["body_quat_w"][frame, pelvis]
    data.qvel[:3] = motion["body_lin_vel_w"][frame, pelvis]
    data.qvel[3:6] = motion["body_ang_vel_w"][frame, pelvis]
    for i, name in enumerate(names):
        name = name.removesuffix("_joint")
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = motion["joint_pos"][frame, i]
        data.qvel[model.jnt_dofadr[jid]] = motion["joint_vel"][frame, i]
    mujoco.mj_forward(model, data)

    history = ObservationHistory(default)
    frames = []
    for k in range(10):
        q = motion["body_quat_w"][k, pelvis]
        frames.append(np.concatenate((
            motion["joint_pos"][k] - default,
            motion["joint_vel"][k],
            quat_inverse_rotate(q, motion["body_ang_vel_w"][k, pelvis]),
            quat_inverse_rotate(q, np.array([0, 0, -1], dtype=np.float32)),
            np.zeros(29, dtype=np.float32),
        )))
    history.initialize(np.asarray(frames, dtype=np.float32))
    policy = SuperMimicONNXPolicy(ROOT / "assets/policy.onnx")
    action = np.clip(policy(history.packed(), goals[frame]), -5, 5)
    target = default + 0.18 * action
    print("Holding first policy action for 2 seconds")
    print("action norm:", np.linalg.norm(action))
    print("target range:", target.min(), target.max())

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in range(100):
            if not viewer.is_running():
                break
            for aid, value in zip(actuator_ids, target):
                data.ctrl[aid] = value
            for _ in range(10):
                mujoco.mj_step(model, data)
            viewer.cam.lookat[:] = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
            viewer.sync()
            print(f"t={data.time:.2f}s pelvis_z={data.qpos[2]:.3f}")
            if data.qpos[2] < 0.3 or not np.isfinite(data.qpos).all():
                print("FAIL: fixed target is unstable")
                return
            time.sleep(0.01)
    print("PASS: fixed first target remained above safety height")


if __name__ == "__main__":
    main()
