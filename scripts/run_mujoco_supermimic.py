#!/usr/bin/env python3
from pathlib import Path
import sys

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supermimic_deploy.onnx_policy import SuperMimicONNXPolicy
from supermimic_deploy.observation import ObservationHistory
from supermimic_deploy.trajectory import load_precomputed_goals


def quat_conjugate(q):
    out = q.copy()
    out[1:] *= -1.0
    return out


def quat_mul(a, b):
    return np.array([
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
    ])


def rotate_inverse(q, vector):
    vq = np.array([0.0, vector[0], vector[1], vector[2]])
    return quat_mul(quat_mul(quat_conjugate(q), vq), q)[1:]


def main():
    xml = ROOT / "assets/A00_robot_mujoco_supermimic.xml"
    npz = ROOT / "assets/walk2_subject3.npz"
    goals = ROOT / "assets/walk2_subject3_goals.npz"
    default = np.load(ROOT / "assets/default_joint_pos.npy").astype(np.float32)

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    policy = SuperMimicONNXPolicy(ROOT / "assets/policy.onnx")
    trajectory = np.load(npz)
    goal_trajectory = load_precomputed_goals(goals)
    history = ObservationHistory(default)

    joint_names = [str(x) for x in trajectory["joint_names"]]
    qpos_addresses = []
    qvel_addresses = []
    actuator_ids = []
    for name in joint_names:
        joint_name = name[:-6] if name.endswith("_joint") else name
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise RuntimeError(f"MuJoCo joint not found: {joint_name}")
        qpos_addresses.append(model.jnt_qposadr[jid])
        qvel_addresses.append(model.jnt_dofadr[jid])
        actuator = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name + "_servo"
        )
        if actuator < 0:
            raise RuntimeError(f"MuJoCo actuator not found: {joint_name}_servo")
        actuator_ids.append(actuator)

    data.qpos[0:3] = trajectory["body_pos_w"][9, 1]
    data.qpos[3:7] = trajectory["body_quat_w"][9, 1]
    data.qvel[0:3] = trajectory["body_lin_vel_w"][9, 1]
    data.qvel[3:6] = trajectory["body_ang_vel_w"][9, 1]
    for i, address in enumerate(qpos_addresses):
        data.qpos[address] = trajectory["joint_pos"][9, i]
        data.qvel[qvel_addresses[i]] = trajectory["joint_vel"][9, i]
    mujoco.mj_forward(model, data)

    previous_action = np.zeros(29, dtype=np.float32)
    raw_action = np.zeros(29, dtype=np.float32)
    frame = 0
    sim_steps_per_policy = 4
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and frame < len(goal_trajectory):
            joint_pos = np.array([data.qpos[a] for a in qpos_addresses])
            joint_vel = np.array([data.qvel[a] for a in qvel_addresses])
            base_ang_vel = data.qvel[3:6].copy()
            gravity = rotate_inverse(data.qpos[3:7], np.array([0.0, 0.0, -1.0]))
            obs = history.update(
                joint_pos, joint_vel, base_ang_vel, gravity, previous_action
            )
            goal = goal_trajectory.future_goal(frame)
            raw_action = policy(obs, goal)
            raw_action = np.clip(raw_action, -5.0, 5.0)
            target = default + 0.18 * raw_action
            for i, actuator in enumerate(actuator_ids):
                data.ctrl[actuator] = target[i]
            previous_action = raw_action

            for _ in range(sim_steps_per_policy):
                mujoco.mj_step(model, data)

            frame += 1
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                print("Stopping: qpos or qvel contains NaN/Inf")
                break
            if data.qpos[2] < 0.3 or data.qpos[2] > 3.0:
                print(f"Stopping: pelvis height became abnormal: {data.qpos[2]:.3f} m")
                break
            viewer.cam.lookat[:] = data.xpos[pelvis_id]
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 120
            if frame % 25 == 0:
                print(f"frame={frame} time={data.time:.2f}s pelvis_z={data.qpos[2]:.3f} action_norm={np.linalg.norm(raw_action):.3f}")
            viewer.sync()



if __name__ == "__main__":
    main()
