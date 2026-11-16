#!/usr/bin/env python3
from pathlib import Path
import time
import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    motion = np.load(ROOT / "assets/walk2_subject3.npz")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/A00_robot_mujoco_supermimic.xml"))
    data = mujoco.MjData(model)
    names = [str(x) for x in motion["joint_names"]]
    bodies = [str(x) for x in motion["body_names"]]
    pelvis = bodies.index("pelvis")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    qids, aids = [], []
    for n in names:
        n = n
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_servo")
        qids.append(jid); aids.append(aid)

    data.qpos[:3] = motion["body_pos_w"][0, pelvis]
    data.qpos[3:7] = motion["body_quat_w"][0, pelvis]
    data.qvel[:3] = motion["body_lin_vel_w"][0, pelvis]
    data.qvel[3:6] = motion["body_ang_vel_w"][0, pelvis]
    for i, jid in enumerate(qids):
        data.qpos[model.jnt_qposadr[jid]] = motion["joint_pos"][0, i]
        data.qvel[model.jnt_dofadr[jid]] = motion["joint_vel"][0, i]
    mujoco.mj_forward(model, data)

    frame = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and frame < len(motion["joint_pos"]):
            for i, aid in enumerate(aids):
                data.ctrl[aid] = motion["joint_pos"][frame, i]
            for _ in range(4):
                mujoco.mj_step(model, data)
            viewer.cam.lookat[:] = data.xpos[pelvis_id]
            viewer.sync()
            if frame % 50 == 0:
                actual = np.array([data.qpos[model.jnt_qposadr[j]] for j in qids])
                errors = motion["joint_pos"][frame] - actual
                forces = np.asarray(data.actuator_force)[aids]
                limits = np.asarray(model.actuator_forcerange)[aids, 1]
                saturated = np.sum(np.abs(forces) >= 0.98 * limits)
                order = np.argsort(np.abs(errors))[::-1][:5]
                worst_items = []
                for i in order:
                    name = str(motion["joint_names"][i])
                    worst_items.append(
                        f"{name} err={errors[i]:+.3f} "
                        f"target={motion["joint_pos"][frame, i]:+.3f} "
                        f"actual={actual[i]:+.3f} force={forces[i]:+.2f}"
                    )
                worst = "; ".join(worst_items)
                print(f"frame={frame} time={data.time:.2f}s pelvis_z={data.qpos[2]:.3f} max_joint_error={np.max(np.abs(errors)):.3f} saturated={saturated}/29 max_force={np.max(np.abs(forces)):.2f}")
                print("  worst:", worst)
            frame += 1
            time.sleep(0.01)
    print("PD reference tracking finished")


if __name__ == "__main__":
    main()
