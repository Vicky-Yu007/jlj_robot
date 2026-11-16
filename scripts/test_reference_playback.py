#!/usr/bin/env python3
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    motion = np.load(ROOT / "assets/walk2_subject3.npz")
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "assets/A00_robot_mujoco_supermimic.xml")
    )
    data = mujoco.MjData(model)

    joint_names = [str(x) for x in motion["joint_names"]]
    body_names = [str(x) for x in motion["body_names"]]
    pelvis_id = body_names.index("pelvis")
    mujoco_pelvis_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
    )

    joint_ids = []
    actuator_ids = []
    for joint_name in joint_names:
        name = joint_name.removesuffix("_joint")
        jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_servo"
        )
        if jid < 0 or aid < 0:
            raise RuntimeError(f"Missing MuJoCo mapping for {name}")
        joint_ids.append(jid)
        actuator_ids.append(aid)

    frame = 0
    data.qpos[:3] = motion["body_pos_w"][frame, pelvis_id]
    data.qpos[3:7] = motion["body_quat_w"][frame, pelvis_id]
    data.qvel[:3] = motion["body_lin_vel_w"][frame, pelvis_id]
    data.qvel[3:6] = motion["body_ang_vel_w"][frame, pelvis_id]

    for i, jid in enumerate(joint_ids):
        data.qpos[model.jnt_qposadr[jid]] = motion["joint_pos"][frame, i]
        data.qvel[model.jnt_dofadr[jid]] = motion["joint_vel"][frame, i]

    mujoco.mj_forward(model, data)

    print("Reference playback started")
    print("frames:", len(motion["joint_pos"]))
    print("fps:", float(motion["fps"][0]))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -15

        while viewer.is_running() and frame < len(motion["joint_pos"]):
            data.qpos[:3] = motion["body_pos_w"][frame, pelvis_id]
            data.qpos[3:7] = motion["body_quat_w"][frame, pelvis_id]
            data.qvel[:3] = motion["body_lin_vel_w"][frame, pelvis_id]
            data.qvel[3:6] = motion["body_ang_vel_w"][frame, pelvis_id]

            for i, jid in enumerate(joint_ids):
                data.qpos[model.jnt_qposadr[jid]] = motion["joint_pos"][frame, i]
                data.qvel[model.jnt_dofadr[jid]] = motion["joint_vel"][frame, i]

            mujoco.mj_forward(model, data)

            for i, aid in enumerate(actuator_ids):
                data.ctrl[aid] = motion["joint_pos"][frame, i]

            for _ in range(4):
                mujoco.mj_step(model, data)

            viewer.cam.lookat[:] = data.xpos[mujoco_pelvis_id]
            viewer.sync()

            if frame % 50 == 0:
                print(f"frame={frame} time={data.time:.2f}s pelvis_z={data.qpos[2]:.3f}")

            frame += 1
            time.sleep(0.02)

    print("Reference playback finished")


if __name__ == "__main__":
    main()
