#!/usr/bin/env python3
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
import time

ROOT = Path(__file__).resolve().parents[1]

motion = np.load(ROOT / "assets/walk2_subject3.npz")
model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/A00_robot_mujoco_supermimic.xml"))
data = mujoco.MjData(model)

names = [str(x) for x in motion["joint_names"]]
body_names = [str(x) for x in motion["body_names"]]
pelvis = body_names.index("pelvis")
frame = 9
qpos_ids = []
actuator_ids = []

data.qpos[:3] = motion["body_pos_w"][frame, pelvis]
data.qpos[3:7] = motion["body_quat_w"][frame, pelvis]

for i, joint_name in enumerate(names):
    name = joint_name.removesuffix("_joint")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_servo")
    qpos_ids.append(model.jnt_qposadr[jid])
    actuator_ids.append(aid)
    data.qpos[qpos_ids[-1]] = motion["joint_pos"][frame, i]

mujoco.mj_forward(model, data)
target = motion["joint_pos"][frame].astype(np.float64)

print("Holding reference frame 9 joint targets")
print("target range:", target.min(), target.max())

with mujoco.viewer.launch_passive(model, data) as viewer:
    for _ in range(100):
        if not viewer.is_running():
            break
        for aid, value in zip(actuator_ids, target):
            data.ctrl[aid] = value
        for _ in range(10):
            mujoco.mj_step(model, data)
        viewer.cam.lookat[:] = data.xpos[pelvis]
        viewer.sync()
        print(f"t={data.time:.2f}s pelvis_z={data.qpos[2]:.3f}")
        if data.qpos[2] < 0.3 or not np.isfinite(data.qpos).all():
            print("FAIL: reference target is unstable")
            break
        time.sleep(0.01)
