#!/usr/bin/env python3
from pathlib import Path
import time
import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "assets/A00_robot_mujoco_supermimic.xml")
    )
data = mujoco.MjData(model)
motion = np.load(ROOT / "assets/walk2_subject3.npz")

names = [str(x) for x in motion["joint_names"]]
pelvis = [str(x) for x in motion["body_names"]].index("pelvis")
frame = 9

ankle_name = "right_ankle_pitch"
ankle_index = names.index("right_ankle_pitch_joint")
ankle_jid = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, ankle_name
)
ankle_aid = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, ankle_name + "_servo"
)

data.qpos[:3] = motion["body_pos_w"][frame, pelvis]
data.qpos[3:7] = motion["body_quat_w"][frame, pelvis]
for i, joint_name in enumerate(names):
    name = joint_name.removesuffix("_joint")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = motion["joint_pos"][frame, i]

mujoco.mj_forward(model, data)
base_target = float(motion["joint_pos"][frame, ankle_index])

print("right ankle pitch direction test")
print("base target:", base_target)
print("torque limit:", model.actuator_forcerange[ankle_aid])

with mujoco.viewer.launch_passive(model, data) as viewer:
    for delta in (0.05, -0.05):
        target = base_target + delta
        print("command target:", target)
        for _ in range(25):
            if not viewer.is_running():
                return
            data.ctrl[ankle_aid] = target
            mujoco.mj_step(model, data)
            viewer.sync()
            print(
                f"target={target:+.3f} "
                f"actual={data.qpos[model.jnt_qposadr[ankle_jid]]:+.3f} "
                f"force={data.actuator_force[ankle_aid]:+.3f}"
            )
            time.sleep(0.02)

    main()
