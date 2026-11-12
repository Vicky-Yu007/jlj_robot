#!/usr/bin/env python3
"""Playback generated npz in MuJoCo; install mujoco and pass the tuned XML."""
import argparse, numpy as np
try: import mujoco, mujoco.viewer
except ImportError: raise SystemExit('Install mujoco first: pip install mujoco')
ap=argparse.ArgumentParser(); ap.add_argument('--xml',required=True); ap.add_argument('--data',required=True); a=ap.parse_args()
m=mujoco.MjModel.from_xml_path(a.xml); d=mujoco.MjData(m); x=np.load(a.data); q=x['joint_pos']
with mujoco.viewer.launch_passive(m,d) as v:
 for pose in q:
  d.qpos[:]=pose; d.qvel[:]=0; mujoco.mj_forward(m,d); v.sync(); import time; time.sleep(1/50)
