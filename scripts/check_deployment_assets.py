#!/usr/bin/env python3
"""Run all deployment-side static checks without ROS2 or a robot."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def run(command):
    print("$", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    root = Path(__file__).resolve().parents[1]
    run([sys.executable, str(root / "scripts/check_policy_contract.py"), str(root / "assets/policy.onnx")])
    run([sys.executable, str(root / "scripts/check_joint_mapping.py"), str(root / "assets/A00_robot_mujoco_actual_used.xml"), str(root / "assets/walk2_subject3.npz"), str(root / "assets/default_joint_pos.npy")])
    run([sys.executable, str(root / "scripts/check_model_assets.py"), str(root / "assets/A00_robot_mujoco_actual_used.xml")])
    print("All static deployment checks passed")


if __name__ == "__main__":
    main()
