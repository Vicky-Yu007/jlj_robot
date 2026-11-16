from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def xml_joint_names(path: str | Path) -> list[str]:
    text = Path(path).read_text()
    return re.findall(r'<joint\s+name="([^"]+)"', text)


def normalize(name: str) -> str:
    return name[:-6] if name.endswith("_joint") else name


def validate(xml_path: str | Path, npz_path: str | Path, default_path: str | Path) -> None:
    xml = xml_joint_names(xml_path)
    data = np.load(npz_path)
    recorded = [str(x) for x in data["joint_names"]]
    default = np.asarray(np.load(default_path)).reshape(-1)
    if len(xml) != 29:
        raise ValueError(f"MuJoCo XML contains {len(xml)} joints, expected 29")
    if len(recorded) != 29 or default.size != 29:
        raise ValueError("trajectory and default pose must both contain 29 joints")
    xml_normalized = [normalize(x) for x in xml]
    recorded_normalized = [normalize(x) for x in recorded]
    if xml_normalized != recorded_normalized:
        for i, (a, b) in enumerate(zip(xml_normalized, recorded_normalized)):
            if a != b:
                raise ValueError(f"joint order mismatch at {i}: XML={a}, trajectory={b}")
        raise ValueError("joint name lists have different lengths")
    print("PASS: XML, trajectory, and default pose have 29 joints in the same order")
    for i, (x, r) in enumerate(zip(xml, recorded)):
        print(f"{i:02d}: {x} <-> {r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("xml")
    parser.add_argument("trajectory")
    parser.add_argument("default_joint_pos")
    args = parser.parse_args()
    validate(args.xml, args.trajectory, args.default_joint_pos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
