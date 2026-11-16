from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("xml", type=Path)
    args = parser.parse_args()
    xml = args.xml.resolve()
    text = xml.read_text()
    meshdir_match = re.search(r'<compiler[^>]*meshdir="([^"]+)"', text)
    meshdir = Path(meshdir_match.group(1)) if meshdir_match else Path(".")
    missing = []
    for filename in re.findall(r'<mesh[^>]*file="([^"]+)"', text):
        path = xml.parent / meshdir / filename
        if not path.exists():
            missing.append(str(path))
    if missing:
        print("Missing mesh files:")
        print("\n".join(missing))
        return 1
    print("PASS: every XML mesh file exists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
