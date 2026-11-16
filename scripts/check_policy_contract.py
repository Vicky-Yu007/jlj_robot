#!/usr/bin/env python3
"""Validate the exported SuperMimic ONNX input/output contract."""

from __future__ import annotations

import argparse
from pathlib import Path


def tensor_shape(value_info):
    dims = []
    tensor = value_info.type.tensor_type
    for dim in tensor.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        else:
            dims.append(None)
    return dims


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()

    import onnx

    model = onnx.load(args.model)
    inputs = [(item.name, tensor_shape(item)) for item in model.graph.input]
    outputs = [(item.name, tensor_shape(item)) for item in model.graph.output]

    print("Inputs:")
    for name, shape in inputs:
        print(f"  {name}: {shape}")

    print("Outputs:")
    for name, shape in outputs:
        print(f"  {name}: {shape}")

    shapes = sorted(shape[-1] for _, shape in inputs if shape and shape[-1] is not None)
    if len(inputs) != 2 or shapes != [680, 930]:
        raise SystemExit("FAIL: expected two inputs with final dimensions 930 and 680")

    if len(outputs) != 1 or outputs[0][1][-1] != 29:
        raise SystemExit("FAIL: expected one 29-dimensional output")

    print("PASS: SuperMimic ONNX contract is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
