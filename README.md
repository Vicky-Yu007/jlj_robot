# JLJ Robot SuperMimic Deployment

This repository contains a simulation-first deployment pipeline for the SuperMimic policy trained for the JLJ self-developed robot.

## Pipeline

- 10-frame proprioception history: 930 values
- 10 future reference frames: 680-value motion goal
- ONNX policy inputs: `obs_hist [1, 930]` and `goal [1, 680]`
- Action output: 29 values

The project keeps simulation and hardware interfaces separate. MuJoCo validation must pass before hardware integration.

## Repository layout

```text
config/                 Deployment configuration
scripts/                Model and trajectory validation tools
supermimic_deploy/      Python reference implementation
assets/                 Local-only model and trajectory files
tests/                  Shape and contract tests
```

## Current status

The scaffold defines the SuperMimic input contract. Add the exported ONNX model, training configuration, MuJoCo XML, and `walk2_subject3.npz` under `assets/` locally. Large binary files should not be committed until their size and storage policy are decided.

## Model contract

The ONNX model must expose two inputs and one output:

```text
obs_hist [1, 930]
goal     [1, 680]
action   [1, 29]
```

Run the contract checker with:

```bash
python scripts/check_policy_contract.py assets/policy.onnx
```
