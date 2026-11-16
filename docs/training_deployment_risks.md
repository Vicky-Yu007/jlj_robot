# A00 training deployment risks

This report records six issues that affect deployment.

## Important conclusions

- The training FK must use the A00 model. The deployment must never use a G1 FK or G1 default pose.
- The training action order must be verified independently. Grouped actuator declarations may reorder action dimensions in the simulator. Matching XML joint order to NPZ joint order is necessary but is not sufficient to prove the ONNX action order.
- Preserve the original retargeted trajectory. Do not clip or rewrite the motion data unless the training run used that exact preprocessing.
- Keep all tracked wrist roll and pitch bodies required by the training configuration.
- Use action scale 0.18, the recorded default pose, and the final PD gains from the same training run.
- Treat the training checkpoint and deployment package as one versioned pair. Do not mix a policy from one checkpoint with another run's configuration.

## Required action-order proof

Before trusting a MuJoCo rollout, produce an explicit table:

| policy index | training action name | MuJoCo joint | default pose index | hardware logical joint |
|---:|---|---|---:|---|

The table must come from the resolved training environment or an exported action-name manifest. It must not be inferred only from the order of YAML actuator groups.

A practical test is to set one action dimension to a small positive value while all other action dimensions are zero, run one simulation step from the default pose, and verify which named joint moves. Repeat for all 29 dimensions. Save the result as an action-order manifest.

## Hardware mapping boundary

A correct MuJoCo rollout validates policy-to-logical-joint mapping only when the simulation uses the same XML axes, offsets, default pose, and actuator convention as training. It does not validate logical-joint-to-motor mapping.

The real robot still requires checks for motor names, CAN IDs, direction signs, zero offsets, and parallel ankle/waist mechanisms. Those checks must use the hardware repository motor and parallel-mechanism YAML files.
