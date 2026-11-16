# Hardware mapping extracted from jijwholebodyctr

The sibling deployment uses 29 logical joints. Direct joints are mapped by motor name. The following logical joints are not direct one-motor mappings:

- left_ankle_pitch + left_ankle_roll -> left_ankle_parallel_2 and left_ankle_parallel_1
- right_ankle_pitch + right_ankle_roll -> right_ankle_parallel_2 and right_ankle_parallel_1
- waist_pitch + waist_roll -> waist_motor1 and waist_motor2

The waist mechanism uses motor_direction [1, -1]. Both ankle mechanisms currently use [1, 1]. These direction values belong to the physical mechanism adapter and must not be applied to the policy action vector.

The policy should output logical joint targets in the training order. The hardware adapter then converts ankle and waist logical targets to motor commands using the sibling repository's parallel-mechanism solver.

A MuJoCo rollout can validate the policy-to-logical-joint layer. It cannot validate CAN IDs, motor direction, encoder zero offsets, or parallel mechanism inverse kinematics. Those require the sibling hardware configuration and a one-joint-at-a-time hardware test.
