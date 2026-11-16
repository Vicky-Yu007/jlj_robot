from __future__ import annotations

OBS_FRAME_DIM = 93
OBS_HISTORY = 10
OBS_DIM = OBS_FRAME_DIM * OBS_HISTORY
GOAL_FRAME_DIM = 68
GOAL_FRAMES = 10
GOAL_DIM = GOAL_FRAME_DIM * GOAL_FRAMES
ACTION_DIM = 29

OBS_TERMS = (
    ("joint_pos", 29),
    ("joint_vel", 29),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("previous_action", 29),
)


def validate_dimensions(obs, goal, action) -> None:
    if len(obs) != OBS_DIM:
        raise ValueError(f"obs_hist must contain {OBS_DIM} values, got {len(obs)}")
    if len(goal) != GOAL_DIM:
        raise ValueError(f"goal must contain {GOAL_DIM} values, got {len(goal)}")
    if len(action) != ACTION_DIM:
        raise ValueError(f"action must contain {ACTION_DIM} values, got {len(action)}")
