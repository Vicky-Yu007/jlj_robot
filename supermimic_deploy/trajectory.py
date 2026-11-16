from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GoalTrajectory:
    """Reference trajectory storage for the 680-dimensional future goal."""

    goals: np.ndarray
    fps: float

    @classmethod
    def from_goal_array(cls, goals: np.ndarray, fps: float = 50.0) -> "GoalTrajectory":
        values = np.asarray(goals, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 680:
            raise ValueError(f"expected [frames, 680], got {values.shape}")
        return cls(values, float(fps))

    def future_goal(self, frame: int, future_frames: int = 10) -> np.ndarray:
        if future_frames != 10:
            raise ValueError("SuperMimic policy contract requires 10 future frames")
        indices = np.minimum(
            np.arange(frame, frame + future_frames, dtype=np.int64),
            len(self.goals) - 1,
        )
        return self.goals[indices].reshape(-1).copy()


def load_precomputed_goals(path: str | Path, fps: float = 50.0) -> GoalTrajectory:
    values = np.load(path)
    if isinstance(values, np.lib.npyio.NpzFile):
        if "goal" not in values:
            raise KeyError("NPZ must contain a precomputed 'goal' array")
        values = values["goal"]
    return GoalTrajectory.from_goal_array(values, fps)
