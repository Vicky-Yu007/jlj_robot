from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GoalTrajectory:
    """Per-frame precomputed SuperMimic goals with shape [frames, 680]."""

    goals: np.ndarray
    fps: float

    @classmethod
    def from_goal_array(cls, goals: np.ndarray, fps: float = 50.0) -> "GoalTrajectory":
        values = np.asarray(goals, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 680:
            raise ValueError(f"expected [frames, 680], got {values.shape}")
        if len(values) == 0:
            raise ValueError("goal trajectory must contain at least one frame")
        return cls(values, float(fps))

    def future_goal(self, frame: int, future_frames: int = 10) -> np.ndarray:
        """Return the already-packed 680-d goal for the requested reference frame."""
        if future_frames != 10:
            raise ValueError("SuperMimic policy contract requires 10 future frames")
        index = int(np.clip(frame, 0, len(self.goals) - 1))
        return self.goals[index].copy()

    def __len__(self) -> int:
        return len(self.goals)


def load_precomputed_goals(path: str | Path, fps: float | None = None) -> GoalTrajectory:
    path = Path(path)
    with np.load(path) as loaded:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if "goal" not in loaded:
                raise KeyError("NPZ must contain a precomputed 'goal' array")
            values = loaded["goal"]
            stored_fps = float(np.asarray(loaded["fps"]).reshape(-1)[0]) if "fps" in loaded else 50.0
        else:
            values = loaded
            stored_fps = 50.0
    return GoalTrajectory.from_goal_array(values, stored_fps if fps is None else fps)
