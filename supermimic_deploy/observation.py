from __future__ import annotations

import numpy as np


class ObservationHistory:
    """Build the 930-d term-major history expected by SuperMimic."""

    def __init__(self, default_joint_pos: np.ndarray, history_length: int = 10):
        default = np.asarray(default_joint_pos, dtype=np.float32).reshape(-1)
        if default.shape != (29,):
            raise ValueError(f"default_joint_pos must have shape (29,), got {default.shape}")
        if history_length != 10:
            raise ValueError("the trained SuperMimic policy requires history_length=10")
        self.default_joint_pos = default
        self.history_length = history_length
        self._frames = np.zeros((history_length, 93), dtype=np.float32)
        self._initialized = False

    def reset(self) -> None:
        self._frames.fill(0.0)
        self._initialized = False

    def _frame(self, joint_pos, joint_vel, base_ang_vel, projected_gravity, previous_action):
        values = [
            np.asarray(joint_pos, dtype=np.float32).reshape(-1) - self.default_joint_pos,
            np.asarray(joint_vel, dtype=np.float32).reshape(-1),
            np.asarray(base_ang_vel, dtype=np.float32).reshape(-1),
            np.asarray(projected_gravity, dtype=np.float32).reshape(-1),
            np.asarray(previous_action, dtype=np.float32).reshape(-1),
        ]
        expected = (29, 29, 3, 3, 29)
        if any(value.shape != (size,) for value, size in zip(values, expected)):
            raise ValueError("SuperMimic observation term has an unexpected shape")
        return np.concatenate(values)

    def initialize(self, frames) -> None:
        values = np.asarray(frames, dtype=np.float32)
        if values.shape != (self.history_length, 93):
            raise ValueError(f"expected history frames {(self.history_length, 93)}, got {values.shape}")
        self._frames[:] = values
        self._initialized = True

    def packed(self) -> np.ndarray:
        expected = (29, 29, 3, 3, 29)
        chunks = []
        start = 0
        for size in expected:
            chunks.append(self._frames[:, start:start + size].reshape(-1))
            start += size
        output = np.concatenate(chunks).astype(np.float32, copy=False)
        if output.shape != (930,) or not np.isfinite(output).all():
            raise ValueError("invalid 930-d observation history")
        return output.copy()

    def update(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        previous_action: np.ndarray,
    ) -> np.ndarray:
        frame = self._frame(
            joint_pos, joint_vel, base_ang_vel, projected_gravity, previous_action
        )
        if not self._initialized:
            self._frames[:] = frame
            self._initialized = True
        else:
            self._frames[:-1] = self._frames[1:]
            self._frames[-1] = frame
        return self.packed()
