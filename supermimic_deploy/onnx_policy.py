from __future__ import annotations

from pathlib import Path

import numpy as np


class SuperMimicONNXPolicy:
    def __init__(self, model_path: str | Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 2 or len(outputs) != 1:
            raise ValueError("expected two inputs and one output")
        by_dim = {int(item.shape[-1]): item for item in inputs}
        if set(by_dim) != {680, 930}:
            raise ValueError(f"unexpected input dimensions: {by_dim.keys()}")
        if int(outputs[0].shape[-1]) != 29:
            raise ValueError("expected a 29-dimensional action output")
        self.obs_name = by_dim[930].name
        self.goal_name = by_dim[680].name
        self.output_name = outputs[0].name

    def __call__(self, obs_hist: np.ndarray, goal: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs_hist, dtype=np.float32).reshape(1, 930)
        ref = np.asarray(goal, dtype=np.float32).reshape(1, 680)
        if not np.isfinite(obs).all() or not np.isfinite(ref).all():
            raise ValueError("policy input contains NaN or Inf")
        action = self.session.run(
            [self.output_name],
            {self.obs_name: obs, self.goal_name: ref},
        )[0]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError("invalid policy output")
        return action
