"""訓練期間だけから統計量を求める全体標準化。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


@dataclass
class GlobalStandardScaler:
    """全時刻・全地点で共有する平均と標準偏差を保持する。

    DOLの公開実装と同じく、地点別ではなく単一の平均 ``mean`` と
    標準偏差 ``std`` を使用する。fitにはwarm-up訓練期間だけを渡し、
    validationやonline期間から統計量が漏れないようにする。
    """

    epsilon: float = 1e-6
    mean: float | None = None
    std: float | None = None

    def fit(self, values: np.ndarray | Tensor) -> "GlobalStandardScaler":
        is_empty = (
            values.size == 0
            if isinstance(values, np.ndarray)
            else values.numel() == 0
        )
        if is_empty:
            raise ValueError("cannot fit scaler on empty values")

        if isinstance(values, Tensor):
            if not torch.is_floating_point(values):
                values = values.float()
            mean = float(values.mean().item())
            std = float(values.std(unbiased=False).item())
        else:
            array = np.asarray(values, dtype=np.float64)
            mean = float(array.mean())
            std = float(array.std())

        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError("scaler statistics must be finite")
        if std < self.epsilon:
            raise ValueError(
                f"standard deviation {std} is smaller than epsilon {self.epsilon}"
            )

        self.mean = mean
        self.std = std
        return self

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def transform(self, values: np.ndarray | Tensor) -> np.ndarray | Tensor:
        mean, std = self._statistics()
        return (values - mean) / std

    def inverse_transform(self, values: np.ndarray | Tensor) -> np.ndarray | Tensor:
        mean, std = self._statistics()
        return values * std + mean

    def state_dict(self) -> dict[str, Any]:
        mean, std = self._statistics()
        return {"epsilon": self.epsilon, "mean": mean, "std": std}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.epsilon = float(state["epsilon"])
        self.mean = float(state["mean"])
        self.std = float(state["std"])
        self._statistics()

    def _statistics(self) -> tuple[float, float]:
        if not self.is_fitted:
            raise RuntimeError("scaler must be fitted before use")
        assert self.mean is not None
        assert self.std is not None
        if self.std < self.epsilon:
            raise RuntimeError("stored standard deviation is invalid")
        return self.mean, self.std
