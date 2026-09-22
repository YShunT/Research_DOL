"""MAE、global/sample RMSE、WMAPEをrawと同じ一括集計で計算する。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import Tensor


@dataclass(frozen=True)
class ForecastMetrics:
    mae: float
    rmse: float
    sample_rmse: float
    wmape: float
    horizon_mae: tuple[float, ...]
    horizon_rmse: tuple[float, ...]
    horizon_wmape: tuple[float, ...]
    num_values: int


def compute_forecast_metrics(
    prediction: Tensor,
    target: Tensor,
    epsilon: float = 1.17e-6,
) -> ForecastMetrics:
    """全予測を連結してから、rawのutils.metrics.metricと同じNumPy演算で集計する。"""

    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same (B,H,N) shape")
    if prediction.numel() == 0:
        raise ValueError("metrics received empty predictions")

    preds = prediction.detach().cpu().numpy()
    trues = target.detach().cpu().numpy()
    if not np.isfinite(preds).all() or not np.isfinite(trues).all():
        raise ValueError("metrics received non-finite values")

    error = preds - trues
    absolute_error = np.abs(error)
    squared_error = np.square(error)
    absolute_target = np.abs(trues)
    # rawのmetric()は全要素を一度にNumPyで集計する。
    mae = float(np.mean(absolute_error))
    rmse = float(np.sqrt(np.mean(squared_error)))
    wmape = float(absolute_error.sum() / np.clip(absolute_target.sum(), epsilon, None))
    # sample_rmseとホライズン別指標はrawにはない追加の補助指標。
    sample_rmse = float(np.sqrt(np.mean(squared_error, axis=(1, 2))).mean())
    horizon_mae = np.mean(absolute_error, axis=(0, 2))
    horizon_rmse = np.sqrt(np.mean(squared_error, axis=(0, 2)))
    horizon_wmape = absolute_error.sum(axis=(0, 2)) / np.clip(
        absolute_target.sum(axis=(0, 2)), epsilon, None
    )
    return ForecastMetrics(
        mae=mae,
        rmse=rmse,
        sample_rmse=sample_rmse,
        wmape=wmape,
        horizon_mae=tuple(float(value) for value in horizon_mae),
        horizon_rmse=tuple(float(value) for value in horizon_rmse),
        horizon_wmape=tuple(float(value) for value in horizon_wmape),
        num_values=prediction.numel(),
    )
