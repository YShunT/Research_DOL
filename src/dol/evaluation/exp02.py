"""exp02のオンライン逐次診断。予測配列を全量保存せず集約する。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..online.adapter import OnlineStepResult
from ..pipeline import DOLComponents


class OnlineDiagnostics:
    """週・地点・ホライズンごとの誤差とLSLの週次snapshotを蓄積する。"""

    def __init__(self, components: DOLComponents, total_windows: int) -> None:
        self.components = components
        self.period = components.config.data.steps_per_week
        self.horizon = components.config.data.forecast_horizon
        self.num_nodes = components.config.data.num_nodes
        self.num_weeks = (total_windows + self.period - 1) // self.period
        shape = (self.num_weeks, self.num_nodes)
        self.absolute_sum = np.zeros(shape, dtype=np.float64)
        self.signed_sum = np.zeros(shape, dtype=np.float64)
        self.squared_sum = np.zeros(shape, dtype=np.float64)
        self.target_sum = np.zeros(shape, dtype=np.float64)
        self.windows = np.zeros(self.num_weeks, dtype=np.int64)
        self.awake = np.zeros(self.num_weeks, dtype=np.int64)
        self.updates = np.zeros(self.num_weeks, dtype=np.int64)
        self.horizon_absolute_sum = np.zeros(self.horizon, dtype=np.float64)
        self.horizon_squared_sum = np.zeros(self.horizon, dtype=np.float64)
        self.horizon_target_sum = np.zeros(self.horizon, dtype=np.float64)
        self.snapshots: list[dict[str, np.ndarray | int]] = []
        self.reference_features = self._make_reference_features()
        self.reference_correction = self._correction()
        self.initial_weights = self._generated_weights()
        self.initial_B = self._parameter_array("B")
        self.take_snapshot(-1)

    def _make_reference_features(self) -> torch.Tensor:
        data = self.components.data.train
        indices = np.unique(np.linspace(0, len(data) - 1, min(8, len(data)), dtype=int))
        samples = torch.stack([data[int(index)][0] for index in indices])
        model = self.components.model
        inputs = samples.to(next(model.parameters()).device, dtype=torch.float32)
        x = inputs.unsqueeze(1).transpose(2, 3)
        if x.shape[-1] < model.receptive_field:
            x = torch.nn.functional.pad(x, (model.receptive_field - x.shape[-1], 0, 0, 0))
        model.eval()
        with torch.no_grad():
            return model.input_projection(x).detach()

    def _correction(self) -> torch.Tensor:
        with torch.no_grad():
            return self.components.model.location_specific(self.reference_features).detach()

    def _generated_weights(self) -> np.ndarray:
        learner = self.components.model.location_specific
        if hasattr(learner, "generated_weights"):
            with torch.no_grad():
                return learner.generated_weights().detach().cpu().numpy().copy()
        return np.zeros((self.num_nodes, 0), dtype=np.float32)

    def _parameter_array(self, name: str) -> np.ndarray:
        parameter = getattr(self.components.model.location_specific, name, None)
        if parameter is None:
            return np.empty((0,), dtype=np.float32)
        return parameter.detach().cpu().numpy().copy()

    def take_snapshot(self, step: int) -> None:
        weights = self._generated_weights()
        correction = self._correction()
        # 各参照入力・時刻におけるチャネル方向のL2ノルム二乗を平均する。
        distance = (
            (correction - self.reference_correction)
            .square()
            .sum(dim=1)
            .mean(dim=(0, 2))
        )
        b = self._parameter_array("B")
        self.snapshots.append(
            {
                "step": step,
                "E": self._parameter_array("E"),
                "B": b,
                "weight_change": (
                    np.sqrt(np.square(weights - self.initial_weights).sum(axis=1))
                    if weights.shape[1] else np.zeros(self.num_nodes, dtype=np.float32)
                ),
                "basis_change": (
                    float(np.linalg.norm(b - self.initial_B)) if b.size else 0.0
                ),
                "correction_rms_change": torch.sqrt(distance).cpu().numpy(),
            }
        )

    def on_step(self, result: OnlineStepResult) -> None:
        week = result.step // self.period
        pred = result.prediction.squeeze(0).numpy()
        target = result.target.squeeze(0).numpy()
        error = pred - target
        self.absolute_sum[week] += np.abs(error).sum(axis=0)
        self.signed_sum[week] += error.sum(axis=0)
        self.squared_sum[week] += np.square(error).sum(axis=0)
        self.target_sum[week] += np.abs(target).sum(axis=0)
        self.horizon_absolute_sum += np.abs(error).sum(axis=1)
        self.horizon_squared_sum += np.square(error).sum(axis=1)
        self.horizon_target_sum += np.abs(target).sum(axis=1)
        self.windows[week] += 1
        self.awake[week] += int(result.phase == "awake")
        self.updates[week] += int(result.update_mae is not None)
        if (result.step + 1) % self.period == 0:
            self.take_snapshot(result.step)

    def finish(self, last_step: int, path: Path) -> None:
        if not self.snapshots or self.snapshots[-1]["step"] != last_step:
            self.take_snapshot(last_step)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            absolute_sum=self.absolute_sum,
            signed_sum=self.signed_sum,
            squared_sum=self.squared_sum,
            target_sum=self.target_sum,
            windows=self.windows,
            awake=self.awake,
            updates=self.updates,
            horizon_absolute_sum=self.horizon_absolute_sum,
            horizon_squared_sum=self.horizon_squared_sum,
            horizon_target_sum=self.horizon_target_sum,
            snapshot_steps=np.asarray([s["step"] for s in self.snapshots], dtype=np.int64),
            E=np.asarray([s["E"] for s in self.snapshots]),
            B=np.asarray([s["B"] for s in self.snapshots]),
            weight_change=np.asarray([s["weight_change"] for s in self.snapshots]),
            basis_change=np.asarray([s["basis_change"] for s in self.snapshots]),
            correction_rms_change=np.asarray(
                [s["correction_rms_change"] for s in self.snapshots]
            ),
        )
