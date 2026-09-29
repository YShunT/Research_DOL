"""exp03 B: 固定checkpointに対する層別介入と全窓の予測評価。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..evaluation.metrics import compute_forecast_metrics
from .diagnostics import compare_functions, effective_down_error
from .storage import Records
from .svd import Decomposition, Weights, approximate, parameter_count


@dataclass(frozen=True)
class Intervention:
    name: str
    down_rank: int | None = None
    up_rank: int | None = None
    projection_seed: int | None = None

    @property
    def key(self) -> str:
        return f"{self.name}:d{self.down_rank}:u{self.up_rank}:p{self.projection_seed}"


def interventions(ranks: tuple[int, ...], grid: bool, random: bool) -> list[Intervention]:
    cases = [Intervention("original"), Intervention("bypass")]
    cases += [Intervention("down", r, None) for r in ranks]
    cases += [Intervention("up", None, r) for r in ranks]
    pairs = [(d, u) for d in ranks for u in ranks] if grid else [(r, r) for r in ranks]
    cases += [Intervention("both", d, u) for d, u in pairs]
    if random:
        cases += [Intervention("random", r, r, p)
                  for r in ranks if r in (4, 8, 16, 32) for p in (0, 1, 2)]
    return cases


def candidate_weights(weights: Weights, case: Intervention, decompositions) -> Weights:
    if case.name == "bypass":
        return Weights(*(np.zeros_like(value) for value in weights.arrays().values()))
    return approximate(weights, case.down_rank, case.up_rank, decompositions,
                       case.projection_seed)


@torch.no_grad()
def predict(components, dataset, batch_size: int, limit: int | None = None,
            log=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Float32 CPU arrays retain the existing raw-compatible metric reductions."""
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        generator=torch.Generator().manual_seed(0))
    model = components.model
    model.eval()
    device = next(model.parameters()).device
    shape = (len(dataset), components.config.data.forecast_horizon,
             components.config.data.num_nodes)
    forecasts = torch.empty(shape, dtype=torch.float32)
    truth = torch.empty_like(forecasts)
    offset = 0
    for inputs, targets in loader:
        output = model(inputs.to(device=device, dtype=torch.float32), components.fixed_supports)
        size = len(inputs)
        # Same CPU conversion and inverse scaling as OnlineAdapter.
        forecasts[offset:offset + size] = components.data.scaler.inverse_transform(output.cpu())
        truth[offset:offset + size] = components.data.scaler.inverse_transform(targets.float()).clamp_min(0)
        offset += size
        if log and offset % (batch_size * 128) == 0:
            log.info("prediction %d/%d", offset, len(dataset))
    return forecasts, truth


def prediction_summary(prediction: torch.Tensor, truth: torch.Tensor,
                       baseline: torch.Tensor) -> dict:
    clipped = prediction.clamp_min(0)
    record = asdict(compute_forecast_metrics(clipped, truth))
    # Reduce in chunks to avoid another full float64 copy of the online forecast.
    before = after = 0.
    node_absolute = np.zeros(truth.shape[2])
    node_squared = np.zeros_like(node_absolute)
    for p, y, ref in zip(prediction.split(256), truth.split(256), baseline.split(256)):
        before += (p.double() - ref.double()).square().sum().item()
        after += (p.clamp_min(0).double() - ref.clamp_min(0).double()).square().sum().item()
        error = p.clamp_min(0).double() - y.double()
        node_absolute += error.abs().sum((0, 1)).numpy()
        node_squared += error.square().sum((0, 1)).numpy()
    record.update(
        prediction_rmse_before_clip=float(np.sqrt(before / truth.numel())),
        prediction_rmse_after_clip=float(np.sqrt(after / truth.numel())),
        node_mae=(node_absolute / (len(truth) * truth.shape[1])).tolist(),
        node_rmse=np.sqrt(node_squared / (len(truth) * truth.shape[1])).tolist(),
    )
    return record


def run_interventions(components, weights: Weights, features: torch.Tensor,
                      dataset, path: Path, ranks, *, grid: bool, random: bool,
                      batch_size: int, limit: int | None, log) -> None:
    records = Records(path)
    cases = interventions(ranks, grid, random)
    if all(records.contains(case.key) for case in cases):
        records.complete()
        return
    learner = components.model.location_specific
    original = Weights.from_learner(learner)
    decompositions = {layer: Decomposition.fit(weights.matrix(layer)) for layer in ("down", "up")}
    try:
        weights.apply(learner)
        baseline, truth = predict(components, dataset, batch_size, limit, log)
        base_metrics = asdict(compute_forecast_metrics(baseline.clamp_min(0), truth))
        for case in cases:
            if records.contains(case.key):
                continue
            log.info("%s | %s", path, case.key)
            candidate = candidate_weights(weights, case, decompositions)
            candidate.apply(learner)
            forecast = baseline if case.name == "original" else predict(
                components, dataset, batch_size, limit, log
            )[0]
            result = prediction_summary(forecast, truth, baseline)
            changes = {
                name: (result[name] - base_metrics[name]) / base_metrics[name]
                if base_metrics[name] != 0 else None
                for name in ("mae", "rmse", "sample_rmse", "wmape")
            }
            function = compare_functions(learner, weights, candidate, features, batch_size)
            full_reconstruction = (
                case.name != "bypass" and case.projection_seed is None
                and all(rank is None or rank == decompositions[layer].max_rank
                        for layer, rank in (("down", case.down_rank), ("up", case.up_rank)))
            )
            if full_reconstruction and (
                function["max_absolute_error"] > 1e-5
                or not torch.allclose(forecast, baseline, atol=1e-5, rtol=1e-5)
            ):
                raise ArithmeticError("full-rank reconstruction changed model predictions")
            weights_error = {}
            for layer in ("down", "up"):
                delta = candidate.matrix(layer) - weights.matrix(layer)
                denom = np.linalg.norm(decompositions[layer].residual)
                weights_error[layer] = {
                    "frobenius": float(np.linalg.norm(delta)),
                    "relative_to_centered": float(np.linalg.norm(delta) / denom)
                    if denom > 1e-12 else None,
                }
            stored = (0 if case.name == "bypass" else
                      parameter_count(weights, case.down_rank, case.up_rank))
            records.save(case.key, {
                **asdict(case), "factor_parameters": stored,
                "dense_parameters": parameter_count(weights, None, None),
                "prediction": result, "relative_metric_change": changes,
                "function": function, "weight_error": weights_error,
                "effective_down": effective_down_error(components.model, weights, candidate),
                "practical_preservation": (
                    case.name in ("down", "up", "both")
                    and stored < parameter_count(weights, None, None)
                    and all(changes[m] is not None and changes[m] <= .01
                            for m in ("mae", "rmse", "sample_rmse"))
                    and function["function_relative_rmse"] is not None
                    and function["function_relative_rmse"] <= .05
                ),
            })
        records.complete()
    finally:
        original.apply(learner)
