"""exp01のLSL補正関数を固定バックボーン上でrank別に蒸留する。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from ..artifacts import write_json
from ..config import ExperimentConfig
from ..layers.napl import NAPLLocationLearner
from ..pipeline import build_components, load_warmup_checkpoint
from .metrics import compute_forecast_metrics


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _features_and_correction(
    model: Any,
    dataset: Any,
    count: int,
    batch_size: int = 16,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    indices = np.unique(np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int))
    inputs = torch.stack([dataset[int(index)][0].float() for index in indices])
    targets = torch.stack([dataset[int(index)][1].float() for index in indices])
    features: list[Tensor] = []
    corrections: list[Tensor] = []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in inputs.split(batch_size):
            x = batch.to(device).unsqueeze(1).transpose(2, 3)
            if x.shape[-1] < model.receptive_field:
                x = torch.nn.functional.pad(
                    x, (model.receptive_field - x.shape[-1], 0, 0, 0)
                )
            embedding = model.input_projection(x)
            features.append(embedding.cpu())
            corrections.append(model.location_specific(embedding).cpu())
    return inputs, targets, torch.cat(features), torch.cat(corrections)


def _validation_prediction_metrics(
    model: Any,
    student: NAPLLocationLearner,
    inputs: Tensor,
    targets: Tensor,
    supports: list[Tensor],
    scaler: Any,
) -> dict[str, Any]:
    original = model.location_specific
    device = next(model.parameters()).device
    forecasts: dict[str, Tensor] = {}
    try:
        for name, learner in (("teacher", original), ("student", student)):
            model.location_specific = learner
            model.eval()
            outputs = []
            with torch.no_grad():
                for batch in inputs.split(16):
                    normalized = model(batch.to(device), supports)
                    outputs.append(scaler.inverse_transform(normalized.cpu()).clamp_min(0))
            forecasts[name] = torch.cat(outputs)
    finally:
        model.location_specific = original
    truth = scaler.inverse_transform(targets).clamp_min(0)
    return {
        "teacher": asdict(compute_forecast_metrics(forecasts["teacher"], truth)),
        "student": asdict(compute_forecast_metrics(forecasts["student"], truth)),
        "forecast_mae_vs_teacher": float(
            (forecasts["student"] - forecasts["teacher"]).abs().mean().item()
        ),
    }


def run_distillation(
    root: Path = Path("experiments/exp02"),
    *,
    only_seed: int | None = None,
    only_rank: int | None = None,
    train_samples: int = 1024,
    validation_samples: int = 256,
    max_epochs: int = 100,
    patience: int = 10,
) -> dict[str, Any]:
    from ..exp02 import RANKS, SEEDS, _check_source_clean, _git_commit

    if only_seed is not None and only_seed not in SEEDS:
        raise ValueError("invalid seed")
    if only_rank is not None and only_rank not in RANKS:
        raise ValueError("invalid rank")
    _check_source_clean()
    code_commit = _git_commit()
    root = Path(root)
    all_records: list[dict[str, Any]] = []
    for seed in SEEDS:
        if only_seed is not None and only_seed != seed:
            continue
        checkpoint = Path(f"experiments/exp01/seeds/seed{seed}/checkpoints/warmup.pt")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        config = replace(ExperimentConfig(), warmup=replace(ExperimentConfig().warmup, seed=seed))
        components = build_components(config)
        load_warmup_checkpoint(components, checkpoint)
        teacher = components.model
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        _, _, train_features, train_correction = _features_and_correction(
            teacher, components.data.train, train_samples
        )
        val_inputs, val_targets, val_features, val_correction = _features_and_correction(
            teacher, components.data.validation, validation_samples
        )
        device = next(teacher.parameters()).device
        for rank in RANKS:
            if only_rank is not None and rank != only_rank:
                continue
            path = root / "diagnostics/distillation" / f"seed{seed}" / f"r{rank}"
            metrics_path = path / "metrics.json"
            if metrics_path.exists():
                record = json.loads(metrics_path.read_text(encoding="utf-8"))
                if record.get("status") == "completed":
                    if (
                        record.get("teacher_checkpoint_sha256") != _sha256(checkpoint)
                        or record.get("code_commit") != code_commit
                    ):
                        raise RuntimeError(f"teacher checkpoint changed: {checkpoint}")
                    all_records.append(record)
                    continue
                raise RuntimeError(f"incomplete distillation: {metrics_path}")
            path.mkdir(parents=True, exist_ok=True)
            student = NAPLLocationLearner(
                num_nodes=config.data.num_nodes,
                channels=config.model.residual_channels,
                bottleneck_channels=config.model.lsl_bottleneck_channels,
                dropout=config.model.lsl_dropout,
                rank=rank,
                with_shared=True,
            ).to(device)
            optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
            loader = DataLoader(
                TensorDataset(train_features, train_correction),
                batch_size=16, shuffle=True,
            )
            best_loss = float("inf")
            best_state = None
            best_epoch = 0
            stale = 0
            for epoch in range(1, max_epochs + 1):
                student.train()
                for features, target_correction in loader:
                    optimizer.zero_grad(set_to_none=True)
                    output = student(features.to(device))
                    loss = torch.nn.functional.mse_loss(output, target_correction.to(device))
                    loss.backward()
                    optimizer.step()
                student.eval()
                with torch.no_grad():
                    validation_loss = 0.0
                    for features, target_correction in zip(
                        val_features.split(16), val_correction.split(16)
                    ):
                        difference = student(features.to(device)) - target_correction.to(device)
                        validation_loss += float(difference.square().sum().item())
                    validation_loss /= val_correction.numel()
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in student.state_dict().items()
                    }
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break
            if best_state is None:
                raise RuntimeError("distillation completed no epoch")
            student.load_state_dict(best_state)
            student.eval()
            node_squared = torch.zeros(config.data.num_nodes, dtype=torch.float64)
            teacher_squared = torch.zeros(config.data.num_nodes, dtype=torch.float64)
            with torch.no_grad():
                for features, target_correction in zip(
                    val_features.split(16), val_correction.split(16)
                ):
                    student_output = student(features.to(device)).cpu()
                    node_squared += (student_output - target_correction).double().square().sum(
                        dim=(0, 1, 3)
                    )
                    teacher_squared += target_correction.double().square().sum(dim=(0, 1, 3))
            node_count = val_correction.shape[0] * val_correction.shape[1] * val_correction.shape[3]
            node_rmse = torch.sqrt(node_squared / node_count).numpy()
            relative_rmse = float(
                torch.sqrt(node_squared.sum() / teacher_squared.sum().clamp_min(1e-12)).item()
            )
            prediction = _validation_prediction_metrics(
                teacher, student, val_inputs, val_targets,
                components.fixed_supports, components.data.scaler,
            )
            checkpoint_path = path / "checkpoints/student.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint_path)
            record = {
                "status": "completed",
                "seed": seed,
                "rank": rank,
                "teacher_checkpoint_sha256": _sha256(checkpoint),
                "code_commit": code_commit,
                "train_samples": train_features.shape[0],
                "validation_samples": val_features.shape[0],
                "best_epoch": best_epoch,
                "function_rmse": math_sqrt(best_loss),
                "function_relative_rmse": relative_rmse,
                "location_function_rmse": node_rmse.tolist(),
                "validation_prediction": prediction,
                "student_parameters": sum(p.numel() for p in student.parameters()),
            }
            write_json(metrics_path, record)
            all_records.append(record)
    all_records = []
    for seed in SEEDS:
        for rank in RANKS:
            path = root / "diagnostics/distillation" / f"seed{seed}" / f"r{rank}" / "metrics.json"
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("status") == "completed":
                    all_records.append(record)
    output = {"status": "partial", "records": all_records}
    if len(all_records) == len(SEEDS) * len(RANKS):
        output["status"] = "completed"
    write_json(root / "diagnostics/distillation/metrics.json", output)
    return output


def math_sqrt(value: float) -> float:
    return float(np.sqrt(value))
