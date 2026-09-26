"""設定からDOLのデータ・グラフ・モデル・学習器を組み立てる。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data.chicago import ChicagoDataBundle, prepare_chicago_data
from .evaluation.metrics import ForecastMetrics, compute_forecast_metrics
from .graph.supports import build_double_transition_supports, load_adjacency
from .models.dol import DOLForecastModel
from .online.adapter import OnlineAdapter, OnlineStepResult
from .online.replay import ReservoirReplayBuffer
from .training.warmup import (
    EpochRecord,
    WarmupResult,
    WarmupTrainer,
    _resolve_device,
    set_random_seed,
)


@dataclass(frozen=True)
class DOLComponents:
    config: ExperimentConfig
    data: ChicagoDataBundle
    model: DOLForecastModel
    fixed_supports: list[Tensor]
    optimizer: AdamW
    replay: ReservoirReplayBuffer


@dataclass(frozen=True)
class OnlineEvaluation:
    metrics: ForecastMetrics
    processed_steps: int
    online_updates: int
    awake_steps: int
    hibernate_steps: int
    validation_samples_seen: int
    initial_replay_size: int
    elapsed_seconds: float
    update_seconds: float = 0.0
    inference_seconds: float = 0.0
    predictions: Tensor | None = None
    targets: Tensor | None = None


def build_components(
    config: ExperimentConfig,
    location_learner_factory: Callable[[], nn.Module] | None = None,
) -> DOLComponents:
    """ファイル読込を含め、実験に必要な静的部品を構築する。"""

    device = _resolve_device(config.runtime.device)
    if device.type == "cuda":
        # cuda.manual_seedが公開実装と同じ対象deviceへ適用されるよう、
        # モデル構築とseed設定より前にcurrent deviceを選ぶ。
        torch.cuda.set_device(device)
    set_random_seed(config.warmup.seed, config.runtime.deterministic)
    data = prepare_chicago_data(config.data)
    adjacency = load_adjacency(
        config.data.adjacency_path,
        expected_num_nodes=config.data.num_nodes,
    )
    # 公開実装は隣接行列を一度float32へ変換した後、
    # float64の単位行列を加えて正規化し、最後にfloat32へ戻す。
    fixed_supports = build_double_transition_supports(adjacency.double())
    fixed_supports = [
        support.to(device=device, dtype=torch.float32)
        for support in fixed_supports
    ]
    location_learner = location_learner_factory() if location_learner_factory else None
    model = DOLForecastModel(
        config.data, config.model, location_learner=location_learner
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.warmup.learning_rate,
        weight_decay=config.warmup.weight_decay,
    )
    replay = ReservoirReplayBuffer(config.online.buffer_capacity, device=device)
    return DOLComponents(
        config=config,
        data=data,
        model=model,
        fixed_supports=fixed_supports,
        optimizer=optimizer,
        replay=replay,
    )


def build_dataloaders(
    components: DOLComponents,
) -> tuple[
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
]:
    """warm-upはshuffle、validation/onlineは時系列順でLoaderを作る。"""

    train = DataLoader(
        components.data.train,
        batch_size=components.config.warmup.batch_size,
        shuffle=True,
        drop_last=True,
    )
    validation = DataLoader(
        components.data.validation,
        batch_size=components.config.warmup.batch_size,
        shuffle=False,
        drop_last=False,
    )
    online = DataLoader(
        components.data.online,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    return train, validation, online


def run_warmup(
    components: DOLComponents,
    checkpoint_path: Path | None = None,
    epoch_callback: Callable[[EpochRecord], None] | None = None,
) -> WarmupResult:
    train_loader, validation_loader, _ = build_dataloaders(components)
    trainer = WarmupTrainer(
        model=components.model,
        fixed_supports=components.fixed_supports,
        scaler=components.data.scaler,
        config=components.config,
        optimizer=components.optimizer,
        replay=components.replay,
    )
    return trainer.fit(
        train_loader,
        validation_loader,
        checkpoint_path,
        epoch_callback=epoch_callback,
    )


def load_warmup_checkpoint(components: DOLComponents, checkpoint_path: Path) -> None:
    """オンライン評価前にwarm-up checkpointとscalerを復元する。"""

    trainer = WarmupTrainer(
        model=components.model,
        fixed_supports=components.fixed_supports,
        scaler=components.data.scaler,
        config=components.config,
        optimizer=components.optimizer,
        replay=components.replay,
    )
    trainer.load_checkpoint(checkpoint_path, load_optimizer=False)


def run_online_evaluation(
    components: DOLComponents,
    max_steps: int | None = None,
    collect_predictions: bool = False,
    progress_interval: int | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    seed_validation_before_online: bool = False,
    update_mode: str = "all",
    step_callback: Callable[[OnlineStepResult], None] | None = None,
) -> OnlineEvaluation:
    """online区間を時系列順に予測・適応し、指標を集計する。

    rawの通常train→test経路ではwarm-up中のSMBをそのまま使い、
    checkpointからmode=onlineで始める経路だけvalidationを再投入する。
    """

    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when specified")
    if progress_interval is not None and progress_interval <= 0:
        raise ValueError("progress_interval must be positive when specified")
    _, validation_loader, online_loader = build_dataloaders(components)
    adapter = OnlineAdapter(
        model=components.model,
        fixed_supports=components.fixed_supports,
        scaler=components.data.scaler,
        config=components.config,
        optimizer=components.optimizer,
        replay=components.replay,
        update_mode=update_mode,
    )

    # rawのtest()はmode=onlineのときだけvali()を追加で呼ぶ。
    if seed_validation_before_online:
        for inputs, target in validation_loader:
            adapter.seed_with_observed_batch(inputs, target)
    validation_samples_seen = components.replay.num_seen
    initial_replay_size = len(adapter.replay)

    processed = 0
    updates = 0
    awake_steps = 0
    hibernate_steps = 0
    update_seconds = 0.0
    inference_seconds = 0.0
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    total_steps = min(len(online_loader), max_steps or len(online_loader))
    started_at = time.perf_counter()
    for inputs, target in online_loader:
        result = adapter.process(inputs, target)
        if step_callback is not None:
            step_callback(result)
        predictions.append(result.prediction)
        targets.append(result.target)
        processed += 1
        updates += int(result.update_mae is not None)
        awake_steps += int(result.phase == "awake")
        hibernate_steps += int(result.phase == "hibernate")
        update_seconds += result.update_seconds
        inference_seconds += result.inference_seconds
        if (
            progress_callback is not None
            and progress_interval is not None
            and (processed % progress_interval == 0 or processed == total_steps)
        ):
            progress_callback(processed, total_steps, updates)
        if max_steps is not None and processed >= max_steps:
            break

    elapsed_seconds = time.perf_counter() - started_at
    if not predictions:
        raise RuntimeError("no online predictions were evaluated")
    prediction_tensor = torch.cat(predictions, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    del predictions, targets

    return OnlineEvaluation(
        metrics=compute_forecast_metrics(prediction_tensor, target_tensor),
        processed_steps=processed,
        online_updates=updates,
        awake_steps=awake_steps,
        hibernate_steps=hibernate_steps,
        validation_samples_seen=validation_samples_seen,
        initial_replay_size=initial_replay_size,
        elapsed_seconds=elapsed_seconds,
        update_seconds=update_seconds,
        inference_seconds=inference_seconds,
        predictions=prediction_tensor if collect_predictions else None,
        targets=target_tensor if collect_predictions else None,
    )
