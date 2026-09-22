"""DOL全体を初期期間で学習するwarm-up trainer。"""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from ..config import ExperimentConfig
from ..data.scaler import GlobalStandardScaler
from ..models.dol import DOLForecastModel
from ..online.replay import ReservoirReplayBuffer
from .losses import original_scale_mae


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """公開実装と同じ順番とoffsetで乱数seedを設定する。"""

    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + 3)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_mae: float
    validation_mae: float


@dataclass(frozen=True)
class WarmupResult:
    best_epoch: int
    best_validation_mae: float
    history: tuple[EpochRecord, ...]
    checkpoint_path: Path | None


class WarmupTrainer:
    """訓練20%で全パラメータを学習し、検証5%で早期停止する。"""

    def __init__(
        self,
        model: DOLForecastModel,
        fixed_supports: list[Tensor],
        scaler: GlobalStandardScaler,
        config: ExperimentConfig,
        optimizer: Optimizer | None = None,
        replay: ReservoirReplayBuffer | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.scaler = scaler
        self.device = _resolve_device(config.runtime.device)
        self.model.to(self.device)
        self.fixed_supports = [
            support.to(self.device, dtype=torch.float32) for support in fixed_supports
        ]
        self.optimizer = (
            optimizer
            if optimizer is not None
            else AdamW(
                self.model.parameters(),
                lr=config.warmup.learning_rate,
                weight_decay=config.warmup.weight_decay,
            )
        )
        self.replay = replay

    def fit(
        self,
        train_loader: DataLoader[tuple[Tensor, Tensor]],
        validation_loader: DataLoader[tuple[Tensor, Tensor]],
        checkpoint_path: Path | None = None,
        epoch_callback: Callable[[EpochRecord], None] | None = None,
    ) -> WarmupResult:
        """early stoppingまで学習し、最良重みをmodelへ復元する。"""

        self.model.unfreeze_for_warmup()
        best_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, Tensor] | None = None
        patience_count = 0
        history: list[EpochRecord] = []

        for epoch in range(1, self.config.warmup.max_epochs + 1):
            train_loss = self._train_epoch(train_loader)
            validation_loss = self.evaluate(
                validation_loader,
                populate_replay=True,
            )
            record = EpochRecord(epoch, train_loss, validation_loss)
            history.append(record)
            if epoch_callback is not None:
                epoch_callback(record)

            if validation_loss <= best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                patience_count = 0
                best_state = copy.deepcopy(self.model.state_dict())
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path, epoch, validation_loss)
            else:
                patience_count += 1
                if patience_count >= self.config.warmup.patience:
                    break

            self._adjust_learning_rate(epoch)

        if best_state is None:
            raise RuntimeError("warm-up did not complete any epoch")
        self.model.load_state_dict(best_state)
        return WarmupResult(
            best_epoch=best_epoch,
            best_validation_mae=best_loss,
            history=tuple(history),
            checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        )

    def _train_epoch(self, loader: DataLoader[tuple[Tensor, Tensor]]) -> float:
        self.model.train()
        losses: list[float] = []
        for inputs, target in loader:
            inputs = inputs.to(self.device, dtype=torch.float32)
            target = target.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.model(inputs, self.fixed_supports)
            loss = original_scale_mae(prediction, target, self.scaler)
            loss.backward()
            self.optimizer.step()

            losses.append(float(loss.detach().item()))
        if not losses:
            raise ValueError("training loader is empty")
        return float(np.mean(losses))

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader[tuple[Tensor, Tensor]],
        populate_replay: bool = False,
    ) -> float:
        """検証MAEを返し、warm-up中は公開実装と同じくSMBも更新する。"""

        self.model.eval()
        losses: list[float] = []
        for inputs, target in loader:
            inputs = inputs.to(self.device, dtype=torch.float32)
            target = target.to(self.device, dtype=torch.float32)
            prediction = self.model(inputs, self.fixed_supports)
            if populate_replay:
                if self.replay is None:
                    raise RuntimeError("replay buffer is required during warm-up validation")
                for batch_index in range(inputs.shape[0]):
                    self.replay.add(inputs[batch_index], target[batch_index])
            # rawのvali()はCPU上の標準化スケールでMAEを計算する。
            loss = (prediction.detach().cpu() - target.detach().cpu()).abs().mean()
            losses.append(float(loss.item()))
        if not losses:
            raise ValueError("validation loader is empty")
        self.model.train()
        return float(np.mean(losses))

    def _adjust_learning_rate(self, epoch: int) -> None:
        """公開実装の ``type1`` scheduleと同じくepochごとに0.5倍する。"""

        learning_rate = self.config.warmup.learning_rate * (
            self.config.warmup.learning_rate_decay ** (epoch - 1)
        )
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate

    def save_checkpoint(self, path: Path, epoch: int, validation_mae: float) -> None:
        """再開・再現に必要な状態を1ファイルへ保存する。"""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "epoch": epoch,
            "validation_mae": validation_mae,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": asdict(self.config),
            "seed": self.config.warmup.seed,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: Path, load_optimizer: bool = False) -> dict[str, Any]:
        """保存したwarm-up checkpointを読み込む。"""

        payload = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])
        self.scaler.load_state_dict(payload["scaler"])
        if load_optimizer:
            self.optimizer.load_state_dict(payload["optimizer"])
        return payload


def _resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"{name} was requested, but CUDA is unavailable. "
            "Set RuntimeConfig(device='cpu') for a CPU check."
        )
    return device
