"""遅延教師、SMB、episodic memory、AH周期を統合するオンライン適応器。"""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch
from torch import Tensor
from torch.optim import Optimizer

from ..config import ExperimentConfig
from ..data.scaler import GlobalStandardScaler
from ..models.dol import DOLForecastModel
from ..training.losses import original_scale_mae
from ..training.warmup import _resolve_device
from .delay import DelayedSupervisionQueue
from .replay import ReservoirReplayBuffer
from .schedule import AwakeHibernateSchedule


@dataclass(frozen=True)
class OnlineStepResult:
    step: int
    phase: str
    prediction: Tensor
    target: Tensor
    update_mae: float | None
    released_samples: int
    replay_size: int
    update_seconds: float = 0.0
    inference_seconds: float = 0.0


class OnlineAdapter:
    """1時点ずつ予測し、Awake時だけLSLを更新する。

    論文Algorithm 1に従い、online開始前に
    :meth:`seed_with_observed_batch` でvalidation標本をSMBへ登録できる。
    validation標本の正解はすでに観測済みなので、遅延キューは通さない。

    1ステップの処理順序は次のとおり。

    1. Hステップ前の予測に対応する教師を解禁してSMBへ入れる。
    2. AwakeかつSMBが非空なら、EMを抽出してLSLを更新する。
    3. 現在入力を予測する。
    4. 現在の教師をHステップ先まで遅延キューへ隠す。
    5. AH状態を進め、休眠へ入る境界でSMBをリセットする。

    この順序により、現在予測の未来教師が同じステップの更新に混ざらない。
    """

    def __init__(
        self,
        model: DOLForecastModel,
        fixed_supports: list[Tensor],
        scaler: GlobalStandardScaler,
        config: ExperimentConfig,
        optimizer: Optimizer,
        replay: ReservoirReplayBuffer,
        update_mode: str = "all",
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.config = config
        self.device = _resolve_device(config.runtime.device)
        self.model.to(self.device)
        self.model.freeze_for_online_adaptation(update_mode)
        self.update_mode = update_mode
        self.fixed_supports = [
            support.to(self.device, dtype=torch.float32) for support in fixed_supports
        ]

        # 公開実装はwarm-upで使ったoptimizerとその状態を
        # onlineにそのまま引き継ぐ。requires_gradはLSLだけ有効にする。
        self.optimizer = optimizer
        self.delay_queue = DelayedSupervisionQueue(
            delay_steps=config.data.forecast_horizon
        )
        self.replay = replay
        self.schedule = AwakeHibernateSchedule(
            awake_steps=config.online.awake_weeks * config.data.steps_per_week,
            hibernate_steps=config.online.hibernate_weeks * config.data.steps_per_week,
        )
        self.step = 0

    def seed_with_observed_batch(self, inputs: Tensor, target: Tensor) -> int:
        """観測済みvalidation標本をonline開始前のSMBへ登録する。

        論文Algorithm 1の3--4行目に対応する。バッチの各標本は
        reservoir samplingへ1件ずつ渡すため、validation標本数が容量を
        超えても最大 ``buffer_capacity`` 件の代表標本が残る。
        """

        if self.step != 0 or len(self.delay_queue) != 0:
            raise RuntimeError("validation memory can only be seeded before online starts")
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0)
        if target.ndim == 2:
            target = target.unsqueeze(0)
        if inputs.ndim != 3 or target.ndim != 3:
            raise ValueError("inputs and target must have shapes (batch, time, nodes)")
        if inputs.shape[0] != target.shape[0]:
            raise ValueError("inputs and target must have the same batch size")
        expected_inputs = (
            self.config.data.input_length,
            self.config.data.num_nodes,
        )
        expected_target = (
            self.config.data.forecast_horizon,
            self.config.data.num_nodes,
        )
        if tuple(inputs.shape[1:]) != expected_inputs:
            raise ValueError(
                f"expected input shape (*, {expected_inputs[0]}, {expected_inputs[1]}), "
                f"got {tuple(inputs.shape)}"
            )
        if tuple(target.shape[1:]) != expected_target:
            raise ValueError(
                f"expected target shape (*, {expected_target[0]}, {expected_target[1]}), "
                f"got {tuple(target.shape)}"
            )

        # 公開実装と同じくdevice上のfloat32へ変換してからSMBへ入れる。
        inputs = inputs.to(self.device, dtype=torch.float32)
        target = target.to(self.device, dtype=torch.float32)
        for batch_index in range(inputs.shape[0]):
            self.replay.add(inputs[batch_index], target[batch_index])
        return inputs.shape[0]

    def process(self, inputs: Tensor, target: Tensor) -> OnlineStepResult:
        """batch size 1の時系列窓を1つ処理する。"""

        if inputs.ndim != 3 or target.ndim != 3:
            raise ValueError("inputs and target must have shapes (1, time, nodes)")
        if inputs.shape[0] != 1 or target.shape[0] != 1:
            raise ValueError("online adaptation requires batch size 1")

        released = self.delay_queue.release(self.step)
        for sample in released:
            self.replay.add(sample.inputs, sample.target)

        phase = self.schedule.phase
        update_started = time.perf_counter()
        update_loss = (
            self._update_from_replay()
            if self.schedule.is_awake and self.update_mode != "none"
            else None
        )
        update_seconds = time.perf_counter() - update_started if update_loss is not None else 0.0

        inference_started = time.perf_counter()
        self.model.eval()
        inputs_device = inputs.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            prediction_normalized = self.model(inputs_device, self.fixed_supports)

        target_device = target.to(self.device, dtype=torch.float32)
        # rawのtest()と同じくCPUへ転送してから逆標準化・非負クリップする。
        prediction = self.scaler.inverse_transform(
            prediction_normalized.detach().cpu()
        ).clamp_min(0)
        target_original = self.scaler.inverse_transform(
            target_device.detach().cpu()
        ).clamp_min(0)
        inference_seconds = time.perf_counter() - inference_started

        self.delay_queue.submit(
            inputs=inputs_device.squeeze(0),
            target=target_device.squeeze(0),
            forecast_step=self.step,
        )

        transition = self.schedule.advance()
        if transition.entered_hibernation:
            # 論文のreset。以後の休眠期間に到着する標本を次の覚醒で使う。
            self.replay.clear()

        result = OnlineStepResult(
            step=self.step,
            phase=phase,
            prediction=prediction.detach().cpu(),
            target=target_original.detach().cpu(),
            update_mae=update_loss,
            released_samples=len(released),
            replay_size=len(self.replay),
            update_seconds=update_seconds,
            inference_seconds=inference_seconds,
        )
        self.step += 1
        return result

    def _update_from_replay(self) -> float | None:
        if len(self.replay) == 0:
            return None

        losses: list[float] = []
        # eval modeのまま勾配を取る。凍結BN統計と無効化Dropoutは公開実装と一致する。
        self.model.eval()
        for _ in range(self.config.online.update_steps):
            replay_batch = self.replay.sample(self.config.online.replay_size)
            inputs = replay_batch.inputs
            target = replay_batch.target

            prediction = self.model(inputs, self.fixed_supports)
            loss = original_scale_mae(prediction, target, self.scaler)
            loss.backward()
            self.optimizer.step()
            # 公開実装と同じくstep後に勾配を消去する。
            self.optimizer.zero_grad()
            losses.append(float(loss.detach().item()))
        return sum(losses) / len(losses)
