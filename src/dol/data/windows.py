"""連続時系列をwarm-up・validation・online用の窓へ変換する。"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor
from torch.utils.data import Dataset

from ..config import DataConfig


@dataclass(frozen=True)
class SplitBoundaries:
    """時刻軸上の半開区間 ``[start, end)`` を表す。"""

    train: tuple[int, int]
    validation: tuple[int, int]
    online: tuple[int, int]

    def validate(self, total_steps: int) -> None:
        intervals = (self.train, self.validation, self.online)
        if intervals[0][0] != 0 or intervals[-1][1] != total_steps:
            raise ValueError("splits must cover the complete series")
        if intervals[0][1] != intervals[1][0] or intervals[1][1] != intervals[2][0]:
            raise ValueError("splits must be contiguous and non-overlapping")
        if any(start >= end for start, end in intervals):
            raise ValueError("every split must contain at least one time step")


def compute_split_boundaries(total_steps: int, config: DataConfig) -> SplitBoundaries:
    """公開実装と同じ20%/5%/75%の境界を計算する。

    丸め誤差はvalidationへ割り当てる。具体的にはtrainとonlineの長さを
    ``int`` で決め、残りをvalidationとする。
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    train_steps = int(total_steps * config.train_ratio)
    online_steps = int(total_steps * config.online_ratio)
    validation_steps = total_steps - train_steps - online_steps

    boundaries = SplitBoundaries(
        train=(0, train_steps),
        validation=(train_steps, train_steps + validation_steps),
        online=(train_steps + validation_steps, total_steps),
    )
    boundaries.validate(total_steps)
    return boundaries


class TimeWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """多地点系列から過去窓と未来窓を時系列順に返す。

    Parameters
    ----------
    series:
        標準化済み系列。shapeは ``(total_steps, num_nodes)``。
    target_interval:
        予測対象の先頭時刻が属する分割区間 ``[start, end)``。
        validation/onlineの最初の予測でも、入力には直前分割の履歴を使える。
    input_length:
        過去窓長L。
    forecast_horizon:
        未来窓長H。教師の全H点が区間内に収まる窓だけを生成する。

    戻り値 ``x`` と ``y`` はそれぞれ ``(L, N)``、``(H, N)``。
    """

    def __init__(
        self,
        series: Tensor,
        target_interval: tuple[int, int],
        input_length: int,
        forecast_horizon: int,
    ) -> None:
        if series.ndim != 2:
            raise ValueError(
                "series must have shape (time, nodes), "
                f"got {tuple(series.shape)}"
            )
        if input_length <= 0 or forecast_horizon <= 0:
            raise ValueError("window lengths must be positive")

        start, end = target_interval
        if not 0 <= start < end <= series.shape[0]:
            raise ValueError("target_interval is outside the series")

        self.series = series
        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.first_target_start = max(start, input_length)
        self.last_target_start = end - forecast_horizon
        if self.last_target_start < self.first_target_start:
            raise ValueError("split is too short to form a complete forecast window")

    def __len__(self) -> int:
        return self.last_target_start - self.first_target_start + 1

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)

        target_start = self.first_target_start + index
        input_start = target_start - self.input_length
        target_end = target_start + self.forecast_horizon
        x = self.series[input_start:target_start]
        y = self.series[target_start:target_end]
        return x, y

    def forecast_origin(self, index: int) -> int:
        """窓の入力末尾の次、すなわち教師窓の先頭時刻を返す。"""

        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.first_target_start + index
