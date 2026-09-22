"""予測ホライズン分だけ教師の利用を遅らせ、未来漏洩を防ぐ。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class DelayedSample:
    """全H点の正解が利用可能になった入力・教師ペア。"""

    inputs: Tensor
    target: Tensor
    forecast_step: int
    available_step: int


class DelayedSupervisionQueue:
    """時刻sで発行したH-step予測を時刻s+Hまで隠す。

    オフライン評価用Datasetは未来教師を同時に返すが、このqueueへ格納した
    教師は ``available_step`` より前には取り出せない。これにより、コード上で
    online更新が未来を参照しないことを保証する。
    """

    def __init__(self, delay_steps: int) -> None:
        if delay_steps <= 0:
            raise ValueError("delay_steps must be positive")
        self.delay_steps = delay_steps
        self._queue: deque[DelayedSample] = deque()

    def submit(
        self,
        inputs: Tensor,
        target: Tensor,
        forecast_step: int,
    ) -> None:
        if inputs.ndim != 2 or target.ndim != 2:
            raise ValueError("inputs and target must have shapes (time, nodes)")
        if forecast_step < 0:
            raise ValueError("forecast_step must be non-negative")
        self._queue.append(
            DelayedSample(
                inputs=inputs.detach().clone(),
                target=target.detach().clone(),
                forecast_step=forecast_step,
                available_step=forecast_step + self.delay_steps,
            )
        )

    def release(self, current_step: int) -> list[DelayedSample]:
        """現在までに教師窓全体が観測済みとなった標本を返す。"""

        if current_step < 0:
            raise ValueError("current_step must be non-negative")
        released: list[DelayedSample] = []
        while self._queue and self._queue[0].available_step <= current_step:
            released.append(self._queue.popleft())
        return released

    def clear(self) -> None:
        self._queue.clear()

    def __len__(self) -> int:
        return len(self._queue)
