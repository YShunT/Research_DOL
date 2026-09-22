"""Streaming Memory Bufferをリザーバサンプリングで実装する。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class ReplaySample:
    inputs: Tensor
    target: Tensor


class ReservoirReplayBuffer:
    """到着済み教師ペアから最大M件を一様確率で保持する。

    rawのBufferと同じく、保存先device上に固定容量のTensorを確保する。
    """

    def __init__(self, capacity: int, device: torch.device | str = "cpu") -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.device = torch.device(device)
        self._inputs: Tensor | None = None
        self._targets: Tensor | None = None
        self.num_seen = 0

    def add(self, inputs: Tensor, target: Tensor) -> None:
        if inputs.ndim != 2 or target.ndim != 2:
            raise ValueError("inputs and target must have shapes (time, nodes)")
        if self._inputs is None or self._targets is None:
            self._inputs = torch.zeros(
                (self.capacity, *inputs.shape), device=self.device, dtype=torch.float32
            )
            self._targets = torch.zeros(
                (self.capacity, *target.shape), device=self.device, dtype=torch.float32
            )
        if inputs.shape != self._inputs.shape[1:] or target.shape != self._targets.shape[1:]:
            raise ValueError("replay sample shape changed")

        if self.num_seen < self.capacity:
            index = self.num_seen
        else:
            # 公開実装のreservoirはNumPyの大域乱数列を使う。
            index = int(np.random.randint(0, self.num_seen + 1))
        if index < self.capacity:
            self._inputs[index].copy_(inputs.detach())
            self._targets[index].copy_(target.detach())
        self.num_seen += 1

    def sample(self, size: int) -> ReplaySample:
        if size <= 0:
            raise ValueError("size must be positive")
        if len(self) == 0 or self._inputs is None or self._targets is None:
            raise RuntimeError("cannot sample from an empty replay buffer")
        sample_size = min(size, len(self))
        indices = np.random.choice(len(self), size=sample_size, replace=False)
        indices_device = torch.as_tensor(indices, device=self.device, dtype=torch.long)
        return ReplaySample(
            inputs=self._inputs.index_select(0, indices_device),
            target=self._targets.index_select(0, indices_device),
        )

    def clear(self) -> None:
        self._inputs = None
        self._targets = None
        self.num_seen = 0

    def __len__(self) -> int:
        return min(self.num_seen, self.capacity)
