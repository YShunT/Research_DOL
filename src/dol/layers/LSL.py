"""
DOLの構造を実装する
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import List

"""
nn.Conv1d -> (in_channels, out_channels, kernel_size)
"""

class NodeAdapter(nn.Module):
    """
    1地点の特徴を変換する32->4->32のMLP
    Kernelsizeは1のConvadを使用

    Args:
        channels: 入出力の特徴チャネル数。
        bottleneck_channels: 中間チャネル数。
        dropout: down projection後に適用するDropout率。

    Input: (batchsize, channles, time_steps)
    Output: (batchsize, channles, time_steps)

    """

    def __init__(
            self,
            channels: int = 32,
            bottleneck_channels: int = 4,
            dropout: float = 0.5,
    ) -> None:
        super().__init__()

        if channels <=0 or bottleneck_channels <= 0:
            raise ValueError("channels sizes must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in (0,1)")

        self.channels = channels
        self.bottleneck_channels = bottleneck_channels
        self.down_projection = nn.Conv1d(
            in_channels = channels,
            out_channels = bottleneck_channels,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.up_projection = nn.Conv1d(
            in_channels = bottleneck_channels,
            out_channels = channels,
            kernel_size = 1,
        )

    def forward(self, node_features: Tensor) -> Tensor:
        """
        1地点の補正量を返す
        """

        if node_features.ndim != 3:
            raise ValueError("node_features must have 3 dims")

        if node_features.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {node_features.shape[1]}")

        hidden = self.down_projection(node_features)
        hidden = self.dropout(hidden)
        hidden = self.activation(hidden)
        return self.up_projection(hidden)


class LocationSpecificLearner(nn.Module):
    """
    各地点に独立したNodeAdapterを割り当てる

    Input: (batch_size, channels, nodes, time_step)
    Output: (batch_size, channels, nodes, time_step)
    """

    def __init__(
            self,
            num_nodes: int = 77,
            channels: int = 32,
            bottleneck_channels: int = 4,
            dropout: float = 0.5,
    ) -> None:
        super().__init__()

        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")

        self.num_nodes = num_nodes
        self.channels = channels

        #NodeAdapterをNNに学習させるためにModuleListに格納
        self.adapters = nn.ModuleList(
            NodeAdapter(channels, bottleneck_channels, dropout)
            for _ in range(num_nodes) #各地点に独立に適応
        )

    def forward(self, features: Tensor) -> Tensor:
        """
        全地点の補正値を返す
        Arges : features[batch_size,channels,nodes,time_step]
        """

        self._validate(features)

        node_outputs: List[Tensor] = [
            adapter(features[:, :, node_index, :])
            for node_index, adapter in enumerate(self.adapters)
        ]
        return torch.stack(node_outputs, dim=2) #node軸に合体し、形状を合わせる

    #先頭に_をつけることでclass内特有の定義であることを暗示
    def _validate(self, features: Tensor) -> None:
        if features.ndim != 4:
            raise ValueError("features must have chape (batch_size,channels,nodes,time_step)")

        if features.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {features.shape[1]}")
        if features.shape[2] != self.num_nodes:
            raise ValueError(f"expected {self.num_nodes} nodes, got {features.shape[2]}")

    @property #読み取り専用へ
    def parameter_count(self) -> int:
        """
        LSLが保持するパラメータ数を返す
        """
        #parameter.numel()はパラメータ要素数を計算する
        return sum(parameter.numel() for parameter in self.parameters())
