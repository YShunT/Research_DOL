"""Graph WaveNet型の拡散Graph Convolution。

公開実装 ``layers/gcn.py`` と同じく、ノード軸の右から
支持行列を掛ける ``X @ A`` で伝播する。入力と出力の
基本shapeは ``(batch, channels, nodes, time)``。
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor


class NodePropagation(nn.Module):
    """公開実装の ``einsum('ncvl,vw->ncwl')`` を明示的に表す。"""

    def forward(self, features: Tensor, support: Tensor) -> Tensor:
        self._validate_inputs(features, support)
        return torch.einsum("bcvt,vw->bcwt", features, support).contiguous()

    @staticmethod
    def _validate_inputs(features: Tensor, support: Tensor) -> None:
        if features.ndim != 4:
            raise ValueError(
                "features must have shape (batch, channels, nodes, time), "
                f"got {tuple(features.shape)}"
            )
        if support.ndim != 2 or support.shape[0] != support.shape[1]:
            raise ValueError(
                "support must have shape (nodes, nodes), "
                f"got {tuple(support.shape)}"
            )
        if features.shape[2] != support.shape[0]:
            raise ValueError(
                f"node mismatch: features has {features.shape[2]}, "
                f"support has {support.shape[0]}"
            )
        if features.device != support.device:
            raise ValueError("features and support must be on the same device")
        if features.dtype != support.dtype:
            raise ValueError("features and support must use the same dtype")


class DiffusionGraphConvolution(nn.Module):
    """複数の支持行列についてk次まで伝播し、1×1 Convで統合する。

    原DOLの既定値は ``support_count=3``、``diffusion_order=2``。入力自身に
    6個の拡散結果を加えて連結するため、projection入力は元の7倍になる。
        ノード軸の表記では X W_0 + sum_s sum_{k=1}^K X P_s^k W_{s,k}。
    1×1 Convの重みは(out_channels, expanded_channels, 1, 1)であり、
    論文のW(in_channels, out_channels)とはチャネル軸の配置が逆になる。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        support_count: int,
        diffusion_order: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("channel sizes must be positive")
        if support_count <= 0:
            raise ValueError("support_count must be positive")
        if diffusion_order <= 0:
            raise ValueError("diffusion_order must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.support_count = support_count
        self.diffusion_order = diffusion_order
        self.propagation = NodePropagation()

        self.expanded_channels = in_channels * (
            1 + support_count * diffusion_order
        )
        self.projection = nn.Conv2d(
            in_channels=self.expanded_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: Tensor, supports: Sequence[Tensor]) -> Tensor:
        if features.ndim != 4:
            raise ValueError(
                "features must have shape (batch, channels, nodes, time), "
                f"got {tuple(features.shape)}"
            )
        if features.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, "
                f"got {features.shape[1]}"
            )
        if len(supports) != self.support_count:
            raise ValueError(
                f"expected {self.support_count} supports, got {len(supports)}"
            )

        outputs = [features]
        for support in supports:
            propagated = features
            for _ in range(self.diffusion_order):
                propagated = self.propagation(propagated, support)
                outputs.append(propagated)

        combined = torch.cat(outputs, dim=1)
        projected = self.projection(combined)
        return self.dropout(projected)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class AdaptiveAdjacency(nn.Module):
    """学習する低ランク因子から行確率型の隣接行列を生成する。

    ``softmax(ReLU(E_source @ E_target), dim=1)`` は原DOLと同じ。ここでの
    rankはグラフ生成用埋め込み次元であり、NAPL-R8のrankとは別である。
    出力P[i, j]は行ごとに正規化され、公開実装と同じく
    特徴のノード軸へ右から掛けられる。
    """

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 10,
        initial_adjacency: Optional[Tensor] = None,
    ) -> None:
        super().__init__()

        if num_nodes <= 0 or embedding_dim <= 0:
            raise ValueError("num_nodes and embedding_dim must be positive")
        if embedding_dim > num_nodes:
            raise ValueError("embedding_dim cannot exceed num_nodes")

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.source_embedding = nn.Parameter(torch.empty(num_nodes, embedding_dim))
        self.target_embedding = nn.Parameter(torch.empty(embedding_dim, num_nodes))
        self.reset_parameters(initial_adjacency)

    def reset_parameters(self, initial_adjacency: Optional[Tensor] = None) -> None:
        with torch.no_grad():
            if initial_adjacency is None:
                self.source_embedding.normal_()
                self.target_embedding.normal_()
                return

            self._validate_initial_adjacency(initial_adjacency)
            adjacency = initial_adjacency.to(
                dtype=self.source_embedding.dtype,
                device=self.source_embedding.device,
            )
            left, singular_values, right_transpose = torch.linalg.svd(
                adjacency,
                full_matrices=False,
            )
            scale = torch.diag(singular_values[: self.embedding_dim].sqrt())
            self.source_embedding.copy_(
                left[:, : self.embedding_dim] @ scale
            )
            self.target_embedding.copy_(
                scale @ right_transpose[: self.embedding_dim, :]
            )

    def forward(self) -> Tensor:
        logits = torch.relu(self.source_embedding @ self.target_embedding)
        return torch.softmax(logits, dim=1)

    def _validate_initial_adjacency(self, adjacency: Tensor) -> None:
        if adjacency.ndim != 2 or tuple(adjacency.shape) != (
            self.num_nodes,
            self.num_nodes,
        ):
            raise ValueError(
                "initial_adjacency must have shape "
                f"({self.num_nodes}, {self.num_nodes}), got {tuple(adjacency.shape)}"
            )
        if not torch.is_floating_point(adjacency):
            raise TypeError("initial_adjacency must use a floating-point dtype")
