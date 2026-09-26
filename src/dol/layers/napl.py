"""NAPL型の低ランク地点別LSL。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .LSL import NodeAdapter


def _adapter_vector(channels: int, bottleneck: int, dropout: float) -> Tensor:
    adapter = NodeAdapter(channels, bottleneck, dropout)
    return torch.cat(
        (
            adapter.down_projection.weight.detach().flatten(),
            adapter.down_projection.bias.detach(),
            adapter.up_projection.weight.detach().flatten(),
            adapter.up_projection.bias.detach(),
        )
    )


class ZeroLocationLearner(nn.Module):
    """LSLなし対照。入力埋め込みへの補正を0にする。"""

    def forward(self, features: Tensor) -> Tensor:
        return torch.zeros_like(features)


class NAPLLocationLearner(nn.Module):
    """地点重みを theta_shared + E @ B または E @ B で生成する。"""

    def __init__(
        self,
        num_nodes: int,
        channels: int,
        bottleneck_channels: int,
        dropout: float,
        rank: int,
        with_shared: bool,
    ) -> None:
        super().__init__()
        if min(num_nodes, channels, bottleneck_channels) <= 0 or rank < 0:
            raise ValueError("invalid NAPL dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if rank == 0 and not with_shared:
            raise ValueError("rank 0 without shared weights is the no-LSL control")
        self.num_nodes = num_nodes
        self.channels = channels
        self.bottleneck_channels = bottleneck_channels
        self.dropout = dropout
        self.rank = rank
        self.with_shared = with_shared
        self.weight_count = 2 * channels * bottleneck_channels + bottleneck_channels + channels

        if with_shared:
            self.theta_shared = nn.Parameter(
                _adapter_vector(channels, bottleneck_channels, dropout)
            )
        else:
            self.register_parameter("theta_shared", None)
        if rank:
            self.E = nn.Parameter(torch.empty(num_nodes, rank))
            self.B = nn.Parameter(torch.empty(rank, self.weight_count))
            with torch.no_grad():
                if with_shared:
                    # warm-up開始時は通常の共有LSLの近くに置く。
                    nn.init.normal_(self.E, std=0.01)
                    nn.init.normal_(self.B, std=0.01)
                else:
                    # E@Bの初期重みを独立LSLと同程度のスケールにする。
                    nn.init.normal_(self.E, std=1 / math.sqrt(rank))
                    self.B.copy_(
                        torch.stack(
                            [
                                _adapter_vector(channels, bottleneck_channels, dropout)
                                for _ in range(rank)
                            ]
                        )
                    )
        else:
            self.register_parameter("E", None)
            self.register_parameter("B", None)

    def generated_weights(self) -> Tensor:
        if self.rank:
            weights = self.E @ self.B
        else:
            assert self.theta_shared is not None
            weights = self.theta_shared.new_zeros(self.num_nodes, self.weight_count)
        if self.theta_shared is not None:
            weights = weights + self.theta_shared.unsqueeze(0)
        return weights

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 4 or features.shape[1:3] != (self.channels, self.num_nodes):
            raise ValueError("features must have shape (batch, channels, nodes, time)")
        c, m = self.channels, self.bottleneck_channels
        weights = self.generated_weights()
        split1 = m * c
        split2 = split1 + m
        split3 = split2 + c * m
        down_weight = weights[:, :split1].reshape(self.num_nodes, m, c)
        down_bias = weights[:, split1:split2]
        up_weight = weights[:, split2:split3].reshape(self.num_nodes, c, m)
        up_bias = weights[:, split3:]

        x = features.permute(0, 2, 3, 1)  # (batch, node, time, channel)
        hidden = torch.einsum("bntc,nmc->bntm", x, down_weight)
        hidden = hidden + down_bias[None, :, None, :]
        hidden = F.relu(F.dropout(hidden, p=self.dropout, training=self.training))
        output = torch.einsum("bntm,ncm->bntc", hidden, up_weight)
        output = output + up_bias[None, :, None, :]
        return output.permute(0, 3, 1, 2)
