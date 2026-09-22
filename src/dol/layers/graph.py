
import torch
from torch import Tensor
import torch.nn as nn
from typing import Sequence


class NodePropagation(nn.Module):
    """
    公開実装と同じ ``X @ A`` のノード伝播。
    Input : (B,C,N,T)
    Output : (B,C,N,T)
    """

    def forward(self,features: Tensor, support: Tensor) -> Tensor:
        #self._validate_inputs(features,support)
        return torch.einsum("bcvt,vw->bcwt", features, support).contiguous()


class DiffusionGraphConvolution(nn.Module):
    """
    複数の支持行列についてk次まで伝播し、1×1 Convで統合する。
    Z = X W_0 + sum_s sum_{k=1}^K X P_s^k W_{s,k}
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            support_count: int,
            diffusion_order: int =2,
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

        self.expanded_channels = in_channels * ( 1 + support_count * diffusion_order)

        self.projection = nn.Conv2d(
            in_channels = self.expanded_channels,
            out_channels = out_channels,
            kernel_size = 1,
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

        # 各項は(B,C_in,N,T)。連結後は(B,(1+support_count*K)*C_in,N,T)。
        combined = torch.cat(outputs, dim=1)
        # 各地点・時刻でチャネルを変換し、(B,C_out,N,T)へ戻す。
        return self.dropout(self.projection(combined))


# 以前のスペルでimportしていたコードとの互換性を保つ。
DifuusionGraphConvolution = DiffusionGraphConvolution
