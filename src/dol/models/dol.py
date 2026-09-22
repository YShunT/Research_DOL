"""Location-Specific Learnerを備えたGraph WaveNet型DOLモデル。"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ..config import DataConfig, ModelConfig
from ..layers.LSL import LocationSpecificLearner
from ..layers.graph_convolution import AdaptiveAdjacency, DiffusionGraphConvolution


class GraphWaveNetBlock(nn.Module):
    """gated temporal convolution、graph convolution、残差を1層にまとめる。

    入力 ``x`` は ``(B, C_res, N, T)``。時間カーネルにpaddingを置かないため、
    出力時間長は ``T - dilation * (kernel_size - 1)`` になる。
    """

    def __init__(
        self,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        support_count: int,
        diffusion_order: int,
        graph_dropout: float,
        temporal_kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        if dilation <= 0:
            raise ValueError("dilation must be positive")

        temporal_arguments = {
            "in_channels": residual_channels,
            "out_channels": dilation_channels,
            "kernel_size": (1, temporal_kernel_size),
            "dilation": (1, dilation),
        }
        self.filter_convolution = nn.Conv2d(**temporal_arguments)
        self.gate_convolution = nn.Conv2d(**temporal_arguments)
        # 公開実装はGCN経路ではforwardに使わないが、学習パラメータと
        # 初期化順を一致させるため保持する。
        self.residual_projection = nn.Conv2d(
            dilation_channels,
            residual_channels,
            kernel_size=1,
        )
        self.skip_projection = nn.Conv2d(
            dilation_channels,
            skip_channels,
            kernel_size=1,
        )
        self.normalization = nn.BatchNorm2d(residual_channels)
        self.graph_convolution = DiffusionGraphConvolution(
            in_channels=dilation_channels,
            out_channels=residual_channels,
            support_count=support_count,
            diffusion_order=diffusion_order,
            dropout=graph_dropout,
        )
        self.dilation = dilation

    def forward(
        self,
        x: Tensor,
        supports: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor]:
        residual = x

        filter_values = torch.tanh(self.filter_convolution(residual))
        gate_values = torch.sigmoid(self.gate_convolution(residual))
        gated = filter_values * gate_values

        skip = self.skip_projection(gated)
        spatial = self.graph_convolution(gated, supports)
        residual_tail = residual[..., -spatial.shape[-1] :]
        output = self.normalization(spatial + residual_tail)
        return output, skip


class DOLForecastModel(nn.Module):
    """過去L点から全地点の未来H点を予測するDOL。

    Parameters
    ----------
    data_config:
        ノード数、入力長、予測ホライズン。
    model_config:
        Graph WaveNetとLSLの構造。
    adaptive_initial_adjacency:
        学習隣接行列をSVD初期化したい場合の ``(N,N)`` 行列。省略時は
        原実装と同じランダム初期化。

    Input
    -----
    ``inputs``: ``(batch, input_length, num_nodes)``

    Output
    ------
    ``prediction``: ``(batch, forecast_horizon, num_nodes)``
    """

    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        adaptive_initial_adjacency: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.model_config = model_config
        self.num_nodes = data_config.num_nodes
        self.forecast_horizon = data_config.forecast_horizon
        self.receptive_field = model_config.receptive_field

        self.input_projection= nn.Conv2d(
            in_channels=1,
            out_channels=model_config.residual_channels,
            kernel_size=1,
        )
        self.location_specific = LocationSpecificLearner(
            num_nodes=data_config.num_nodes,
            channels=model_config.residual_channels,
            bottleneck_channels=model_config.lsl_bottleneck_channels,
            dropout=model_config.lsl_dropout,
        )

        self.adaptive_adjacency: AdaptiveAdjacency | None
        if model_config.use_adaptive_graph:
            self.adaptive_adjacency = AdaptiveAdjacency(
                num_nodes=data_config.num_nodes,
                embedding_dim=model_config.adaptive_graph_rank,
                initial_adjacency=adaptive_initial_adjacency,
            )
        else:
            self.adaptive_adjacency = None

        blocks: list[GraphWaveNetBlock] = []
        for _ in range(model_config.blocks):
            for layer_index in range(model_config.layers_per_block):
                blocks.append(
                    GraphWaveNetBlock(
                        residual_channels=model_config.residual_channels,
                        dilation_channels=model_config.dilation_channels,
                        skip_channels=model_config.skip_channels,
                        support_count=model_config.support_count,
                        diffusion_order=model_config.diffusion_order,
                        graph_dropout=model_config.graph_dropout,
                        temporal_kernel_size=model_config.temporal_kernel_size,
                        dilation=2**layer_index,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

        self.output_projection = nn.Sequential(
            nn.Conv2d(
                model_config.skip_channels,
                model_config.end_channels,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                model_config.end_channels,
                data_config.forecast_horizon,
                kernel_size=1,
            ),
        )

    def forward(self, inputs: Tensor, fixed_supports: Sequence[Tensor]) -> Tensor:
        self._validate_inputs(inputs, fixed_supports)

        # (B,L,N) -> (B,1,N,L)
        x = inputs.unsqueeze(1).transpose(2, 3)
        if x.shape[-1] < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - x.shape[-1], 0, 0, 0))

        embedded = self.input_projection(x)
        node_correction = self.location_specific(embedded)
        x = embedded + node_correction

        supports = list(fixed_supports)
        if self.adaptive_adjacency is not None:
            supports.append(self.adaptive_adjacency())

        skip_total: Tensor | None = None
        for block in self.blocks:
            x, skip = block(x, supports)
            if skip_total is None:
                skip_total = skip
            else:
                skip_total = skip_total[..., -skip.shape[-1] :] + skip

        if skip_total is None:
            raise RuntimeError("DOL must contain at least one Graph WaveNet block")

        # 入力が受容野より長い場合も、最新の予測起点だけを出力する。
        last_origin = torch.relu(skip_total[..., -1:])
        prediction = self.output_projection(last_origin)
        return prediction.squeeze(-1)

    def _validate_inputs(
        self,
        inputs: Tensor,
        fixed_supports: Sequence[Tensor],
    ) -> None:
        if inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape (batch, time, nodes), "
                f"got {tuple(inputs.shape)}"
            )
        if inputs.shape[2] != self.num_nodes:
            raise ValueError(
                f"expected {self.num_nodes} nodes, got {inputs.shape[2]}"
            )
        if len(fixed_supports) != self.model_config.fixed_support_count:
            raise ValueError(
                f"expected {self.model_config.fixed_support_count} fixed supports, "
                f"got {len(fixed_supports)}"
            )
        for index, support in enumerate(fixed_supports):
            if tuple(support.shape) != (self.num_nodes, self.num_nodes):
                raise ValueError(
                    f"support {index} must have shape ({self.num_nodes}, {self.num_nodes})"
                )
            if support.device != inputs.device or support.dtype != inputs.dtype:
                raise ValueError(
                    "inputs and supports must use the same device and dtype"
                )

    def freeze_for_online_adaptation(self) -> None:
        """全体を凍結し、Location-Specific Learnerだけを学習可能にする。"""

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.location_specific.parameters():
            parameter.requires_grad_(True)
        # evalでも勾配計算は可能。BN統計とDropoutをwarm-up時の状態に固定する。
        self.eval()

    def unfreeze_for_warmup(self) -> None:
        """warm-up用に全パラメータを学習可能へ戻す。"""

        for parameter in self.parameters():
            parameter.requires_grad_(True)
        self.train()

    def online_parameters(self) -> Iterator[nn.Parameter]:
        return iter(self.location_specific.parameters())

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def online_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.location_specific.parameters())
