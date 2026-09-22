"""DOLの設定。

CLIの文字列をそのまま各クラスへ渡さず、意味ごとのdataclassに分ける。
すべてPythonコードから明示的に生成するため、argparseは使用しない。
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Chicago-Tのデータと時系列分割に関する設定。"""

    data_path: Path = Path("datasets/chicago-t/chicago20_23.npz")
    adjacency_path: Path = Path("datasets/chicago-t/adj_chicago.npy")
    num_nodes: int = 77
    input_length: int = 12
    forecast_horizon: int = 12
    interval_minutes: int = 15
    steps_per_week: int = 672
    train_ratio: float = 0.20
    validation_ratio: float = 0.05
    online_ratio: float = 0.75

    def __post_init__(self) -> None:
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if self.input_length <= 0 or self.forecast_horizon <= 0:
            raise ValueError("input_length and forecast_horizon must be positive")

        ratio_sum = self.train_ratio + self.validation_ratio + self.online_ratio
        if abs(ratio_sum - 1.0) > 1e-12:
            raise ValueError(f"data split ratios must sum to 1.0, got {ratio_sum}")


@dataclass(frozen=True)
class ModelConfig:
    """DOLモデルの構造設定。"""

    residual_channels: int = 32
    dilation_channels: int = 32
    skip_channels: int = 256
    end_channels: int = 512
    lsl_bottleneck_channels: int = 4
    lsl_dropout: float = 0.5
    graph_dropout: float = 0.3
    adaptive_graph_rank: int = 10
    diffusion_order: int = 2
    fixed_support_count: int = 2
    use_adaptive_graph: bool = True
    blocks: int = 4
    layers_per_block: int = 2
    temporal_kernel_size: int = 2

    def __post_init__(self) -> None:
        channel_values = (
            self.residual_channels,
            self.dilation_channels,
            self.skip_channels,
            self.end_channels,
            self.lsl_bottleneck_channels,
        )
        if any(value <= 0 for value in channel_values):
            raise ValueError("all channel sizes must be positive")
        if self.blocks <= 0 or self.layers_per_block <= 0:
            raise ValueError("blocks and layers_per_block must be positive")
        if self.adaptive_graph_rank <= 0:
            raise ValueError("adaptive_graph_rank must be positive")
        if self.diffusion_order <= 0 or self.fixed_support_count <= 0:
            raise ValueError("diffusion_order and fixed_support_count must be positive")
        if self.temporal_kernel_size <= 1:
            raise ValueError("temporal_kernel_size must be greater than 1")
        if not 0.0 <= self.lsl_dropout < 1.0:
            raise ValueError("lsl_dropout must be in [0, 1)")
        if not 0.0 <= self.graph_dropout < 1.0:
            raise ValueError("graph_dropout must be in [0, 1)")

    @property
    def receptive_field(self) -> int:
        """Graph WaveNet部分が参照する時間ステップ数を返す。

        各block内でdilationを1, 2, 4, ...と増やし、blockが変わると1へ戻す。
        原DOLの既定値では13になる。
        """

        dilation_sum_per_block = sum(
            2**layer_index for layer_index in range(self.layers_per_block)
        )
        return 1 + (
            self.blocks
            * (self.temporal_kernel_size - 1)
            * dilation_sum_per_block
        )

    @property
    def support_count(self) -> int:
        """固定支持行列に学習型隣接行列を加えた総数。"""

        return self.fixed_support_count + int(self.use_adaptive_graph)


@dataclass(frozen=True)
class WarmupConfig:
    """全モデルを事前学習するwarm-upの設定。"""

    seed: int = 42
    batch_size: int = 32
    max_epochs: int = 150
    patience: int = 10
    learning_rate: float = 1e-3
    learning_rate_decay: float = 0.5
    weight_decay: float = 1e-2

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("batch_size, max_epochs, and patience must be positive")
        if (
            self.learning_rate <= 0.0
            or not 0.0 < self.learning_rate_decay <= 1.0
            or self.weight_decay < 0.0
        ):
            raise ValueError("invalid optimizer settings")


@dataclass(frozen=True)
class OnlineConfig:
    """warm-up後のストリーミング適応設定。

    ``learning_rate`` と ``weight_decay`` は現在のCLIでは記録用で、
    新規online optimizerは ``WarmupConfig`` の値で作る。
    ``batch_size`` も現行DataLoaderでは使わず、raw互換の1に固定する。
    """

    batch_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    awake_weeks: int = 1
    hibernate_weeks: int = 1
    buffer_capacity: int = 1_000
    replay_size: int = 8
    update_steps: int = 1

    def __post_init__(self) -> None:
        positive_values = (
            self.batch_size,
            self.awake_weeks,
            self.hibernate_weeks,
            self.buffer_capacity,
            self.replay_size,
            self.update_steps,
        )
        if any(value <= 0 for value in positive_values):
            raise ValueError("online integer settings must be positive")
        if self.replay_size > self.buffer_capacity:
            raise ValueError("replay_size cannot exceed buffer_capacity")


@dataclass(frozen=True)
class RuntimeConfig:
    """実行環境。CLIではなくコード上で明示的に変更する。"""

    device: str = "cuda:0"
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be 'cpu' or a CUDA device such as 'cuda:0'")


@dataclass(frozen=True)
class ArtifactConfig:
    """実験記録の保存方法。数値計算の条件には影響しない。"""

    experiments_root: Path = Path("experiments")
    save_predictions: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    """1実験に必要な設定をまとめる最上位オブジェクト。"""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    warmup: WarmupConfig = field(default_factory=WarmupConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)


DEFAULT_CONFIG = ExperimentConfig()
