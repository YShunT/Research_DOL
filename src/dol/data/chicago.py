"""Chicago-TをDOL用の3分割データセットへ準備する。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import DataConfig
from .scaler import GlobalStandardScaler
from .windows import SplitBoundaries, TimeWindowDataset, compute_split_boundaries


@dataclass(frozen=True) #インスタンスを読み取り専用へ / __init__を自動生成
class ChicagoDataBundle:
    """warm-up、validation、onlineのデータと標準化情報。"""

    train: TimeWindowDataset
    validation: TimeWindowDataset
    online: TimeWindowDataset
    scaler: GlobalStandardScaler
    boundaries: SplitBoundaries
    num_nodes: int
    total_steps: int


def load_chicago_demand(path: Path, expected_num_nodes: int = 77) -> np.ndarray: 
    """配布NPZから乗車需要だけを ``float64`` で取り出す。

    配布ファイルはTimestampを含むobject配列なので ``allow_pickle=True`` が
    必要である。信頼できる配布ファイル以外には使用しない。
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive: # allow_pickle; データ内にTimestampが混ざって保存されているため
        if "data" not in archive:
            raise KeyError(f"{path} does not contain key 'data'")
        raw = archive["data"]

    if raw.ndim != 3 or raw.shape[1] != expected_num_nodes:
        raise ValueError(
            "Chicago-T data must have shape (time, nodes, columns) with "
            f"{expected_num_nodes} nodes, got {raw.shape}"
        )

    demand = np.asarray(raw[..., -1], dtype=float) #float64に変換
    if not np.isfinite(demand).all():
        raise ValueError("demand contains non-finite values")
    if (demand < 0).any():
        raise ValueError("demand must be non-negative")
    return demand #R^(T,N)


def prepare_chicago_data(config: DataConfig) -> ChicagoDataBundle:
    """Chicago-Tを読み、訓練統計で標準化して3データセットを返す。"""

    demand = load_chicago_demand(config.data_path, config.num_nodes)
    boundaries = compute_split_boundaries(demand.shape[0], config)

    train_start, train_end = boundaries.train
    scaler = GlobalStandardScaler().fit(demand[train_start:train_end])
    normalized = scaler.transform(demand)
    series = torch.from_numpy(normalized)

    dataset_arguments = {
        "series": series,
        "input_length": config.input_length,
        "forecast_horizon": config.forecast_horizon,
    }
    return ChicagoDataBundle(
        train=TimeWindowDataset(
            target_interval=boundaries.train,
            **dataset_arguments,
        ),
        validation=TimeWindowDataset(
            target_interval=boundaries.validation,
            **dataset_arguments,
        ),
        online=TimeWindowDataset(
            target_interval=boundaries.online,
            **dataset_arguments,
        ),
        scaler=scaler,
        boundaries=boundaries,
        num_nodes=demand.shape[1],
        total_steps=demand.shape[0],
    )
