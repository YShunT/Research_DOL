"""DOLが使用する固定グラフの支持行列を生成する。

Chicago-Tの隣接行列 ``A`` に自己ループを加え、Graph WaveNetと同じ
double-transition形式を作る。

``P_forward = D(A + I)^-1 (A + I)``
``P_backward = D((A + I)^T)^-1 (A + I)^T``

返す支持行列は公開実装と同じく転置せずNodePropagationへ渡し、
特徴のノード軸へ右から ``X @ P`` として掛ける。

原実装の ``utils/graph_algo.py`` はNumPy、SciPy、ファイル読込みを一つの
モジュールで扱う。この実装では、読込みとTensor演算を小さな関数へ分ける。
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


def load_adjacency(
    path: Path,
    expected_num_nodes: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``.npy``の隣接行列を指定したdevice・dtypeのTensorとして読む。

    ``device=None``の場合はCPUへ読み込む。モデルをGPUへ配置する場合は、
    ``device="cuda:0"``のように明示して支持行列も同じdeviceへ置く。
    """

    path = Path(path)
    if path.suffix != ".npy":
        raise ValueError(f"adjacency file must be .npy, got {path.suffix!r}")
    if not path.is_file():
        raise FileNotFoundError(f"adjacency file not found: {path}")

    adjacency_array = np.load(path, allow_pickle=False)
    adjacency = torch.from_numpy(adjacency_array).to(device=device, dtype=dtype)
    _validate_adjacency(adjacency)

    if expected_num_nodes is not None and adjacency.shape[0] != expected_num_nodes:
        raise ValueError(
            f"expected {expected_num_nodes} nodes, got {adjacency.shape[0]}"
        )
    return adjacency


def add_self_loops(adjacency: Tensor) -> Tensor:
    """隣接行列へ単位行列を加える。

    原DOLのChicago-T経路と同じく加算であり、対角成分を単に1へ置換する
    操作ではない。入力Tensorは変更しない。
    """

    _validate_adjacency(adjacency)
    identity = torch.eye(
        adjacency.shape[0],
        dtype=adjacency.dtype,
        device=adjacency.device,
    )
    return adjacency + identity


def row_normalize(adjacency: Tensor, epsilon: float = 1e-6) -> Tensor:
    """行次数で正規化してrandom-walk遷移行列を作る。

    ``epsilon`` の加え方は原実装の ``asym_adj`` に合わせる。孤立行は0のまま。
    """

    _validate_adjacency(adjacency)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    degree = adjacency.sum(dim=1)
    inverse_degree = torch.where(
        degree == 0,
        torch.zeros_like(degree),
        torch.reciprocal(degree + epsilon),
    )
    return inverse_degree[:, None] * adjacency


def build_double_transition_supports(
    adjacency: Tensor,
    add_identity: bool = True,
    epsilon: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """順方向と逆方向の2つの支持行列を返す。

    Args:
        adjacency: ``(nodes, nodes)`` の固定隣接行列。
        add_identity: 正規化前に自己ループを加えるか。
        epsilon: 次数の逆数を計算するときの安定化定数。
    """

    _validate_adjacency(adjacency)
    graph = add_self_loops(adjacency) if add_identity else adjacency
    forward = row_normalize(graph, epsilon=epsilon)
    backward = row_normalize(graph.transpose(0, 1), epsilon=epsilon)
    return forward.contiguous(), backward.contiguous()


def _validate_adjacency(adjacency: Tensor) -> None:
    if not isinstance(adjacency, Tensor):
        raise TypeError("adjacency must be a torch.Tensor")
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            "adjacency must have shape (nodes, nodes), "
            f"got {tuple(adjacency.shape)}"
        )
    if not torch.is_floating_point(adjacency):
        raise TypeError("adjacency must use a floating-point dtype")
    if not torch.isfinite(adjacency).all():
        raise ValueError("adjacency contains NaN or infinity")
    if torch.any(adjacency < 0):
        raise ValueError("adjacency cannot contain negative weights")
