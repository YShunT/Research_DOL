"""LSL重みの抽出・層別EYM近似・関数同値な隠れユニット整列。

数値解析はCPU float64で行い、モデルへ戻す時だけ元dtypeへ変換する。
このモジュールはデータ読込・実験の実行・結果保存を行わない。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import permutations

import numpy as np
import torch

from ..layers.LSL import LocationSpecificLearner


@dataclass(frozen=True)
class Weights:
    down: np.ndarray  # (nodes, hidden, channels)
    down_bias: np.ndarray
    up: np.ndarray    # (nodes, channels, hidden)
    up_bias: np.ndarray

    @classmethod
    def from_learner(cls, learner: LocationSpecificLearner) -> "Weights":
        def stack(layer: str, name: str) -> np.ndarray:
            values = [getattr(getattr(a, layer), name).detach().cpu().double().numpy()
                      for a in learner.adapters]
            return np.stack(values)
        return cls(
            stack("down_projection", "weight")[..., 0],
            stack("down_projection", "bias"),
            stack("up_projection", "weight")[..., 0],
            stack("up_projection", "bias"),
        )

    @torch.no_grad()
    def apply(self, learner: LocationSpecificLearner) -> None:
        """Parameterを置換せずcopyする。既存optimizerの参照を壊さない。"""
        if len(learner.adapters) != len(self.down):
            raise ValueError("node count differs")
        for n, adapter in enumerate(learner.adapters):
            for layer, weight, bias in (
                (adapter.down_projection, self.down[n], self.down_bias[n]),
                (adapter.up_projection, self.up[n], self.up_bias[n]),
            ):
                layer.weight.copy_(torch.as_tensor(weight[..., None],
                                                    device=layer.weight.device,
                                                    dtype=layer.weight.dtype))
                layer.bias.copy_(torch.as_tensor(bias, device=layer.bias.device,
                                                  dtype=layer.bias.dtype))

    def matrix(self, layer: str) -> np.ndarray:
        value = getattr(self, layer)
        return value.reshape(len(value), -1)

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in
                ("down", "down_bias", "up", "up_bias")}


@dataclass(frozen=True)
class Decomposition:
    mean: np.ndarray
    residual: np.ndarray
    u: np.ndarray
    s: np.ndarray
    vt: np.ndarray
    max_rank: int

    @classmethod
    def fit(cls, matrix: np.ndarray, centered: bool = True) -> "Decomposition":
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("SVD requires a finite matrix")
        mean = matrix.mean(axis=0) if centered else np.zeros(matrix.shape[1])
        residual = matrix - mean
        u, s, vt = np.linalg.svd(residual, full_matrices=False)
        limit = min(matrix.shape[0] - int(centered), matrix.shape[1])
        return cls(mean, residual, u, s, vt, limit)

    def reconstruct(self, rank: int) -> np.ndarray:
        if not 0 <= rank <= self.max_rank:
            raise ValueError(f"rank {rank} outside [0, {self.max_rank}]")
        return self.mean + (self.u[:, :rank] * self.s[:rank]) @ self.vt[:rank]

    def summary(self) -> dict:
        energy = self.s ** 2
        total = float(energy.sum())
        zero = total <= 1e-24
        fractions = energy / total if not zero else np.zeros_like(energy)
        ranks = np.arange(self.max_rank + 1)
        # Reverse sums avoid cancellation near the full-rank endpoint.
        tail = np.r_[np.cumsum(energy[::-1])[::-1], 0.][ranks]
        kept = np.maximum(total - tail, 0.)
        cumulative = kept / total if not zero else None
        record = {
            "singular_values": self.s.tolist(), "ranks": ranks.tolist(),
            "total_energy": total, "mean_norm": float(np.linalg.norm(self.mean)),
            "centered_norm": float(np.sqrt(total)), "zero_residual": zero,
            "cumulative_energy": cumulative.tolist() if cumulative is not None else None,
            "tail_frobenius": np.sqrt(tail).tolist(),
            "relative_frobenius": np.sqrt(tail / total).tolist() if not zero else [0.] * len(ranks),
            "stable_rank": float(total / energy[0]) if not zero else 0.,
            "effective_rank": float(np.exp(-np.sum(
                fractions[fractions > 0] * np.log(fractions[fractions > 0])
            ))) if not zero else 0.,
        }
        for percent in (90, 95, 99):
            record[f"r{percent}"] = (
                min(int(np.searchsorted(cumulative, percent / 100)), self.max_rank)
                if not zero else 0
            )
        # EYM identity is verified on the matrix, not just on the singular values.
        checks = []
        for rank in sorted(set((0, min(4, self.max_rank), self.max_rank))):
            error = np.linalg.norm(self.residual - (self.reconstruct(rank) - self.mean))
            expected = np.linalg.norm(self.s[rank:])
            if not np.isclose(error, expected, atol=1e-10, rtol=1e-8):
                raise ArithmeticError("EYM reconstruction check failed")
            checks.append({"rank": rank, "actual": float(error), "expected": float(expected)})
        record["reconstruction_checks"] = checks
        return record


def approximate(weights: Weights, down_rank: int | None, up_rank: int | None,
                decompositions: dict[str, Decomposition],
                projection_seed: int | None = None) -> Weights:
    changes = {}
    for index, (layer, rank) in enumerate((("down", down_rank), ("up", up_rank))):
        if rank is None:
            continue
        decomposition = decompositions[layer]
        if not 0 <= rank <= decomposition.max_rank:
            raise ValueError("invalid rank")
        if projection_seed is None:
            matrix = decomposition.reconstruct(rank)
        else:
            # Local RNG keeps model/SMB random streams untouched. Nested across ranks.
            rng = np.random.default_rng(np.random.SeedSequence([projection_seed, index]))
            q, _ = np.linalg.qr(rng.standard_normal(
                (decomposition.residual.shape[1], decomposition.max_rank)
            ))
            basis = q[:, :rank]
            matrix = decomposition.mean + decomposition.residual @ basis @ basis.T
        changes[layer] = matrix.reshape(getattr(weights, layer).shape)
    return replace(weights, **changes)


def align_hidden(weights: Weights) -> tuple[Weights, dict]:
    """Positive ReLU scaling and joint down/bias/up permutations around a medoid."""
    down, bias, up = weights.down.copy(), weights.down_bias.copy(), weights.up.copy()
    scales = np.sqrt((down ** 2).sum(axis=2) + bias ** 2)
    tiny = scales <= 1e-12
    scales[tiny] = 1.
    down /= scales[..., None]
    bias /= scales
    up *= scales[:, None, :]
    incoming = np.concatenate((down, bias[..., None]), axis=2)
    outgoing = up.transpose(0, 2, 1)
    permutations_ = np.array(list(permutations(range(down.shape[1]))))
    denominators = [float(np.mean(np.sum(x ** 2, axis=(1, 2))))
                    for x in (incoming, outgoing)]
    nodes = len(down)
    costs = np.zeros((nodes, nodes))
    choices = np.zeros((nodes, nodes), dtype=int)
    for reference in range(nodes):
        distances = np.zeros((nodes, len(permutations_)))
        for block, denominator in zip((incoming, outgoing), denominators):
            if denominator > 0:
                differences = block[:, permutations_, :] - block[reference][None, None, :, :]
                distances += np.sum(differences ** 2, axis=(2, 3)) / denominator
        choices[reference] = distances.argmin(axis=1)
        costs[reference] = distances.min(axis=1)
    reference = int(costs.sum(axis=1).argmin())
    chosen = permutations_[choices[reference]]
    for n, order in enumerate(chosen):
        down[n] = down[n, order]
        bias[n] = bias[n, order]
        up[n] = up[n][:, order]
    return Weights(down, bias, up, weights.up_bias.copy()), {
        "reference_node": reference, "permutations": chosen.tolist(),
        "scales": scales.tolist(), "unscaled_tiny_units": int(tiny.sum()),
    }


def parameter_count(weights: Weights, down_rank: int | None,
                    up_rank: int | None) -> int:
    n = len(weights.down)
    count = weights.down_bias.size + weights.up_bias.size
    for layer, rank in (("down", down_rank), ("up", up_rank)):
        size = getattr(weights, layer)[0].size
        count += n * size if rank is None else size + (n + size) * rank
    return count


def fixed_basis_error(matrix: np.ndarray, reference: Decomposition, rank: int) -> float | None:
    centered = matrix - matrix.mean(axis=0)
    norm = np.linalg.norm(centered)
    if norm <= 1e-12:
        return None
    basis = reference.vt[:rank]
    return float(np.linalg.norm(centered - centered @ basis.T @ basis) / norm)
