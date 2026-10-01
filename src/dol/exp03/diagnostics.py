"""同じ入力でLSLの各段階を比較する、学習を伴わない診断。"""
from __future__ import annotations

import copy

import numpy as np
import torch

from .svd import Weights


def sample_indices(length: int, count: int) -> list[int]:
    return np.unique(np.linspace(0, length - 1, min(count, length), dtype=int)).tolist()


@torch.no_grad()
def collect_features(model, dataset, count: int, batch_size: int = 32):
    indices = sample_indices(len(dataset), count)
    inputs = torch.stack([dataset[i][0].float() for i in indices])
    features = []
    device = next(model.parameters()).device
    model.eval()
    for batch in inputs.split(batch_size):
        x = batch.to(device).unsqueeze(1).transpose(2, 3)
        padding = max(0, model.receptive_field - x.shape[-1])
        x = torch.nn.functional.pad(x, (padding, 0, 0, 0))
        features.append(model.input_projection(x).cpu())
    return indices, torch.cat(features)


@torch.no_grad()
def stages(learner, features):
    """実際のConv1dを使い、(batch, channels, nodes, time)で返す。"""
    pre, hidden, output = [], [], []
    for n, adapter in enumerate(learner.adapters):
        a = adapter.down_projection(features[:, :, n, :])
        z = adapter.activation(a)  # eval modeではdropoutは恒等変換
        pre.append(a)
        hidden.append(z)
        output.append(adapter.up_projection(z))
    return tuple(torch.stack(x, dim=2) for x in (pre, hidden, output))


def relative(error_squared: float, baseline_squared: float) -> float | None:
    return float(np.sqrt(error_squared / baseline_squared)) if baseline_squared > 1e-24 else None


@torch.no_grad()
def compare_functions(learner, reference: Weights, candidate: Weights,
                      features: torch.Tensor, batch_size: int = 32) -> dict:
    """小さいCPU cloneで比較し、元modelとoptimizerには触れない。"""
    original = copy.deepcopy(learner).cpu().eval()
    modified = copy.deepcopy(original)
    reference.apply(original)
    candidate.apply(modified)
    sums = np.zeros(3)
    counts = np.zeros(3, dtype=np.int64)
    baseline_squared = residual_squared = input_squared = 0.
    node_squared = np.zeros(len(reference.down))
    node_baseline = np.zeros(len(reference.down))
    activation_mismatch = near_zero = activation_count = 0
    max_error = 0.
    for h in features.split(batch_size):
        before = stages(original, h)
        after = stages(modified, h)
        for i, (left, right) in enumerate(zip(before, after)):
            delta = (right.double() - left.double()).square()
            sums[i] += delta.sum().item()
            counts[i] += delta.numel()
        correction = before[2].double()
        difference = after[2].double() - correction
        node_squared += difference.square().sum(dim=(0, 1, 3)).numpy()
        node_baseline += correction.square().sum(dim=(0, 1, 3)).numpy()
        baseline_squared += correction.square().sum().item()
        residual_squared += (h.double() + correction).square().sum().item()
        input_squared += h.double().square().sum().item()
        activation_mismatch += ((before[0] > 0) != (after[0] > 0)).sum().item()
        near_zero += (before[0].abs() <= 1e-6).sum().item()
        activation_count += before[0].numel()
        max_error = max(max_error, difference.abs().max().item())
    node_rmse = np.sqrt(node_squared / (counts[2] / len(node_squared)))
    return {
        "preactivation_rmse": float(np.sqrt(sums[0] / counts[0])),
        "hidden_rmse": float(np.sqrt(sums[1] / counts[1])),
        "function_rmse": float(np.sqrt(sums[2] / counts[2])),
        "function_relative_rmse": relative(sums[2], baseline_squared),
        "residual_relative_rmse": relative(sums[2], residual_squared),
        "correction_to_input_rms": relative(baseline_squared, input_squared),
        "correction_rms": float(np.sqrt(baseline_squared / counts[2])),
        "max_absolute_error": max_error,
        "activation_mismatch_rate": activation_mismatch / activation_count,
        "near_zero_rate": near_zero / activation_count,
        "node_function_rmse": node_rmse.tolist(),
        "node_function_relative_rmse": [relative(a, b) for a, b in zip(node_squared, node_baseline)],
        "node_rmse_summary": dict(zip(("median", "p90", "max"),
                                     np.quantile(node_rmse, (.5, .9, 1)).tolist())),
        "num_windows": len(features),
    }


@torch.no_grad()
def verify_alignment(learner, original: Weights, aligned: Weights, features) -> dict:
    left = copy.deepcopy(learner).cpu().eval()
    right = copy.deepcopy(left)
    original.apply(left)
    aligned.apply(right)
    generator = torch.Generator().manual_seed(301)
    random_features = torch.randn((8, features.shape[1], features.shape[2], features.shape[3]),
                                  generator=generator)
    max_absolute = max_relative = 0.
    for probe in (features, random_features):
        for batch in probe.split(32):
            a, b = left(batch), right(batch)
            if not torch.allclose(a, b, atol=1e-6, rtol=1e-5):
                raise ArithmeticError("hidden-unit alignment changed the LSL function")
            max_absolute = max(max_absolute, (a - b).abs().max().item())
            max_relative = max(max_relative, ((a - b).abs() / a.abs().clamp_min(1e-6)).max().item())
    return {"passed": True, "atol": 1e-6, "rtol": 1e-5,
            "max_absolute_error": max_absolute, "max_relative_error": max_relative}


@torch.no_grad()
def structure_diagnostics(model, weights: Weights, features) -> dict:
    vectors = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).double()
    centered = vectors - vectors.mean(0)
    covariance = centered.T @ centered / len(centered)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0).numpy()
    a = model.input_projection.weight.detach().cpu().double().numpy().reshape(-1)
    c = model.input_projection.bias.detach().cpu().double().numpy()
    projection = np.column_stack((a, c))
    local = {}
    for layer in ("down", "up"):
        s = np.linalg.svd(getattr(weights, layer), compute_uv=False)
        local[layer] = {"singular_values": s.tolist(),
                        "tail_frobenius": [np.linalg.norm(s[:, q:], axis=1).tolist()
                                           for q in range(1, s.shape[1] + 1)]}
    learner = copy.deepcopy(model.location_specific).cpu().eval()
    weights.apply(learner)
    active = np.zeros(weights.down_bias.shape)
    z_squared = np.zeros_like(active)
    count = 0
    max_composition_error = 0.
    composition_verified = True
    # h lies in span(a,c). A pseudoinverse recovers a representative scalar x.
    for h in features.split(32):
        pre, z, _ = stages(learner, h)
        active += (z > 0).sum(dim=(0, 3)).T.numpy()
        z_squared += z.double().square().sum(dim=(0, 3)).T.numpy()
        count += h.shape[0] * h.shape[-1]
        if float(np.dot(a, a)) > 1e-24:
            hh = h.permute(0, 2, 3, 1).double().numpy()
            x = (hh - c) @ a / np.dot(a, a)
            slopes = weights.down @ a
            intercepts = weights.down @ c + weights.down_bias
            composed = x[..., None] * slopes[None, :, None, :] + intercepts[None, :, None, :]
            direct = pre.permute(0, 2, 3, 1).double().numpy()
            max_composition_error = max(
                max_composition_error, float(np.max(np.abs(composed - direct)))
            )
            composition_verified &= bool(np.allclose(composed, direct, atol=1e-5, rtol=1e-5))
        else:
            direct = pre.permute(0, 2, 3, 1).double().numpy()
            composed = weights.down @ c + weights.down_bias
            max_composition_error = max(
                max_composition_error,
                float(np.max(np.abs(direct - composed[None, :, None, :]))),
            )
            composition_verified &= bool(np.allclose(
                direct, composed[None, :, None, :], atol=1e-5, rtol=1e-5
            ))
    if not composition_verified:
        raise ArithmeticError("input projection composition check failed")
    return {
        "input_covariance_eigenvalues": eigenvalues.tolist(),
        "input_affine_span_singular_values": np.linalg.svd(projection, compute_uv=False).tolist(),
        "input_composition_max_absolute_error": max_composition_error,
        "input_composition_verified": composition_verified,
        "local_svd": local, "hidden_activation_rate": (active / count).tolist(),
        "hidden_output_contribution_rms": (
            np.sqrt(z_squared / count) * np.linalg.norm(weights.up, axis=1)
        ).tolist(),
    }


def effective_down_error(model, original: Weights, candidate: Weights) -> dict:
    a = model.input_projection.weight.detach().cpu().double().numpy().reshape(-1)
    c = model.input_projection.bias.detach().cpu().double().numpy()
    slopes = (candidate.down - original.down) @ a
    intercepts = ((candidate.down - original.down) @ c
                  + candidate.down_bias - original.down_bias)
    return {"slope_rmse": float(np.sqrt(np.mean(slopes ** 2))),
            "intercept_rmse": float(np.sqrt(np.mean(intercepts ** 2)))}
