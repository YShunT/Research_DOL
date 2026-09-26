"""exp02の集計結果と逐次診断からPNG図を生成する。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import DataConfig
from ..data.chicago import prepare_chicago_data

LABELS = {
    ("with_shared", "e_only"): "shared + E-only",
    ("with_shared", "frozen"): "shared + frozen",
    ("no_shared", "eb_update"): "no shared + EB-update",
    ("no_lsl", "frozen"): "no LSL",
}
COLORS = {
    ("with_shared", "e_only"): "tab:blue",
    ("with_shared", "frozen"): "tab:orange",
    ("no_shared", "eb_update"): "tab:green",
    ("no_lsl", "frozen"): "tab:gray",
}
RANKS = (0, 1, 2, 4, 8, 16, 32, 48, 64)
METRIC_LABELS = (
    ("mae", "MAE"),
    ("wmape", "WMAPE (%)"),
    ("sample_rmse", "Sample-wise RMSE"),
    ("rmse", "Global RMSE"),
)


def _groups(root: Path) -> list[dict[str, Any]]:
    path = root / "metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["groups"]


def _baseline() -> dict[str, float]:
    path = Path("experiments/exp01/metrics.json")
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: float(data["summary"][key]["mean"])
        for key in ("mae", "wmape", "sample_rmse", "rmse")
        if data.get("summary", {}).get(key, {}).get("mean") is not None
    }


def _save(fig: Any, root: Path, name: str) -> None:
    path = root / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _unique_legend(axes: Any) -> tuple[list[Any], list[str]]:
    """複数panelから重複しない凡例要素を集める。"""

    unique: dict[str, Any] = {}
    for axis in np.asarray(axes).flat:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
    return list(unique.values()), list(unique)


def plot_rank_metrics(root: Path, groups: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    baseline = _baseline()
    for axis, (key, label) in zip(axes.flat, METRIC_LABELS):
        scale = 100 if key == "wmape" else 1
        for condition in list(LABELS)[:3]:
            points = [
                group for group in groups
                if (group["structure"], group["branch"]) == condition
                and group["summary"][key]["n"]
            ]
            xs = [RANKS.index(group["rank"]) for group in points]
            ys = [group["summary"][key]["mean"] * scale for group in points]
            errors = [(group["summary"][key]["std"] or 0) * scale for group in points]
            axis.errorbar(
                xs, ys, yerr=errors, marker="o", capsize=2,
                color=COLORS[condition], label=LABELS[condition],
            )
            for group, x in zip(points, xs):
                values = [run["metrics"][key] * scale for run in group["runs"].values()]
                axis.scatter([x] * len(values), values, alpha=0.20, s=12, color=COLORS[condition])
        control = next(
            (g for g in groups if g["structure"] == "no_lsl" and g["summary"][key]["n"]),
            None,
        )
        if control:
            axis.scatter(
                [0], [control["summary"][key]["mean"] * scale],
                marker="x", s=90, color=COLORS[("no_lsl", "frozen")], label="no LSL (r=0)"
            )
        if key in baseline:
            value = baseline[key] * scale
            axis.axhline(value, color="black", linestyle="--", linewidth=1, label="exp01")
            axis.axhline(value * 1.01, color="black", linestyle=":", linewidth=0.8)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        axis.set_xticks(range(len(RANKS)), [str(rank) for rank in RANKS])
    for axis in axes[-1]:
        axis.set_xlabel("Basis rank r (discrete candidates)")
    handles, labels = _unique_legend(axes)
    fig.suptitle("exp02: online metrics by basis rank", y=0.99)
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=len(labels), fontsize=8, frameon=False,
    )
    fig.subplots_adjust(top=0.87)
    _save(fig, root, "rank_vs_metrics.png")


def _rff_mean(values: np.ndarray, frequencies: np.ndarray, phases: np.ndarray) -> np.ndarray:
    # values: (time, nodes); feature mean: (nodes, random_features)
    angles = values[:, :, None] * frequencies + phases
    return np.cos(angles).mean(axis=0) * math.sqrt(2 / len(frequencies))


def shift_reference(root: Path) -> tuple[np.ndarray, float]:
    """過去2週のみで地点別RFF-MMDを計算し、train由来の境界を固定する。"""

    path = root / "arrays" / "shift_reference.npz"
    if path.exists():
        with np.load(path) as archive:
            return archive["scores"], float(archive["threshold"])
    plan = json.loads((root / "config.json").read_text(encoding="utf-8"))
    data_config = DataConfig(**plan["base_config"]["data"])
    data = prepare_chicago_data(data_config)
    series = data.train.series.numpy().astype(np.float32, copy=False)
    period = data_config.steps_per_week
    rng = np.random.default_rng(2026)
    frequencies = rng.normal(size=16).astype(np.float32)
    phases = rng.uniform(0, 2 * np.pi, size=16).astype(np.float32)
    train_end = data.boundaries.train[1]
    train_features = [
        _rff_mean(series[start:start + period], frequencies, phases)
        for start in range(0, train_end - period + 1, period)
    ]
    train_scores = [
        np.linalg.norm(second - first, axis=1)
        for first, second in zip(train_features, train_features[1:])
    ]
    if not train_scores:
        raise ValueError("train period too short to calibrate shift threshold")
    threshold = float(np.quantile(np.stack(train_scores), 0.90))
    num_weeks = math.ceil(len(data.online) / period)
    scores = np.zeros((num_weeks, data.num_nodes), dtype=np.float32)
    for week in range(num_weeks):
        boundary = data.online.first_target_start + week * period
        before = series[boundary - 2 * period:boundary - period]
        recent = series[boundary - period:boundary]
        scores[week] = np.linalg.norm(
            _rff_mean(recent, frequencies, phases)
            - _rff_mean(before, frequencies, phases),
            axis=1,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, scores=scores, threshold=threshold,
        frequencies=frequencies, phases=phases,
        train_quantile=0.90, period=period,
    )
    return scores, threshold


def _weekly_metric(archive: Any, key: str, mask: np.ndarray) -> float:
    windows = archive["windows"]
    horizon = len(archive["horizon_absolute_sum"])
    n_nodes = archive["absolute_sum"].shape[1]
    count = np.broadcast_to(windows[:, None] * horizon, (len(windows), n_nodes))
    selected = mask[: len(windows)]
    denominator = count[selected].sum()
    if denominator == 0:
        return float("nan")
    if key == "mae":
        return float(archive["absolute_sum"][selected].sum() / denominator)
    return float(np.sqrt(archive["squared_sum"][selected].sum() / denominator))


def plot_adaptation_gain(root: Path, groups: list[dict[str, Any]]) -> None:
    lookup = {(g["structure"], g["rank"], g["branch"]): g for g in groups}
    scores, threshold = shift_reference(root)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for axis, key, label in zip(axes, ("mae", "rmse"), ("MAE gain", "Global RMSE gain")):
        for period_name, color in (("all", "tab:blue"), ("after shift", "tab:purple")):
            points = []
            for rank in RANKS[1:]:
                active = lookup[("with_shared", rank, "e_only")]["runs"]
                frozen = lookup[("with_shared", rank, "frozen")]["runs"]
                gains = []
                for seed in sorted(set(active) & set(frozen)):
                    if period_name == "all":
                        gains.append(frozen[seed]["metrics"][key] - active[seed]["metrics"][key])
                        continue
                    a_path = root / "variants/with_shared" / f"r{rank}" / "seeds" / f"seed{seed}" / "e_only/arrays/diagnostics.npz"
                    f_path = root / "variants/with_shared" / f"r{rank}" / "seeds" / f"seed{seed}" / "frozen/arrays/diagnostics.npz"
                    if not (a_path.exists() and f_path.exists()):
                        continue
                    with np.load(a_path) as a, np.load(f_path) as f:
                        mask = scores[:len(a["windows"])] > threshold
                        gain = _weekly_metric(f, key, mask) - _weekly_metric(a, key, mask)
                    if math.isfinite(gain):
                        gains.append(gain)
                if gains:
                    points.append((RANKS.index(rank), statistics_mean(gains), statistics_std(gains)))
            if points:
                axis.errorbar(
                    [p[0] for p in points], [p[1] for p in points],
                    yerr=[p[2] for p in points], marker="o", color=color,
                    capsize=2, label=period_name,
                )
        axis.axhline(0, color="black", linewidth=1)
        axis.set_xticks(range(len(RANKS)), [str(rank) for rank in RANKS])
        axis.set_xlabel("Basis rank r")
        axis.set_ylabel(label + " (frozen - E-only)")
        axis.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("Same-checkpoint benefit of updating location coefficients")
    _save(fig, root, "rank_vs_adaptation_gain.png")


def statistics_mean(values: list[float]) -> float:
    return float(np.mean(values))


def statistics_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def plot_cost(root: Path, groups: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    fields = (
        ("lsl_retained_parameters", "LSL retained parameters", 1),
        ("lsl_online_parameters", "Online trainable parameters", 1),
        ("inference_ms_per_window", "Inference ms/window", 1),
        ("update_ms_per_awake_step", "Update ms/step", 1),
        ("gpu_peak_memory_bytes", "GPU peak GiB", 1 / (1024**3)),
        ("elapsed_seconds", "Total online hours", 1 / 3600),
    )
    for axis, (field, title, scale) in zip(axes.flat, fields):
        for condition in list(LABELS)[:3]:
            points = []
            for group in groups:
                if (group["structure"], group["branch"]) != condition:
                    continue
                values = [
                    float(run[field]) * scale for run in group["runs"].values()
                    if run.get(field) is not None
                ]
                if values:
                    points.append((RANKS.index(group["rank"]), statistics_mean(values)))
            if points:
                axis.plot(
                    [p[0] for p in points], [p[1] for p in points],
                    marker="o", color=COLORS[condition], label=LABELS[condition],
                )
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.set_xticks(range(len(RANKS)), [str(rank) for rank in RANKS])
    for axis in axes[-1]:
        axis.set_xlabel("Basis rank r")
    axes[0, 0].legend(fontsize=8)
    _save(fig, root, "rank_vs_cost.png")


def plot_parameter_count(root: Path, groups: list[dict[str, Any]]) -> None:
    """rankごとの保持数とonline更新数を、他のコストから分離して示す。"""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    retained_series = (
        ("with_shared", "shared offset + EB", COLORS[("with_shared", "e_only")]),
        ("no_shared", "no shared offset + EB", COLORS[("no_shared", "eb_update")]),
    )
    for structure, label, color in retained_series:
        by_rank: dict[int, float] = {}
        for group in groups:
            if group["structure"] != structure or not group["runs"]:
                continue
            if structure == "with_shared" and not (
                group["branch"] == "e_only"
                or (group["rank"] == 0 and group["branch"] == "frozen")
            ):
                continue
            sample = next(iter(group["runs"].values()))
            by_rank[group["rank"]] = float(sample["lsl_retained_parameters"])
        points = [
            (RANKS.index(rank), by_rank[rank])
            for rank in RANKS if rank in by_rank
        ]
        if points:
            axes[0].plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o", color=color, label=label,
            )
    no_lsl = next(
        (group for group in groups if group["structure"] == "no_lsl" and group["runs"]),
        None,
    )
    if no_lsl:
        axes[0].scatter(
            [RANKS.index(0)], [0], marker="x", s=80,
            color=COLORS[("no_lsl", "frozen")], label="no LSL",
        )

    for condition in (
        ("with_shared", "e_only"),
        ("with_shared", "frozen"),
        ("no_shared", "eb_update"),
    ):
        points = []
        for group in groups:
            if (group["structure"], group["branch"]) != condition:
                continue
            if not group["runs"]:
                continue
            sample = next(iter(group["runs"].values()))
            points.append(
                (RANKS.index(group["rank"]), float(sample["lsl_online_parameters"]))
            )
        if points:
            axes[1].plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                color=COLORS[condition],
                label=LABELS[condition],
            )

    for axis, title in zip(
        axes,
        ("Retained LSL parameters", "Online-updated LSL parameters"),
    ):
        axis.axhline(
            22_484, color="black", linestyle="--", linewidth=1,
            label="exp01 independent LSL (22,484)",
        )
        axis.set_title(title)
        axis.set_xlabel("Basis rank r (discrete candidates)")
        axis.set_xticks(range(len(RANKS)), [str(rank) for rank in RANKS])
        axis.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Number of parameters")
    handles, labels = _unique_legend(axes)
    fig.suptitle("exp02: LSL parameter counts by basis rank", y=0.99)
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91),
        ncol=3, fontsize=8, frameon=False,
    )
    fig.subplots_adjust(top=0.77)
    _save(fig, root, "rank_vs_parameter_count.png")


def plot_accuracy_cost(root: Path, groups: list[dict[str, Any]]) -> None:
    baseline = _baseline()
    if "mae" not in baseline or "rmse" not in baseline:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, key, label in zip(axes, ("mae", "rmse"), ("Relative MAE change", "Relative RMSE change")):
        for group in groups:
            condition = (group["structure"], group["branch"])
            if condition not in COLORS or not group["summary"][key]["n"]:
                continue
            sample = next(iter(group["runs"].values()))
            x = 100 * (1 - sample["lsl_online_parameters"] / 22484)
            y = 100 * (group["summary"][key]["mean"] / baseline[key] - 1)
            axis.scatter(x, y, color=COLORS[condition])
            axis.annotate(str(group["rank"]), (x, y), fontsize=7)
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set_xlabel("Online parameter reduction vs exp01 (%)")
        axis.set_ylabel(label + " (%)")
        axis.grid(alpha=0.2)
    _save(fig, root, "accuracy_vs_update_cost.png")


def plot_coefficient_change(root: Path) -> None:
    path = root / "variants/with_shared/r8/seeds/seed42/e_only/arrays/diagnostics.npz"
    if not path.exists():
        return
    with np.load(path) as archive:
        coefficients = archive["E"]
        if coefficients.shape[2] == 0:
            return
        delta = np.linalg.norm(coefficients - coefficients[0], axis=2).T
    fig, axis = plt.subplots(figsize=(12, 6))
    image = axis.imshow(delta, aspect="auto", origin="lower", cmap="magma")
    axis.set_xlabel("Online snapshot (weekly)")
    axis.set_ylabel("Location index")
    axis.set_title("E coefficient change: shared r=8, seed 42")
    fig.colorbar(image, ax=axis, label="L2 change from warm-up")
    _save(fig, root, "location_coefficient_change_r8_seed42.png")


def plot_lsl_change(root: Path, groups: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    fields = (
        ("weight_change", "Generated LSL weight change"),
        ("correction_rms_change", "LSL correction RMS change"),
        ("basis_change", "Shared basis B change"),
    )
    for axis, (field, title) in zip(axes, fields):
        for condition in (("with_shared", "e_only"), ("no_shared", "eb_update")):
            points = []
            for group in groups:
                if (group["structure"], group["branch"]) != condition:
                    continue
                values = []
                for seed in group["runs"]:
                    path = (
                        root / "variants" / group["structure"] / f"r{group['rank']}"
                        / "seeds" / f"seed{seed}" / group["branch"]
                        / "arrays/diagnostics.npz"
                    )
                    if path.exists():
                        with np.load(path) as archive:
                            if field not in archive:
                                continue
                            final = archive[field][-1]
                            values.append(float(np.mean(final)))
                if values:
                    points.append((RANKS.index(group["rank"]), statistics_mean(values)))
            if points:
                axis.plot(
                    [p[0] for p in points], [p[1] for p in points],
                    marker="o", color=COLORS[condition], label=LABELS[condition],
                )
        axis.set_title(title)
        axis.set_xticks(range(len(RANKS)), [str(rank) for rank in RANKS])
        axis.set_xlabel("Basis rank r")
        axis.grid(alpha=0.2)
    if axes[0].get_legend_handles_labels()[0]:
        axes[0].legend(fontsize=8)
    _save(fig, root, "rank_vs_lsl_change.png")


def plot_distillation(root: Path) -> None:
    path = root / "diagnostics/distillation/metrics.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not records:
        return
    fields = (
        ("function_rmse", "LSL correction RMSE"),
        ("function_relative_rmse", "Relative correction RMSE"),
        ("validation_mae_delta", "Validation MAE difference"),
        ("validation_rmse_delta", "Validation global RMSE difference"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (field, label) in zip(axes.flat, fields):
        points = []
        for rank in RANKS:
            values = []
            for record in records:
                if record["rank"] != rank:
                    continue
                if field == "validation_mae_delta":
                    prediction = record["validation_prediction"]
                    value = prediction["student"]["mae"] - prediction["teacher"]["mae"]
                elif field == "validation_rmse_delta":
                    prediction = record["validation_prediction"]
                    value = prediction["student"]["rmse"] - prediction["teacher"]["rmse"]
                else:
                    value = record[field]
                values.append(float(value))
            if values:
                points.append((RANKS.index(rank), statistics_mean(values), statistics_std(values)))
        axis.errorbar(
            [p[0] for p in points], [p[1] for p in points],
            yerr=[p[2] for p in points], marker="o", capsize=2,
        )
        axis.set_ylabel(label)
        axis.set_xticks(range(len(RANKS)), [str(rank) for rank in RANKS])
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Basis rank r")
    fig.suptitle("Fixed-backbone approximation of exp01 LSL")
    _save(fig, root, "rank_vs_lsl_function_error.png")

    location = next(
        (record for record in records if record["rank"] == 8 and record["seed"] == 42),
        None,
    )
    if location:
        fig, axis = plt.subplots(figsize=(12, 4))
        axis.bar(range(len(location["location_function_rmse"])), location["location_function_rmse"])
        axis.set_xlabel("Location index")
        axis.set_ylabel("Correction RMSE")
        axis.set_title("Fixed-backbone LSL approximation, r=8, seed 42")
        _save(fig, root, "location_lsl_function_error_r8_seed42.png")


def plot_all(root: Path = Path("experiments/exp02")) -> None:
    root = Path(root)
    groups = _groups(root)
    if not any(group["runs"] for group in groups):
        raise RuntimeError("exp02 has no completed online results to plot")
    plot_rank_metrics(root, groups)
    plot_adaptation_gain(root, groups)
    plot_cost(root, groups)
    plot_parameter_count(root, groups)
    plot_accuracy_cost(root, groups)
    plot_coefficient_change(root)
    plot_lsl_change(root, groups)
    plot_distillation(root)
