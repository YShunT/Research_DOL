"""保存したJSONだけからexp03のPNGを再生成する。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .storage import read_json

LABELS = {"mae": "MAE", "wmape": "WMAPE (%)", "sample_rmse": "Sample RMSE",
          "rmse": "Global RMSE"}
COLORS = {"down": "#0072B2", "up": "#D55E00", "both": "#009E73"}


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_path = path.parent.parent / "config.json"
    if config_path.exists():
        config = read_json(config_path)
        present = len(list(path.parent.parent.glob("seeds/seed*/static/as_stored/spectra.json")))
        fig.text(.01, .005, f"exp03 | {config['scope']} | available seeds={present}",
                 fontsize=8, color="gray")
    fig.tight_layout(rect=(0, .025, 1, .88))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def legend(fig, axes) -> None:
    unique = {}
    for ax in np.asarray(axes).reshape(-1):
        handles, labels = ax.get_legend_handles_labels()
        unique.update(zip(labels, handles))
    if unique:
        fig.legend(unique.values(), unique.keys(), loc="upper center",
                   bbox_to_anchor=(.5, .98), ncol=min(4, len(unique)), fontsize=8)


def spectra_plots(root: Path, figures: Path) -> None:
    for name, field, ylabel in (
        ("singular_spectrum_by_layer.png", "singular_values", "Singular value"),
        ("rank_vs_energy.png", "cumulative_energy", "Cumulative energy"),
        ("rank_vs_weight_error.png", "relative_frobenius", "Relative Frobenius error"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        found = False
        for representation, linestyle in (("as_stored", "-"), ("aligned", "--")):
            paths = sorted(root.glob(f"seeds/seed*/static/{representation}/spectra.json"))
            for layer, ax in zip(("down", "up"), axes):
                values = [read_json(p)["layers"][layer]["centered"][field] for p in paths]
                values = [v for v in values if v is not None]
                if not values:
                    continue
                found = True
                y = np.asarray(values)
                x = np.arange(y.shape[1]) + int(field == "singular_values")
                ax.plot(x, y.mean(0), linestyle, label=f"{representation}, n={len(values)}")
                ax.fill_between(x, y.min(0), y.max(0), alpha=.12)
                if field == "relative_frobenius":
                    for source in paths:
                        summary = read_json(source)["layers"][layer]["centered"]
                        norm = summary["centered_norm"]
                        if norm > 1e-12:
                            ax.scatter(
                                [check["rank"] for check in summary["reconstruction_checks"]],
                                [check["actual"] / norm for check in summary["reconstruction_checks"]],
                                s=10, marker="x", alpha=.4,
                            )
                ax.set(xlabel="Component" if field == "singular_values" else "Rank",
                       ylabel=ylabel, title=layer)
                if field == "singular_values":
                    ax.set_yscale("symlog", linthresh=1e-10)
                if field == "cumulative_energy":
                    for threshold in (.9, .95, .99):
                        ax.axhline(threshold, color="gray", lw=.5, alpha=.5)
                ax.grid(alpha=.2)
        if found:
            legend(fig, axes)
            save(fig, figures / name)
        else:
            plt.close(fig)


def metric_plots(groups: list[dict], figures: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for metric, ax in zip(LABELS, axes.flat):
        scale = 100 if metric == "wmape" else 1
        for representation, style in (("as_stored", "-"), ("aligned", "--")):
            for mode, color in COLORS.items():
                selected = [g for g in groups if g["split"] == "online_frozen"
                            and g["representation"] == representation and g["name"] == mode]
                selected.sort(key=lambda g: g["down_rank"] if mode != "up" else g["up_rank"])
                if not selected:
                    continue
                ranks = [g["down_rank"] if mode != "up" else g["up_rank"] for g in selected]
                means = [scale * g["summary"][metric]["mean"] for g in selected]
                errors = [scale * (g["summary"][metric]["std"] or 0) for g in selected]
                ax.errorbar(ranks, means, yerr=errors, fmt=style + "o", markersize=3,
                            color=color, alpha=.8, label=f"{representation}/{mode}")
            original = next((g for g in groups if g["name"] == "original"
                             and g["representation"] == representation
                             and g["split"] == "online_frozen"), None)
            if original and representation == "as_stored":
                ax.axhline(scale * original["summary"][metric]["mean"], color="black",
                           ls=":", label="Original frozen")
                bypass = next((g for g in groups if g["name"] == "bypass"
                               and g["representation"] == representation
                               and g["split"] == "online_frozen"), None)
                if bypass:
                    ax.axhline(scale * bypass["summary"][metric]["mean"], color="gray",
                               ls="-.", label="LSL bypass")
        ax.set(xlabel="Rank (both: down=up)", ylabel=LABELS[metric])
        ax.grid(alpha=.2)
    legend(fig, axes)
    save(fig, figures / "rank_vs_metrics.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for metric, ax in zip(LABELS, axes.flat):
        selected = [g for g in groups if g["split"] == "validation"
                    and g["representation"] == "as_stored" and g["name"] == "both"]
        if not selected:
            continue
        ranks = sorted({g["down_rank"] for g in selected} | {g["up_rank"] for g in selected})
        matrix = np.full((len(ranks), len(ranks)), np.nan)
        for g in selected:
            matrix[ranks.index(g["up_rank"]), ranks.index(g["down_rank"])] = (
                g["summary"][metric]["mean"] * (100 if metric == "wmape" else 1)
            )
        artist = ax.imshow(matrix, origin="lower", aspect="auto")
        ax.set(xticks=range(len(ranks)), xticklabels=ranks,
               yticks=range(len(ranks)), yticklabels=ranks,
               xlabel="Down rank", ylabel="Up rank", title=f"Validation {LABELS[metric]}")
        fig.colorbar(artist, ax=ax)
    save(fig, figures / "rank_grid_metrics.png")


def function_and_cost_plots(groups: list[dict], figures: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for representation, style in (("as_stored", "-"), ("aligned", "--")):
        for mode, color in COLORS.items():
            selected = [g for g in groups if g["representation"] == representation
                        and g["split"] == "online_frozen" and g["name"] == mode]
            selected.sort(key=lambda g: g["up_rank"] if mode == "up" else g["down_rank"])
            if not selected:
                continue
            ranks = [g["up_rank"] if mode == "up" else g["down_rank"] for g in selected]
            for field, ax in zip(("function_relative_rmse", "hidden_rmse",
                                  "activation_mismatch_rate"), axes):
                y = []
                for group in selected:
                    values = [case["function"][field] for case in group["seeds"].values()
                              if case["function"][field] is not None]
                    y.append(np.mean(values) if values else np.nan)
                ax.plot(ranks, y, style + "o", color=color, markersize=3,
                        label=f"{representation}/{mode}")
                ax.set(xlabel="Rank", ylabel=field)
    legend(fig, axes)
    save(fig, figures / "rank_vs_function_error.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    for mode, color in COLORS.items():
        selected = [g for g in groups if g["representation"] == "as_stored"
                    and g["split"] == "online_frozen" and g["name"] == mode]
        selected.sort(key=lambda g: g["up_rank"] if mode == "up" else g["down_rank"])
        if selected:
            ax.plot([g["up_rank"] if mode == "up" else g["down_rank"] for g in selected],
                    [g["factor_parameters"] for g in selected], "o-", color=color, label=mode)
            first = next(iter(selected[0]["seeds"].values()))
            ax.axhline(first["dense_parameters"], color="black", ls=":", label="Dense original")
    ax.set(xlabel="Rank", ylabel="Stored scalar count (factor representation)")
    legend(fig, [ax])
    save(fig, figures / "rank_vs_parameter_count.png")


def diagnostic_plots(root: Path, figures: Path) -> None:
    structure = sorted(root.glob("seeds/seed*/structure/diagnostics.json"))
    if structure:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for path in structure:
            data = read_json(path)
            seed = path.parents[1].name
            eigenvalues = data["input_covariance_eigenvalues"]
            axes[0].semilogy(range(1, len(eigenvalues) + 1),
                             np.maximum(eigenvalues, 1e-20), label=seed)
            activation = np.asarray(data["hidden_activation_rate"])
            axes[1].plot(range(1, activation.shape[1] + 1), activation.mean(0), "o-", label=seed)
        axes[0].set(xlabel="Input component", ylabel="Covariance eigenvalue")
        axes[1].set(xlabel="Hidden unit (within each seed)", ylabel="Mean activation rate")
        legend(fig, axes)
        save(fig, figures / "input_and_hidden_diagnostics.png")
    trajectories = sorted(root.glob("seeds/seed*/trajectory/spectra.json"))
    if not trajectories:
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fixed_fig, fixed_axes = plt.subplots(1, 2, figsize=(11, 4))
    for path in trajectories:
        records = list(read_json(path)["cases"].values())
        records.sort(key=lambda x: x["step"])
        seed = path.parents[1].name
        for layer, row, fixed_ax in zip(("down", "up"), axes, fixed_axes):
            for field, ax in zip(("state", "weekly_update", "cumulative_update"), row):
                values = [r["layers"][layer][field]["effective_rank"] for r in records]
                ax.plot([r["step"] for r in records], values, label=seed, alpha=.7)
                ax.set(xlabel="Completed step", ylabel="Energy entropy rank", title=f"{layer}/{field}")
            for rank in (4, 8, 16, 32, 76):
                usable = [r for r in records if str(rank) in r["layers"][layer]["fixed_basis_error"]
                          and r["layers"][layer]["fixed_basis_error"][str(rank)] is not None]
                if not usable:
                    continue
                x = [r["step"] for r in usable]
                fixed_ax.plot(x, [r["layers"][layer]["fixed_basis_error"][str(rank)] for r in usable],
                              label=f"{seed}/r{rank} fixed", alpha=.65)
                fixed_ax.plot(x, [r["layers"][layer]["weekly_update"]["relative_frobenius"][rank]
                                  for r in usable], "--", alpha=.65, label=f"{seed}/r{rank} SVD")
            fixed_ax.set(xlabel="Completed step", ylabel="Update relative error", title=layer)
    legend(fig, axes)
    save(fig, figures / "temporal_state_and_update_rank.png")
    # Multiple seeds/ranks can have many labels; place them to the right.
    handles, labels = fixed_axes[0].get_legend_handles_labels()
    if handles:
        fixed_axes[1].legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1),
                             fontsize=6)
    save(fixed_fig, figures / "fixed_basis_vs_dynamic_svd.png")


def plot_all(root: Path) -> None:
    root = Path(root)
    record = read_json(root / "metrics.json")
    figures = root / "figures"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    spectra_plots(root, figures)
    groups = list(record["groups"].values())
    if groups:
        metric_plots(groups, figures)
        function_and_cost_plots(groups, figures)
    diagnostic_plots(root, figures)
