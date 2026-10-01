"""exp03 A/B/Cの実行管理。条件同一性を検査して完了した評価を再利用する。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import gc
import logging
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.utils.data import Subset

from ..config import (ArtifactConfig, DataConfig, ExperimentConfig, ModelConfig,
                      OnlineConfig, RuntimeConfig, WarmupConfig)
from ..pipeline import build_components, load_warmup_checkpoint
from .diagnostics import collect_features, structure_diagnostics, verify_alignment
from .evaluation import run_interventions
from .storage import logger, read_json, sha256, source_identity, suite_lock, write_json
from .svd import Decomposition, Weights, align_hidden
from .trajectory import analyze_trajectory, capture_trajectory

SEEDS = (42, 43, 44, 45, 46)
RANKS = (0, 1, 2, 4, 8, 16, 32, 48, 64, 76)
METRICS = ("mae", "wmape", "sample_rmse", "rmse")


@dataclass(frozen=True)
class Settings:
    device: str = "cuda:0"
    batch_size: int = 32
    diagnostic_count: int = 1024
    smoke: bool = False
    threads: int = 1

    def __post_init__(self):
        if min(self.batch_size, self.diagnostic_count, self.threads) <= 0:
            raise ValueError("batch size, diagnostic count and threads must be positive")

    @property
    def ranks(self):
        return (0, 4, 76) if self.smoke else RANKS

    @property
    def limit(self):
        return 8 if self.smoke else None

    @property
    def samples(self):
        return min(self.diagnostic_count, 8) if self.smoke else self.diagnostic_count


def seed_config(seed: int, device: str) -> ExperimentConfig:
    source = read_json(Path(f"experiments/exp01/seeds/seed{seed}/config.json"))
    data = dict(source["data"])
    for key in ("data_path", "adjacency_path"):
        data[key] = Path(data[key])
    artifacts = dict(source["artifacts"])
    artifacts["experiments_root"] = Path(artifacts["experiments_root"])
    runtime = dict(source["runtime"], device=device)
    return ExperimentConfig(DataConfig(**data), ModelConfig(**source["model"]),
                            WarmupConfig(**source["warmup"]), OnlineConfig(**source["online"]),
                            RuntimeConfig(**runtime), ArtifactConfig(**artifacts))


def checkpoint(seed: int) -> Path:
    return Path(f"experiments/exp01/seeds/seed{seed}/checkpoints/warmup.pt")


def prepare_suite(root: Path, settings: Settings) -> dict:
    identity = source_identity()
    if identity["dirty"] and not settings.smoke:
        raise RuntimeError("Commit src/ and exp03 strategy before a full run; use --smoke for checks.")
    configuration = {
        "schema": 1, "settings": asdict(settings), "seeds": list(SEEDS),
        "ranks": list(settings.ranks), "representations": ["as_stored", "aligned"],
        "stages": ["spectra", "static", "trajectory"],
        "function_tolerance": .05, "prediction_tolerance": .01,
        "source": identity, "scope": "smoke" if settings.smoke else "full",
    }
    manifest = {}
    for seed in SEEDS:
        path = checkpoint(seed)
        digest = sha256(path)
        expected = read_json(path.parent.parent / "metrics.json")["artifacts"]["checkpoint_sha256"]
        if digest != expected:
            raise RuntimeError(f"exp01 checkpoint hash mismatch: {path}")
        manifest[str(seed)] = {
            "checkpoint": str(path), "checkpoint_sha256": digest,
            "config_sha256": sha256(path.parent.parent / "config.json"),
            "metrics_sha256": sha256(path.parent.parent / "metrics.json"),
        }
    base = seed_config(SEEDS[0], settings.device)
    manifest["data"] = {str(p): sha256(p) for p in
                        (base.data.data_path, base.data.adjacency_path)}
    expected_data = read_json(Path("experiments/exp01/config.json"))["data_sha256"]
    for name, path in (("demand", base.data.data_path),
                       ("adjacency", base.data.adjacency_path)):
        if manifest["data"][str(path)] != expected_data[name]:
            raise RuntimeError(f"data differs from exp01: {path}")
    for name, value in (("config.json", configuration), ("input_manifest.json", manifest)):
        path = root / name
        if path.exists() and read_json(path) != value:
            raise RuntimeError(f"run identity changed: {path}; use a different output directory")
    # Validate both files before writing either one.
    write_json(root / "config.json", configuration)
    write_json(root / "input_manifest.json", manifest)
    (root / "git_commit.txt").write_text(
        f'{identity["commit"]}\ndirty: {str(identity["dirty"]).lower()}\n', encoding="utf-8"
    )
    return manifest


def run_spectra(components, path: Path, features, weights: Weights) -> dict[str, Weights]:
    aligned, alignment = align_hidden(weights)
    alignment["verification"] = verify_alignment(
        components.model.location_specific, weights, aligned, features
    )
    representations = {"as_stored": weights, "aligned": aligned}
    for name, value in representations.items():
        directory = path / "static" / name
        arrays = directory / "arrays"
        arrays.mkdir(parents=True, exist_ok=True)
        summaries, factors = {}, {}
        for layer in ("down", "up"):
            centered = Decomposition.fit(value.matrix(layer))
            summaries[layer] = {
                "centered": centered.summary(),
                "uncentered": Decomposition.fit(value.matrix(layer), centered=False).summary(),
            }
            factors.update({f"{layer}_{key}": getattr(centered, key)
                            for key in ("mean", "u", "s", "vt")})
        write_json(directory / "spectra.json", {"status": "completed", "layers": summaries})
        np.savez_compressed(arrays / "factors.npz", **factors, **value.arrays())
    write_json(path / "static/aligned/alignment.json", alignment)
    write_json(path / "structure/diagnostics.json",
               structure_diagnostics(components.model, weights, features))
    return representations


def run_seed(seed: int, root: Path, stage: str, settings: Settings,
             retry_failed: bool, log) -> None:
    path = root / "seeds" / f"seed{seed}"
    path.mkdir(parents=True, exist_ok=True)
    config = seed_config(seed, settings.device)
    write_json(path / "config.json", asdict(config))
    # Static work and online adaptation use fresh, separately seeded components.
    if stage in ("all", "spectra", "static"):
        components = build_components(config)
        load_warmup_checkpoint(components, checkpoint(seed))
        train_indices, features = collect_features(
            components.model, components.data.train, settings.samples, settings.batch_size
        )
        weights = Weights.from_learner(components.model.location_specific)
        representations = run_spectra(components, path, features, weights)
        manifest_path = path / "manifest.json"
        selected = (read_json(manifest_path).get("diagnostic_indices", {})
                    if manifest_path.exists() else {})
        selected["train"] = train_indices
        if stage in ("all", "static"):
            for split, grid in (("validation", True), ("online", False)):
                dataset = getattr(components.data, split)
                # Smoke diagnostics use only the same prefix that is evaluated.
                probe = Subset(dataset, range(min(settings.limit, len(dataset)))) if settings.smoke else dataset
                selected[split], split_features = collect_features(
                    components.model, probe, settings.samples, settings.batch_size
                )
                for name, value in representations.items():
                    run_interventions(
                        components, value, split_features, dataset,
                        path / "static" / name / ("validation.json" if grid else "online_frozen.json"),
                        settings.ranks, grid=grid, random=name == "as_stored",
                        batch_size=settings.batch_size, limit=settings.limit, log=log,
                    )
        write_json(path / "manifest.json", {
            "checkpoint": str(checkpoint(seed)), "checkpoint_sha256": sha256(checkpoint(seed)),
            "diagnostic_indices": selected,
        })
        del components, features, representations
        gc.collect()
    if stage in ("all", "trajectory"):
        directory = path / "trajectory"
        metrics_path = directory / "metrics.json"
        if metrics_path.exists() and read_json(metrics_path).get("status") != "completed":
            if not retry_failed:
                raise RuntimeError(f"incomplete trajectory: {directory}; use --retry-failed")
            archive = path / "attempts" / datetime.now().strftime("%Y%m%dT%H%M%S%f")
            archive.mkdir(parents=True)
            shutil.move(str(directory), str(archive / "trajectory"))
            log.info("previous trajectory archived: %s", archive)
        components = build_components(config)
        load_warmup_checkpoint(components, checkpoint(seed))
        capture_trajectory(
            components, directory, checkpoint(seed).parent.parent / "metrics.json",
            max_steps=settings.limit, interval=4 if settings.smoke else config.data.steps_per_week,
            log=log,
        )
        analyze_trajectory(
            components, directory, (4, 76) if settings.smoke else (4, 8, 16, 32, 76),
            settings.samples, settings.batch_size, log,
        )
        del components
        gc.collect()


def aggregate(root: Path) -> dict:
    """Always derive aggregate numbers from persisted per-seed cases."""
    groups = {}
    completion = {}
    spectral_results = {}
    trajectory_results = {}
    for seed in SEEDS:
        path = root / "seeds" / f"seed{seed}"
        state = {}
        for representation in ("as_stored", "aligned"):
            prefix = path / "static" / representation
            for name in ("spectra", "validation", "online_frozen"):
                file = prefix / f"{name}.json"
                data = read_json(file) if file.exists() else {}
                state[f"{representation}/{name}"] = data.get("status", "missing")
                if name == "spectra":
                    if data:
                        spectral_results[f"{seed}/{representation}"] = data
                    continue
                for key, case in data.get("cases", {}).items():
                    group = groups.setdefault(f"{representation}/{name}/{key}", {
                        "representation": representation, "split": name,
                        "name": case["name"], "down_rank": case["down_rank"],
                        "up_rank": case["up_rank"], "projection_seed": case["projection_seed"],
                        "factor_parameters": case["factor_parameters"], "seeds": {},
                    })
                    group["seeds"][str(seed)] = case
        for name in ("metrics", "spectra", "intervention"):
            file = path / "trajectory" / f"{name}.json"
            state[f"trajectory/{name}"] = read_json(file).get("status") if file.exists() else "missing"
        trajectory_path = path / "trajectory/metrics.json"
        if trajectory_path.exists():
            trajectory_results[str(seed)] = read_json(trajectory_path)
        completion[str(seed)] = state
    for group in groups.values():
        cases = list(group["seeds"].values())
        group["num_seeds"] = len(cases)
        group["summary"] = {}
        for metric in METRICS:
            values = [case["prediction"][metric] for case in cases]
            changes = [case["relative_metric_change"][metric] for case in cases]
            group["summary"][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                "paired_relative_change_mean": (
                    float(np.mean(changes)) if all(x is not None for x in changes) else None
                ),
                "worst_relative_change": (
                    float(max(changes)) if all(x is not None for x in changes) else None
                ),
            }
        group["preserved_seeds"] = sum(case["practical_preservation"] for case in cases)
    complete = all(status == "completed" for states in completion.values() for status in states.values())
    record = {"status": "completed" if complete else "partial", "completion": completion,
              "groups": groups, "spectra": spectral_results,
              "trajectory": trajectory_results,
              "scope": read_json(root / "config.json")["scope"]}
    write_json(root / "metrics.json", record)
    report_path = root / "report.md"
    # The generated section is isolated so handwritten discussion survives reruns.
    marker = "<!-- exp03-generated-results -->"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else "# exp03 実験レポート\n"
    prefix = existing.split(marker)[0]
    rows = [marker, "", f"状態: {record['status']} / scope: {record['scope']}", "",
            "|表現|条件|r_d|r_u|seed数|MAE|global RMSE|sample RMSE|WMAPE (%)|",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in groups.values():
        if group["split"] != "online_frozen" or group["name"] == "random":
            continue
        summary = group["summary"]
        rows.append(
            f"|{group['representation']}|{group['name']}|{group['down_rank']}|{group['up_rank']}|"
            f"{group['num_seeds']}|{summary['mae']['mean']:.6f}|{summary['rmse']['mean']:.6f}|"
            f"{summary['sample_rmse']['mean']:.6f}|{100 * summary['wmape']['mean']:.4f}|"
        )
    rows += ["", "層別スペクトル（地点差）", "",
             "|seed/表現|層|r90|r95|r99|effective rank|",
             "|---|---|---:|---:|---:|---:|"]
    for key, result in spectral_results.items():
        for layer, spectrum in result["layers"].items():
            value = spectrum["centered"]
            rows.append(f"|{key}|{layer}|{value['r90']}|{value['r95']}|"
                        f"{value['r99']}|{value['effective_rank']:.4f}|")
    rows += ["", "Original online軌跡の照合", "",
             "|seed|状態|処理窓|更新回数|exp01との照合|", "|---|---|---:|---:|---|"]
    for seed, result in trajectory_results.items():
        online = result.get("online", {})
        comparison = result.get("exp01_comparison")
        matched = "smoke:対象外" if comparison is None else str(all(comparison.values()))
        rows.append(f"|{seed}|{result['status']}|{online.get('processed_steps', '—')}|"
                    f"{online.get('online_updates', '—')}|{matched}|")
    rows += ["", "同じcheckpointの更新なしOriginalが比較基準。smoke・欠測条件は本実験の結論に使わない。",
             "詳細な対応差・許容幅達成数はmetrics.jsonを参照。", ""]
    report_path.write_text(prefix.rstrip() + "\n\n" + "\n".join(rows), encoding="utf-8")
    return record


def run_suite(root: Path, settings: Settings, *, stage: str = "all",
              only_seed: int | None = None, retry_failed: bool = False) -> dict:
    if stage not in ("all", "spectra", "static", "trajectory"):
        raise ValueError("unknown stage")
    if only_seed is not None and only_seed not in SEEDS:
        raise ValueError("unsupported seed")
    if settings.smoke and root.resolve() == Path("experiments/exp03").resolve():
        raise ValueError("smoke results must use a separate --output, e.g. tmp/exp03-smoke")
    root = Path(root)
    source_root = Path("experiments/exp01").resolve()
    if root.resolve().is_relative_to(source_root):
        raise ValueError("exp01 is a read-only source; choose a different output directory")
    torch.set_num_threads(settings.threads)
    with suite_lock(root):
        prepare_suite(root, settings)
        log = logger(root)
        for seed in SEEDS if only_seed is None else (only_seed,):
            handler = None
            try:
                directory = root / "seeds" / f"seed{seed}"
                directory.mkdir(parents=True, exist_ok=True)
                handler = logging.FileHandler(directory / "run.log", encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
                log.addHandler(handler)
                log.info("exp03 seed=%d stage=%s scope=%s", seed, stage,
                         "smoke" if settings.smoke else "full")
                run_seed(seed, root, stage, settings, retry_failed, log)
            except BaseException:
                log.exception("exp03 interrupted/failed seed=%d", seed)
                aggregate(root)
                raise
            finally:
                if handler is not None:
                    log.removeHandler(handler)
                    handler.close()
            aggregate(root)
        return aggregate(root)
