"""exp02の全rank・5 seed実験を実行、再開、集計する。"""

from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import shutil
import statistics
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from .artifacts import write_json
from .config import ExperimentConfig
from .evaluation.exp02 import OnlineDiagnostics
from .layers.napl import NAPLLocationLearner, ZeroLocationLearner
from .pipeline import (
    build_components,
    load_warmup_checkpoint,
    run_online_evaluation,
    run_warmup,
)

RANKS = (0, 1, 2, 4, 8, 16, 32, 48, 64)
SEEDS = (42, 43, 44, 45, 46)
METRICS = ("mae", "wmape", "sample_rmse", "rmse")
ROOT = Path("experiments/exp02")


@dataclass(frozen=True)
class Variant:
    structure: str
    rank: int

    @property
    def relative_path(self) -> Path:
        if self.structure == "no_lsl":
            return Path("controls/no_lsl")
        return Path("variants") / self.structure / f"r{self.rank}"

    @property
    def branches(self) -> tuple[str, ...]:
        if self.structure == "with_shared":
            return ("frozen",) if self.rank == 0 else ("e_only", "frozen")
        if self.structure == "no_shared":
            return ("eb_update",)
        return ("frozen",)


def all_variants() -> tuple[Variant, ...]:
    return (
        *(Variant("with_shared", rank) for rank in RANKS),
        *(Variant("no_shared", rank) for rank in RANKS if rank > 0),
        Variant("no_lsl", 0),
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _check_source_clean() -> None:
    result = subprocess.run(
        [
            "git", "status", "--porcelain", "--",
            "src", "pyproject.toml", "uv.lock", ".gitignore",
            "experiments/exp02/strategy.md",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "Commit exp02 source, strategy, .gitignore and dependencies before starting; "
            "mixed code revisions would invalidate paired runs."
        )


def _worktree_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
    return bool(result.stdout.strip())


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def planned_config(base: ExperimentConfig) -> dict[str, Any]:
    return {
        "schema": 1,
        "ranks": list(RANKS),
        "seeds": list(SEEDS),
        "conditions": ["with_shared/e_only", "with_shared/frozen", "no_shared/eb_update"],
        "base_config": _jsonable(asdict(base)),
        "data_sha256": {
            "demand": _sha256(base.data.data_path),
            "adjacency": _sha256(base.data.adjacency_path),
        },
        "code_commit": _git_commit(),
        "sample_std_ddof": 1,
    }


def _factory(variant: Variant, config: ExperimentConfig) -> Callable[[], nn.Module]:
    def make() -> nn.Module:
        if variant.structure == "no_lsl":
            return ZeroLocationLearner()
        return NAPLLocationLearner(
            num_nodes=config.data.num_nodes,
            channels=config.model.residual_channels,
            bottleneck_channels=config.model.lsl_bottleneck_channels,
            dropout=config.model.lsl_dropout,
            rank=variant.rank,
            with_shared=variant.structure == "with_shared",
        )

    return make


def _log(stream: Any, message: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} | {message}"
    stream.write(line + "\n")
    stream.flush()
    print(line, flush=True)


def _archive_incomplete(path: Path, names: tuple[str, ...]) -> Path:
    """未完了成果物をattemptsへ移し、既存データを失わず再試行する。"""

    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")
    archive = path / "attempts" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    for name in names:
        source = path / name
        if source.exists():
            shutil.move(str(source), str(archive / name))
    return archive


def _warmup(
    root: Path,
    variant: Variant,
    config: ExperimentConfig,
    log: Any,
    retry_failed: bool,
) -> tuple[Path, dict[str, Any]]:
    path = root / variant.relative_path / "seeds" / f"seed{config.warmup.seed}"
    checkpoint = path / "checkpoints" / "warmup.pt"
    record_path = path / "warmup.json"
    identity = {
        "structure": variant.structure,
        "rank": variant.rank,
        "seed": config.warmup.seed,
        "config": _jsonable(asdict(config)),
    }
    if record_path.exists():
        record = _read_json(record_path)
        if record.get("identity") != identity:
            raise RuntimeError(f"mismatched warm-up: {record_path}")
        if record.get("status") == "completed":
            if _read_json(path / "config.json") != identity:
                raise RuntimeError(f"warm-up config changed: {path / 'config.json'}")
            if not checkpoint.exists() or _sha256(checkpoint) != record["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint missing or changed: {checkpoint}")
            return checkpoint, record
        if not retry_failed:
            raise RuntimeError(f"incomplete warm-up: {record_path}; use --retry-failed")
        archive = _archive_incomplete(path, ("warmup.json", "config.json", "checkpoints"))
        _log(log, f"warm-up previous attempt archived={archive}")
    if checkpoint.exists():
        if not retry_failed:
            raise RuntimeError(f"unverified checkpoint: {checkpoint}; use --retry-failed")
        archive = _archive_incomplete(path, ("config.json", "checkpoints"))
        _log(log, f"warm-up unverified checkpoint archived={archive}")

    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "config.json", identity)
    write_json(record_path, {"status": "running", "identity": identity})
    _log(log, f"warm-up started {variant.relative_path} seed={config.warmup.seed}")
    try:
        components = build_components(config, _factory(variant, config))
        learner = components.model.location_specific
        with torch.no_grad():
            initial_norm = (
                float(learner.generated_weights().norm().item())
                if hasattr(learner, "generated_weights") else 0.0
            )
        result = run_warmup(
            components,
            checkpoint,
            epoch_callback=lambda epoch: _log(
                log,
                f"warm-up {variant.structure} r={variant.rank} seed={config.warmup.seed} "
                f"epoch={epoch.epoch} train_mae={epoch.train_mae:.6f} "
                f"validation_mae={epoch.validation_mae:.6f}",
            ),
        )
        with torch.no_grad():
            trained_norm = (
                float(learner.generated_weights().norm().item())
                if hasattr(learner, "generated_weights") else 0.0
            )
        record = {
            "status": "completed",
            "identity": identity,
            "best_epoch": result.best_epoch,
            "best_validation_mae": result.best_validation_mae,
            "history": [asdict(epoch) for epoch in result.history],
            "initial_weight_norm": initial_norm,
            "trained_weight_norm": trained_norm,
            "checkpoint_sha256": _sha256(checkpoint),
        }
        write_json(record_path, record)
        _log(log, f"warm-up completed {variant.relative_path} seed={config.warmup.seed}")
        del components
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return checkpoint, record
    except BaseException as error:
        write_json(
            record_path,
            {
                "status": "failed",
                "identity": identity,
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        raise


def _online(
    root: Path,
    variant: Variant,
    branch: str,
    config: ExperimentConfig,
    checkpoint: Path,
    warmup: dict[str, Any],
    log: Any,
    retry_failed: bool,
) -> None:
    path = root / variant.relative_path / "seeds" / f"seed{config.warmup.seed}" / branch
    metrics_path = path / "metrics.json"
    if metrics_path.exists():
        record = _read_json(metrics_path)
        if record.get("checkpoint_sha256") != warmup["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint mismatch: {metrics_path}")
        if record.get("status") == "completed":
            if (
                record.get("structure") != variant.structure
                or record.get("rank") != variant.rank
                or record.get("seed") != config.warmup.seed
                or record.get("branch") != branch
            ):
                raise RuntimeError(f"online run identity mismatch: {metrics_path}")
            return
        if not retry_failed:
            raise RuntimeError(f"incomplete online run: {metrics_path}; use --retry-failed")
        archive = _archive_incomplete(path, ("metrics.json", "arrays"))
        _log(log, f"online previous attempt archived={archive}")
    path.mkdir(parents=True, exist_ok=True)
    write_json(metrics_path, {"status": "running", "checkpoint_sha256": warmup["checkpoint_sha256"]})
    _log(log, f"online started {variant.relative_path} seed={config.warmup.seed} {branch}")
    try:
        components = build_components(config, _factory(variant, config))
        load_warmup_checkpoint(components, checkpoint)
        mode = {"e_only": "coefficients", "eb_update": "all", "frozen": "none"}[branch]
        learner = components.model.location_specific
        before = {key: value.detach().clone() for key, value in learner.state_dict().items()}
        total_windows = len(components.data.online)
        diagnostics = OnlineDiagnostics(components, total_windows)
        if torch.cuda.is_available() and config.runtime.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(torch.device(config.runtime.device))
        online = run_online_evaluation(
            components,
            update_mode=mode,
            seed_validation_before_online=True,
            step_callback=diagnostics.on_step,
            progress_interval=1_000,
            progress_callback=lambda processed, total, updates: _log(
                log,
                f"online {variant.structure} r={variant.rank} seed={config.warmup.seed} "
                f"{branch} processed={processed}/{total} updates={updates}",
            ),
        )
        diagnostics.finish(
            online.processed_steps - 1, path / "arrays" / "diagnostics.npz"
        )
        after = learner.state_dict()
        if branch == "frozen" and any(
            not torch.equal(value, after[key]) for key, value in before.items()
        ):
            raise RuntimeError("frozen LSL changed online")
        if branch == "e_only" and any(
            not torch.equal(value, after[key]) for key, value in before.items() if key != "E"
        ):
            raise RuntimeError("shared weights or basis changed during E-only online")
        retained = sum(parameter.numel() for parameter in learner.parameters())
        if branch == "e_only":
            updated = learner.E.numel()
        elif branch == "eb_update":
            updated = retained
        else:
            updated = 0
        gpu_peak = (
            torch.cuda.max_memory_allocated(torch.device(config.runtime.device))
            if torch.cuda.is_available() and config.runtime.device.startswith("cuda")
            else None
        )
        record = {
            "status": "completed",
            "structure": variant.structure,
            "rank": variant.rank,
            "seed": config.warmup.seed,
            "branch": branch,
            "checkpoint_sha256": warmup["checkpoint_sha256"],
            "metrics": asdict(online.metrics),
            "processed_steps": online.processed_steps,
            "online_updates": online.online_updates,
            "awake_steps": online.awake_steps,
            "hibernate_steps": online.hibernate_steps,
            "elapsed_seconds": online.elapsed_seconds,
            "inference_seconds": online.inference_seconds,
            "update_seconds": online.update_seconds,
            "inference_ms_per_window": 1000 * online.inference_seconds / online.processed_steps,
            "update_ms_per_awake_step": (
                1000 * online.update_seconds / online.online_updates
                if online.online_updates else None
            ),
            "lsl_retained_parameters": retained,
            "lsl_online_parameters": updated,
            "gpu_peak_memory_bytes": gpu_peak,
            "diagnostics": "arrays/diagnostics.npz",
        }
        write_json(metrics_path, record)
        _log(
            log, f"online completed {variant.relative_path} seed={config.warmup.seed} "
            f"{branch} mae={online.metrics.mae:.6f} rmse={online.metrics.rmse:.6f}"
        )
        del components
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except BaseException as error:
        write_json(
            metrics_path,
            {
                "status": "failed",
                "checkpoint_sha256": warmup["checkpoint_sha256"],
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        raise


def aggregate(root: Path = ROOT) -> dict[str, Any]:
    """欠測を明示し、同じ条件のseedだけで平均±標本標準偏差を計算する。"""

    groups: list[dict[str, Any]] = []
    completed_runs = 0
    for variant in all_variants():
        for branch in variant.branches:
            runs: list[dict[str, Any]] = []
            for seed in SEEDS:
                path = (
                    root / variant.relative_path / "seeds" / f"seed{seed}"
                    / branch / "metrics.json"
                )
                if path.exists():
                    record = _read_json(path)
                    if record.get("status") == "completed":
                        runs.append(record)
            completed_runs += len(runs)
            summary = {}
            for key in METRICS:
                values = [float(run["metrics"][key]) for run in runs]
                summary[key] = {
                    "mean": statistics.mean(values) if values else None,
                    "std": statistics.stdev(values) if len(values) > 1 else None,
                    "n": len(values),
                }
            groups.append(
                {
                    "structure": variant.structure,
                    "rank": variant.rank,
                    "branch": branch,
                    "summary": summary,
                    "runs": {str(run["seed"]): run for run in runs},
                    "missing_seeds": [seed for seed in SEEDS if seed not in {r["seed"] for r in runs}],
                }
            )
    return {
        "status": "completed" if completed_runs == 130 else "partial",
        "completed_online_runs": completed_runs,
        "expected_online_runs": 130,
        "groups": groups,
    }


def run_suite(
    root: Path = ROOT,
    *,
    only_seed: int | None = None,
    only_rank: int | None = None,
    base: ExperimentConfig | None = None,
    retry_failed: bool = False,
) -> dict[str, Any]:
    if only_seed is not None and only_seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if only_rank is not None and only_rank not in RANKS:
        raise ValueError(f"rank must be one of {RANKS}")
    _check_source_clean()
    worktree_dirty = _worktree_dirty()
    base = base or ExperimentConfig()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    plan = planned_config(base)
    existing = _read_json(root / "config.json") if (root / "config.json").exists() else {}
    if existing and existing != plan:
        raise RuntimeError("exp02 config, data hash or code revision changed")
    log_path = root / "run.log"
    with log_path.open("a", encoding="utf-8") as log:
        try:
            fcntl.flock(log, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("exp02 is already running") from error
        if not existing:
            write_json(root / "config.json", plan)
            (root / "git_commit.txt").write_text(
                f"commit: {plan['code_commit']}\n"
                f"dirty: {str(worktree_dirty).lower()}\n"
                "source_dirty: false\n",
                encoding="utf-8",
            )
        _log(log, f"exp02 started commit={plan['code_commit']}")
        try:
            for variant in all_variants():
                if only_rank is not None and variant.rank != only_rank:
                    continue
                for seed in SEEDS:
                    if only_seed is not None and seed != only_seed:
                        continue
                    if _git_commit() != plan["code_commit"]:
                        raise RuntimeError("code commit changed during exp02")
                    _check_source_clean()
                    config = replace(base, warmup=replace(base.warmup, seed=seed))
                    checkpoint, warmup = _warmup(root, variant, config, log, retry_failed)
                    for branch in variant.branches:
                        _online(root, variant, branch, config, checkpoint, warmup, log, retry_failed)
                        write_json(root / "metrics.json", aggregate(root))
            result = aggregate(root)
            write_json(root / "metrics.json", result)
            _log(log, f"exp02 stopped status={result['status']} completed={result['completed_online_runs']}/130")
            return result
        except BaseException as error:
            result = aggregate(root)
            result["status"] = "failed"
            result["error"] = {"type": type(error).__name__, "message": str(error)}
            write_json(root / "metrics.json", result)
            _log(log, f"exp02 failed: {type(error).__name__}: {error}")
            raise
        finally:
            fcntl.flock(log, fcntl.LOCK_UN)
