"""exp01の独立した5 seed試行とseed間集計。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import shutil
import statistics
import subprocess
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from .artifacts import GitState, capture_git_state, write_json
from .config import ExperimentConfig

SEEDS = (42, 43, 44, 45, 46)
METRICS = ("mae", "rmse", "sample_rmse", "wmape")
PAPER_CHICAGO_T = {"mae": 0.72, "rmse": 2.06, "wmape": 0.368}


def _json_dict(value: object) -> dict:
    """Pathを含むdataclass由来の設定をJSONで比較できる形にする。"""

    return json.loads(json.dumps(value, default=str))


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _seed_config(base: ExperimentConfig, root: Path, seed: int) -> ExperimentConfig:
    return replace(
        base,
        warmup=replace(base.warmup, seed=seed),
        artifacts=replace(base.artifacts, experiments_root=root / "seeds"),
    )


def _suite_config(base: ExperimentConfig, root: Path) -> dict:
    return {
        "seeds": list(SEEDS),
        "shared_config": _json_dict(asdict(base)),
        "data_sha256": {
            "demand": _file_sha256(base.data.data_path),
            "adjacency": _file_sha256(base.data.adjacency_path),
        },
        "output_root": str(root),
        "sample_std_ddof": 1,
    }


def _read_git_state(path: Path) -> GitState:
    lines = dict(
        line.split(": ", 1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if ": " in line
    )
    if not lines.get("commit") or lines.get("dirty") not in ("true", "false"):
        raise ValueError(f"invalid git revision record: {path}")
    return GitState(lines["commit"], lines["dirty"] == "true")


def _write_git_state(path: Path, state: GitState) -> None:
    path.write_text(
        f"commit: {state.commit}\ndirty: {str(state.dirty).lower()}\n",
        encoding="utf-8",
    )


def _source_is_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git status failed: {result.stderr.strip()}")
    return not result.stdout.strip()


def _seed_status(path: Path) -> str:
    if not path.exists():
        return "not_started"
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return "not_started" if not any(path.iterdir()) else "invalid"
    try:
        return str(_read_json(metrics_path).get("status", "invalid"))
    except (OSError, ValueError):
        return "invalid"


def _completed_seed(root: Path, base: ExperimentConfig, seed: int, state: GitState) -> dict:
    path = root / "seeds" / f"seed{seed}"
    metrics = _read_json(path / "metrics.json")
    if metrics.get("status") != "completed" or metrics.get("run_mode") != "warmup_then_online":
        raise ValueError(f"seed{seed} is not a completed independent run")
    expected_config = _json_dict(asdict(_seed_config(base, root, seed)))
    if _read_json(path / "config.json") != expected_config:
        raise ValueError(f"seed{seed} config differs from the planned conditions")
    if _read_git_state(path / "git_commit.txt") != state:
        raise ValueError(f"seed{seed} uses a different Git revision")
    if not (path / "checkpoints" / "warmup.pt").is_file():
        raise FileNotFoundError(f"seed{seed} warm-up checkpoint is missing")
    online = metrics.get("online")
    if not isinstance(online, dict):
        raise ValueError(f"seed{seed} has no online metrics")
    values = {}
    for key in METRICS:
        value = online.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"seed{seed} has invalid {key}")
        values[key] = float(value)
    processed = online.get("processed_steps")
    updates = online.get("online_updates")
    elapsed = metrics.get("elapsed_seconds")
    if (
        isinstance(processed, bool) or not isinstance(processed, int) or processed <= 0
        or isinstance(updates, bool) or not isinstance(updates, int) or not 0 <= updates <= processed
        or isinstance(elapsed, bool) or not isinstance(elapsed, (float, int))
        or not math.isfinite(elapsed) or elapsed < 0
    ):
        raise ValueError(f"seed{seed} has invalid processing counts or duration")
    values["processed_steps"] = processed
    values["online_updates"] = updates
    values["elapsed_seconds"] = float(elapsed)
    return values


def aggregate(root: Path, base: ExperimentConfig, state: GitState) -> dict:
    """5件そろうまでは平均を出さず、完了済みseedの値だけ保持する。"""

    by_seed: dict[str, dict] = {}
    for seed in SEEDS:
        status = _seed_status(root / "seeds" / f"seed{seed}")
        record = {"status": status}
        if status == "completed":
            record.update(_completed_seed(root, base, seed, state))
        by_seed[str(seed)] = record

    result: dict = {
        "status": "completed" if all(item["status"] == "completed" for item in by_seed.values()) else "running",
        "seeds": by_seed,
        "completed_seeds": sum(item["status"] == "completed" for item in by_seed.values()),
    }
    if result["status"] == "completed":
        if len({by_seed[str(seed)]["processed_steps"] for seed in SEEDS}) != 1:
            raise ValueError("online window count differs across seeds")
        if len({by_seed[str(seed)]["online_updates"] for seed in SEEDS}) != 1:
            raise ValueError("online update count differs across seeds")
        result["summary"] = {
            key: {
                "mean": statistics.mean(by_seed[str(seed)][key] for seed in SEEDS),
                "std": statistics.stdev(by_seed[str(seed)][key] for seed in SEEDS),
            }
            for key in METRICS
        }
        result["summary"]["wmape_percent"] = {
            "mean": result["summary"]["wmape"]["mean"] * 100,
            "std": result["summary"]["wmape"]["std"] * 100,
        }
        result["paper_comparison"] = {
            key: {
                "paper_mean": paper_value,
                "relative_difference": (
                    result["summary"][key]["mean"] - paper_value
                ) / paper_value,
                "within_5_percent": abs(
                    result["summary"][key]["mean"] - paper_value
                ) / paper_value <= 0.05,
            }
            for key, paper_value in PAPER_CHICAGO_T.items()
        }
        result["all_primary_means_within_5_percent"] = all(
            item["within_5_percent"] for item in result["paper_comparison"].values()
        )
    return result


def _archive_incomplete(path: Path) -> Path:
    """失敗・中断した試行を同じseedのattemptsへ退避し、削除しない。"""

    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")
    archive = path / "attempts" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    for child in tuple(path.iterdir()):
        if child.name != "attempts":
            shutil.move(str(child), str(archive / child.name))
    return archive


def _log(stream, message: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} | {message}"
    stream.write(line + "\n")
    stream.flush()
    print(line, flush=True)


def run_suite(
    root: Path = Path("experiments/exp01"),
    *,
    retry_failed: bool = False,
    base_config: ExperimentConfig | None = None,
    run_one: Callable | None = None,
    verify_git: bool = True,
) -> dict:
    """5 seedを順番に実行。既完了は検証してスキップし、失敗再試行は明示する。"""

    from run_dol import main as run_dol

    root = Path(root)
    if not (root / "strategy.md").is_file():
        raise FileNotFoundError(root / "strategy.md")
    base = base_config or ExperimentConfig()
    run_one = run_one or run_dol
    current = capture_git_state(Path.cwd())
    prior_metrics = _read_json(root / "metrics.json") if (root / "metrics.json").exists() else {"status": "not_started"}
    first_run = prior_metrics.get("status") == "not_started"
    if first_run:
        if verify_git and current.dirty:
            raise RuntimeError("commit all planned code and strategy before running exp01")
        state = current
    else:
        state = _read_git_state(root / "git_commit.txt")
        if verify_git and (state.dirty or current.commit != state.commit or not _source_is_clean()):
            raise RuntimeError("source revision changed since exp01 started")

    planned = _suite_config(base, root)
    if not first_run and _read_json(root / "config.json") != planned:
        raise ValueError("exp01 configuration or data SHA-256 changed")

    log_path = root / "run.log"
    with log_path.open("a", encoding="utf-8") as log:
        try:
            fcntl.flock(log, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("exp01 is already running") from error
        if first_run:
            write_json(root / "config.json", planned)
            _write_git_state(root / "git_commit.txt", state)
            started_at = datetime.now().astimezone().isoformat(timespec="seconds")
        else:
            started_at = str(prior_metrics.get("started_at", "unknown"))
        _log(log, f"exp01 started revision={state.commit} retry_failed={retry_failed}")
        try:
            for seed in SEEDS:
                if verify_git and (
                    capture_git_state(Path.cwd()).commit != state.commit
                    or not _source_is_clean()
                ):
                    raise RuntimeError("source revision changed during exp01")
                if _suite_config(base, root) != planned:
                    raise RuntimeError("exp01 input data changed during the suite")
                seed_path = root / "seeds" / f"seed{seed}"
                status = _seed_status(seed_path)
                if status == "completed":
                    _completed_seed(root, base, seed, state)
                    _log(log, f"seed{seed} completed already; skipped")
                    continue
                if status != "not_started":
                    if not retry_failed or status not in ("failed", "running", "invalid"):
                        raise RuntimeError(
                            f"seed{seed} status={status}; use --retry-failed to archive and rerun"
                        )
                    archive = _archive_incomplete(seed_path)
                    _log(log, f"seed{seed} previous attempt archived={archive}")
                _log(log, f"seed{seed} started")
                run_one(
                    experiment_name=f"seed{seed}",
                    config=_seed_config(base, root, seed),
                    git_state=state,
                    create_human_documents=False,
                )
                _completed_seed(root, base, seed, state)
                _log(log, f"seed{seed} completed")
                partial = aggregate(root, base, state)
                partial["started_at"] = started_at
                write_json(root / "metrics.json", partial)
            result = aggregate(root, base, state)
            result["started_at"] = started_at
            write_json(root / "metrics.json", result)
            _log(log, "exp01 completed")
            return result
        except BaseException as error:
            try:
                result = aggregate(root, base, state)
            except Exception:
                result = {"seeds": {}, "completed_seeds": 0}
            result["status"] = "failed"
            result["started_at"] = started_at
            result["error"] = {"type": type(error).__name__, "message": str(error)}
            write_json(root / "metrics.json", result)
            _log(log, f"exp01 stopped: {type(error).__name__}: {error}")
            raise
        finally:
            fcntl.flock(log, fcntl.LOCK_UN)
