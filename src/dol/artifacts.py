"""実験ディレクトリと再現用成果物を管理する。"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig


_EXPERIMENT_PATTERN = re.compile(r"^exp(\d+)$")


@dataclass(frozen=True)
class GitState:
    """実験開始時点のGit revisionと作業ツリー状態。"""

    commit: str
    dirty: bool


@dataclass(frozen=True)
class ExperimentPaths:
    """``experiments/exp<ii>`` 内で使うファイルをまとめる。"""

    root: Path
    config: Path
    metrics: Path
    git_commit: Path
    log: Path
    strategy: Path
    report: Path
    checkpoints: Path
    figures: Path
    arrays: Path

    @classmethod
    def from_root(cls, root: Path) -> "ExperimentPaths":
        root = Path(root)
        return cls(
            root=root,
            config=root / "config.json",
            metrics=root / "metrics.json",
            git_commit=root / "git_commit.txt",
            log=root / "run.log",
            strategy=root / "strategy.md",
            report=root / "report.md",
            checkpoints=root / "checkpoints",
            figures=root / "figures",
            arrays=root / "arrays",
        )


@dataclass
class ExperimentRun:
    """1回の実験について、保存先・logger・経過時間を保持する。"""

    paths: ExperimentPaths
    logger: logging.Logger
    started_at: str
    _start_time: float
    archived_attempt: Path | None = None

    @property
    def name(self) -> str:
        return self.paths.root.name

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self._start_time

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        write_json(self.paths.metrics, metrics)

    def save_prediction_arrays(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[Path, Path]:
        """任意保存の予測値・正解値を、Git対象外のnpyへ書き出す。"""

        self.paths.arrays.mkdir(parents=True, exist_ok=True)
        prediction_path = self.paths.arrays / "predictions.npy"
        target_path = self.paths.arrays / "targets.npy"
        np.save(prediction_path, predictions)
        np.save(target_path, targets)
        return prediction_path, target_path

    def close(self) -> None:
        for handler in tuple(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)


def create_experiment_run(
    config: ExperimentConfig,
    experiment_name: str | None = None,
    *,
    reuse_checkpoint: bool = False,
) -> ExperimentRun:
    """templateから新しい実験を作り、設定とGit状態を記録する。

    ``experiment_name=None`` なら、既存の最大番号の次を割り当てる。
    既存ディレクトリは通常上書きしない。checkpoint再利用時だけ
    旧ログ・設定・指標をattemptsへ退避し、checkpointを保持する。
    """

    experiments_root = Path(config.artifacts.experiments_root)
    template_root = experiments_root / "_template"
    git_state = capture_git_state(Path.cwd())

    experiments_root.mkdir(parents=True, exist_ok=True)
    if reuse_checkpoint and experiment_name is None:
        raise ValueError("reuse_checkpoint requires an experiment name")
    archived_attempt: Path | None = None
    if experiment_name is None:
        experiment_root = _reserve_next_experiment(experiments_root)
        newly_created = True
    else:
        if _EXPERIMENT_PATTERN.fullmatch(experiment_name) is None:
            raise ValueError("experiment_name must have the form exp<ii>")
        experiment_root = experiments_root / experiment_name
        if experiment_root.exists():
            newly_created = False
            if reuse_checkpoint:
                archived_attempt = _archive_previous_run(experiment_root)
            elif _experiment_has_started(experiment_root):
                raise FileExistsError(
                    f"{experiment_root} already contains an experiment run"
                )
        else:
            if reuse_checkpoint:
                raise FileNotFoundError(experiment_root)
            experiment_root.mkdir(parents=False)
            newly_created = True

    if newly_created and template_root.is_dir():
        shutil.copytree(template_root, experiment_root, dirs_exist_ok=True)

    paths = ExperimentPaths.from_root(experiment_root)
    paths.checkpoints.mkdir(parents=True, exist_ok=True)
    paths.figures.mkdir(parents=True, exist_ok=True)
    _ensure_human_documents(paths)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    write_json(paths.config, asdict(config))
    write_json(paths.metrics, {"status": "running", "started_at": started_at})
    _write_git_state(paths.git_commit, git_state)
    logger = _build_logger(paths.log, experiment_root.name)
    run = ExperimentRun(
        paths=paths,
        logger=logger,
        started_at=started_at,
        _start_time=time.perf_counter(),
        archived_attempt=archived_attempt,
    )
    logger.info("experiment=%s", run.name)
    logger.info("directory=%s", paths.root)
    logger.info("git_commit=%s dirty=%s", git_state.commit, git_state.dirty)
    if archived_attempt is not None:
        logger.info("previous attempt archived=%s", archived_attempt)
    return run



def _archive_previous_run(experiment_root: Path) -> Path:
    """完了済み試行の記録だけを退避し、warm-up checkpointは残す。"""

    paths = ExperimentPaths.from_root(experiment_root)
    checkpoint = paths.checkpoints / "warmup.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not paths.metrics.is_file():
        raise FileNotFoundError(paths.metrics)
    previous = json.loads(paths.metrics.read_text(encoding="utf-8"))
    if previous.get("status") not in ("completed", "failed"):
        raise RuntimeError("only completed or failed experiments can reuse a checkpoint")

    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f")
    archive = experiment_root / "attempts" / stamp
    archive.mkdir(parents=True, exist_ok=False)
    for path in (paths.config, paths.metrics, paths.git_commit, paths.log):
        if path.exists():
            shutil.move(str(path), str(archive / path.name))
    return archive


def write_json(path: Path, value: Any) -> None:
    """Pathやdataclassを含む値を、途中ファイルを経由してJSON保存する。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def capture_git_state(repository: Path) -> GitState:
    """Git未初期化・初回commit前でも失敗せず状態を返す。"""

    repository = Path(repository)
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else "unborn"

    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = status_result.returncode != 0 or bool(status_result.stdout.strip())
    return GitState(commit=commit, dirty=dirty)


def _reserve_next_experiment(experiments_root: Path) -> Path:
    numbers = []
    for child in experiments_root.iterdir():
        match = _EXPERIMENT_PATTERN.fullmatch(child.name)
        if child.is_dir() and match is not None:
            numbers.append(int(match.group(1)))

    number = max(numbers, default=-1) + 1
    while True:
        candidate = experiments_root / f"exp{number:02d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            number += 1
        else:
            return candidate


def _experiment_has_started(experiment_root: Path) -> bool:
    log_path = experiment_root / "run.log"
    if log_path.is_file() and log_path.stat().st_size > 0:
        return True
    checkpoint_root = experiment_root / "checkpoints"
    if checkpoint_root.is_dir() and any(checkpoint_root.iterdir()):
        return True
    metrics_path = experiment_root / "metrics.json"
    if metrics_path.is_file() and metrics_path.stat().st_size > 0:
        try:
            status = json.loads(metrics_path.read_text(encoding="utf-8")).get("status")
        except (json.JSONDecodeError, AttributeError):
            return True
        return status not in (None, "not_started")
    return False


def _build_logger(path: Path, experiment_name: str) -> logging.Logger:
    logger = logging.getLogger(f"dol.experiment.{experiment_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in tuple(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _write_git_state(path: Path, state: GitState) -> None:
    path.write_text(
        f"commit: {state.commit}\ndirty: {str(state.dirty).lower()}\n",
        encoding="utf-8",
    )


def _ensure_human_documents(paths: ExperimentPaths) -> None:
    if not paths.strategy.exists() or paths.strategy.stat().st_size == 0:
        paths.strategy.write_text(
            "# Strategy\n\n"
            "## 目的\n\n"
            "## 前実験からの変更\n\n"
            "## 仮説\n\n"
            "## 成功条件\n",
            encoding="utf-8",
        )
    if not paths.report.exists() or paths.report.stat().st_size == 0:
        paths.report.write_text(
            "# Report\n\n"
            "## 結果\n\n"
            "## 解釈\n\n"
            "## 次の実験\n",
            encoding="utf-8",
        )


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
