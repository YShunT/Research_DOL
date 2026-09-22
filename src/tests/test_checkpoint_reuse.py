"""checkpoint再利用時の成果物保護を確認する。"""

import json
from dataclasses import asdict

import pytest
import torch

from dol.artifacts import create_experiment_run
from dol.config import ArtifactConfig, ExperimentConfig
from run_dol import _existing_warmup


def _completed_experiment(tmp_path):
    config = ExperimentConfig(
        artifacts=ArtifactConfig(experiments_root=tmp_path / "experiments")
    )
    root = config.artifacts.experiments_root / "exp00"
    checkpoint = root / "checkpoints" / "warmup.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "config": asdict(config),
            "epoch": 2,
            "validation_mae": 0.25,
        },
        checkpoint,
    )
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "warmup": {
                    "best_epoch": 2,
                    "best_validation_mae": 0.25,
                    "history": [],
                },
            }
        )
    )
    (root / "run.log").write_text("old log\n")
    (root / "config.json").write_text("{}\n")
    (root / "git_commit.txt").write_text("old commit\n")
    return config, root, checkpoint


def test_reuse_archives_previous_run_but_keeps_checkpoint(tmp_path):
    config, root, checkpoint = _completed_experiment(tmp_path)
    before = checkpoint.read_bytes()
    assert _existing_warmup(config, "exp00")["best_epoch"] == 2

    run = create_experiment_run(config, "exp00", reuse_checkpoint=True)
    run.close()

    assert checkpoint.read_bytes() == before
    assert run.archived_attempt is not None
    archive = run.archived_attempt
    assert (archive / "run.log").read_text() == "old log\n"
    assert json.loads((archive / "metrics.json").read_text())["status"] == "completed"
    assert json.loads((root / "metrics.json").read_text())["status"] == "running"
    assert "previous attempt archived=" in (root / "run.log").read_text()


def test_reuse_refuses_running_experiment(tmp_path):
    config, root, checkpoint = _completed_experiment(tmp_path)
    (root / "metrics.json").write_text('{"status": "running"}\n')

    with pytest.raises(RuntimeError, match="completed experiment"):
        _existing_warmup(config, "exp00")
    with pytest.raises(RuntimeError, match="completed or failed"):
        create_experiment_run(config, "exp00", reuse_checkpoint=True)

    assert checkpoint.is_file()
    assert (root / "run.log").read_text() == "old log\n"
    assert not (root / "attempts").exists()
