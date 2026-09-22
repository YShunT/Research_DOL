"""Chicago-Tのwarm-upとraw互換の独立したonline評価を実行する。

既存checkpointでonlineだけを再実行するときは
``python src/run_dol.py exp00 --reuse-checkpoint`` を使う。
"""

import gc
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from dol.artifacts import GitState, create_experiment_run
from dol.config import ExperimentConfig
from dol.pipeline import (
    build_components,
    load_warmup_checkpoint,
    run_online_evaluation,
    run_warmup,
)


def _existing_warmup(config: ExperimentConfig, experiment_name: str) -> dict:
    """既存結果とcheckpointを変更する前に、その整合性を確認する。"""

    root = config.artifacts.experiments_root / experiment_name
    checkpoint = root / "checkpoints" / "warmup.pt"
    metrics_path = root / "metrics.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    previous = json.loads(metrics_path.read_text(encoding="utf-8"))
    if previous.get("status") != "completed" or not isinstance(previous.get("warmup"), dict):
        raise RuntimeError("reuse requires a completed experiment with warm-up metrics")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("config") != asdict(config):
        raise ValueError("checkpoint config differs from the requested experiment config")
    warmup = previous["warmup"]
    if (
        warmup.get("best_epoch") != payload.get("epoch")
        or warmup.get("best_validation_mae") != payload.get("validation_mae")
    ):
        raise ValueError("warm-up metrics do not match the checkpoint")
    return warmup


def main(
    experiment_name: str | None = None,
    config: ExperimentConfig | None = None,
    reuse_checkpoint: bool = False,
    git_state: GitState | None = None,
    create_human_documents: bool = True,
) -> None:
    config = config or ExperimentConfig()
    if reuse_checkpoint and experiment_name is None:
        raise ValueError("reuse_checkpoint requires an experiment name")
    warmup_metrics = (
        _existing_warmup(config, experiment_name)
        if reuse_checkpoint and experiment_name is not None
        else None
    )
    experiment = create_experiment_run(
        config,
        experiment_name,
        reuse_checkpoint=reuse_checkpoint,
        git_state=git_state,
        create_human_documents=create_human_documents,
    )
    logger = experiment.logger
    checkpoint = experiment.paths.checkpoints / "warmup.pt"

    try:
        if not reuse_checkpoint:
            logger.info("building data, graph, model, optimizer, and SMB")
            warmup_components = build_components(config)
            logger.info(
                "model parameters=%d online_parameters=%d",
                warmup_components.model.parameter_count,
                warmup_components.model.online_parameter_count,
            )
            logger.info("warm-up started checkpoint=%s", checkpoint)
            warmup = run_warmup(
                warmup_components,
                checkpoint,
                epoch_callback=lambda record: logger.info(
                    "warm-up epoch=%d train_mae=%.6f validation_mae=%.6f",
                    record.epoch,
                    record.train_mae,
                    record.validation_mae,
                ),
            )
            warmup_metrics = {
                "best_epoch": warmup.best_epoch,
                "best_validation_mae": warmup.best_validation_mae,
                "history": [asdict(record) for record in warmup.history],
            }
            logger.info(
                "warm-up completed best_epoch=%d validation_mae=%.6f",
                warmup.best_epoch,
                warmup.best_validation_mae,
            )
            # rawのtrain_only→mode=onlineと同じく、optimizer・SMB・乱数を
            # 新しい実験インスタンスで初期化し、best modelだけを読み込む。
            del warmup_components
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            logger.info("warm-up skipped; reusing checkpoint=%s", checkpoint)

        logger.info("building fresh online model, optimizer, and SMB")
        components = build_components(config)
        load_warmup_checkpoint(components, checkpoint)
        initial_lr = float(components.optimizer.param_groups[0]["lr"])
        initial_optimizer_states = len(components.optimizer.state)
        initial_replay_seen = components.replay.num_seen
        if (
            initial_lr != config.warmup.learning_rate
            or initial_optimizer_states != 0
            or initial_replay_seen != 0
        ):
            raise RuntimeError("online model, optimizer, or SMB was not freshly initialized")
        logger.info(
            "online setup lr=%.6g optimizer_states=%d replay_seen=%d validation_passes=1",
            initial_lr,
            initial_optimizer_states,
            initial_replay_seen,
        )
        logger.info("online evaluation started")
        online = run_online_evaluation(
            components,
            seed_validation_before_online=True,
            collect_predictions=config.artifacts.save_predictions,
            progress_interval=1_000,
            progress_callback=lambda processed, total, updates: logger.info(
                "online processed=%d/%d updates=%d",
                processed,
                total,
                updates,
            ),
        )

        array_artifacts: dict[str, str] = {}
        if online.predictions is not None and online.targets is not None:
            prediction_path, target_path = experiment.save_prediction_arrays(
                online.predictions.numpy(),
                online.targets.numpy(),
            )
            array_artifacts = {
                "predictions": str(prediction_path.relative_to(experiment.paths.root)),
                "targets": str(target_path.relative_to(experiment.paths.root)),
            }

        assert warmup_metrics is not None
        with checkpoint.open("rb") as checkpoint_file:
            checkpoint_hash = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
        metrics = {
            "status": "completed",
            "run_mode": "checkpoint_online" if reuse_checkpoint else "warmup_then_online",
            "started_at": experiment.started_at,
            "elapsed_seconds": experiment.elapsed_seconds,
            "warmup": warmup_metrics,
            "online_setup": {
                "learning_rate": initial_lr,
                "optimizer_states_before": initial_optimizer_states,
                "replay_seen_before": initial_replay_seen,
                "validation_passes": 1,
            },
            "online": {
                **asdict(online.metrics),
                "processed_steps": online.processed_steps,
                "online_updates": online.online_updates,
                "awake_steps": online.awake_steps,
                "hibernate_steps": online.hibernate_steps,
                "validation_samples_seen": online.validation_samples_seen,
                "initial_replay_size": online.initial_replay_size,
                "elapsed_seconds": online.elapsed_seconds,
            },
            "artifacts": {
                "checkpoint": str(checkpoint.relative_to(experiment.paths.root)),
                "checkpoint_sha256": checkpoint_hash,
                **array_artifacts,
            },
        }
        if experiment.archived_attempt is not None:
            metrics["previous_attempt"] = str(
                experiment.archived_attempt.relative_to(experiment.paths.root)
            )
        experiment.write_metrics(metrics)
        logger.info(
            "online completed mae=%.6f rmse=%.6f sample_rmse=%.6f "
            "wmape=%.6f updates=%d/%d",
            online.metrics.mae,
            online.metrics.rmse,
            online.metrics.sample_rmse,
            online.metrics.wmape,
            online.online_updates,
            online.processed_steps,
        )
        logger.info("metrics=%s", experiment.paths.metrics)
    except Exception as error:
        logger.exception("experiment failed")
        experiment.write_metrics(
            {
                "status": "failed",
                "run_mode": "checkpoint_online" if reuse_checkpoint else "warmup_then_online",
                "started_at": experiment.started_at,
                "elapsed_seconds": experiment.elapsed_seconds,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        raise
    finally:
        experiment.close()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(experiment_name=sys.argv[1])
    elif len(sys.argv) == 3 and sys.argv[2] == "--reuse-checkpoint":
        main(experiment_name=sys.argv[1], reuse_checkpoint=True)
    else:
        raise SystemExit("usage: python src/run_dol.py [exp<ii> [--reuse-checkpoint]]")
