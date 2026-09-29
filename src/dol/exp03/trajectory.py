"""exp03 C: Originalのonline軌跡を保存し、終了後にcloneへ介入する。"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..pipeline import run_online_evaluation
from .diagnostics import collect_features, compare_functions
from .evaluation import Intervention, candidate_weights, predict, prediction_summary
from .storage import Records, read_json, sha256, write_json
from .svd import Decomposition, Weights, fixed_basis_error


def load_snapshot(path: Path) -> Weights:
    with np.load(path, allow_pickle=False) as archive:
        return Weights(*(archive[key].astype(np.float64) for key in
                         ("down", "down_bias", "up", "up_bias")))


def distribution_and_error(components, entries: list[dict], evaluation) -> list[dict]:
    """Past-only RFF input shift and subsequent errors, computed after capture.

    Fixed local randomness is independent of the replay buffer RNG.
    Input bandwidth is 1 in train-standardized units.
    """
    period = components.config.data.steps_per_week
    dataset = components.data.online
    rng = np.random.default_rng(303)
    frequency = rng.normal(size=16)
    phase = rng.uniform(0, 2 * np.pi, size=16)

    def embedding_mean(values):
        return (np.sqrt(2 / 16) * np.cos(
            values[..., None] * frequency + phase
        )).mean(axis=0)

    output = []
    for entry in entries:
        completed = entry["processed_windows"]
        end = entry["forecast_origin_index"]
        if completed == 0 or end < 2 * period:
            continue
        older = dataset.series[end - 2 * period:end - period].numpy()
        recent = dataset.series[end - period:end].numpy()
        scores = np.linalg.norm(embedding_mean(recent) - embedding_mean(older), axis=-1)
        next_end = min(completed + period, evaluation.processed_steps)
        error = (evaluation.predictions[completed:next_end].double()
                 - evaluation.targets[completed:next_end].double())
        output.append({
            "step": entry["step"], "input_end_exclusive": end,
            "rff_dimension": 16, "rff_seed": 303, "bandwidth": 1.,
            "node_input_shift": scores.tolist(), "mean_input_shift": float(scores.mean()),
            "following_windows": next_end - completed,
            "following_node_mae": error.abs().mean((0, 1)).tolist() if error.numel() else None,
            "following_node_rmse": error.square().mean((0, 1)).sqrt().tolist()
            if error.numel() else None,
        })
    return output


def capture_trajectory(components, root: Path, expected_path: Path, *,
                       max_steps: int | None, interval: int, log) -> dict:
    """Callbackは保存だけを行う。乱数抽出、モデル変更、診断推論を行わない。"""
    metadata_path = root / "metrics.json"
    if metadata_path.exists():
        record = read_json(metadata_path)
        if record.get("status") == "completed":
            for entry in read_json(root / "snapshot_index.json")["snapshots"]:
                snapshot = root / entry["file"]
                if not snapshot.is_file() or sha256(snapshot) != entry.get("sha256"):
                    raise RuntimeError(f"snapshot missing or changed: {snapshot}")
            return record
        raise RuntimeError(f"incomplete trajectory: {root}; use --retry-failed")
    arrays = root / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    write_json(metadata_path, {"status": "running"})
    snapshots = []
    updates = 0

    def save_snapshot(step: int, phase: str) -> None:
        weights = Weights.from_learner(components.model.location_specific)
        name = f"weights_{step + 1:07d}.npz"
        np.savez_compressed(arrays / name, **{
            key: value.astype(np.float32) for key, value in weights.arrays().items()
        })
        forecast_index = components.data.online.first_target_start + step
        snapshots.append({
            "file": f"arrays/{name}", "sha256": sha256(arrays / name),
            "step": step, "processed_windows": step + 1,
            "phase_at_last_prediction": phase, "updates": updates,
            "forecast_origin_index": forecast_index,
            "minutes_from_series_start": forecast_index * components.config.data.interval_minutes,
            "last_released_forecast_step": (
                step - components.config.data.forecast_horizon
                if step >= components.config.data.forecast_horizon else None
            ),
        })
        write_json(root / "snapshot_index.json", {"snapshots": snapshots})

    def on_step(result) -> None:
        nonlocal updates
        updates += int(result.update_mae is not None)
        if (result.step + 1) % interval == 0:
            save_snapshot(result.step, result.phase)

    save_snapshot(-1, "before_online")
    try:
        evaluation = run_online_evaluation(
            components, max_steps=max_steps, collect_predictions=True,
            seed_validation_before_online=True, step_callback=on_step,
            progress_interval=1000,
            progress_callback=lambda done, total, count: log.info(
                "Original online %d/%d updates=%d", done, total, count
            ),
        )
        actual = {
            **asdict(evaluation.metrics),
            **{key: getattr(evaluation, key) for key in (
                "processed_steps", "online_updates", "awake_steps", "hibernate_steps",
                "validation_samples_seen", "initial_replay_size", "elapsed_seconds",
            )},
        }
        expected = read_json(expected_path)["online"]
        matches = None
        if max_steps is None:
            matches = {
                key: bool(np.isclose(actual[key], expected[key], rtol=1e-5, atol=1e-6))
                for key in ("mae", "rmse", "sample_rmse", "wmape")
            }
            matches.update({
                key: actual[key] == expected[key]
                for key in ("processed_steps", "online_updates", "awake_steps",
                            "hibernate_steps", "validation_samples_seen", "initial_replay_size")
            })
        node_metrics = prediction_summary(
            evaluation.predictions, evaluation.targets, evaluation.predictions
        )
        write_json(root / "shift_and_error.json", {
            "diagnostics": distribution_and_error(components, snapshots, evaluation)
        })
        record = {
            "status": "completed" if matches is None or all(matches.values()) else "failed",
            "scope": "smoke" if max_steps is not None else "full",
            "online": actual, "node_mae": node_metrics["node_mae"],
            "node_rmse": node_metrics["node_rmse"],
            "exp01_comparison": matches, "expected_online": expected,
        }
        write_json(metadata_path, record)
        if record["status"] != "completed":
            raise RuntimeError("Original online does not match exp01; see trajectory/metrics.json")
        return record
    except BaseException as error:
        if read_json(metadata_path).get("status") == "running":
            write_json(metadata_path, {"status": "failed", "error": str(error)})
        raise


def analyze_trajectory(components, root: Path, ranks, diagnostic_count: int,
                       batch_size: int, log) -> None:
    entries = read_json(root / "snapshot_index.json")["snapshots"]
    reference_indices, features = collect_features(components.model, components.data.train,
                                   diagnostic_count, batch_size)
    # All offline computations occur after the original online process.
    from torch.utils.data import Subset
    from .diagnostics import sample_indices
    dataset = Subset(components.data.train,
                     sample_indices(len(components.data.train), diagnostic_count))
    write_json(root / "diagnostic_manifest.json", {"train_indices": reference_indices})
    spectra = Records(root / "spectra.json")
    interventions = Records(root / "intervention.json")
    initial = load_snapshot(root / entries[0]["file"])
    initial_svd = {layer: Decomposition.fit(initial.matrix(layer)) for layer in ("down", "up")}
    learner = components.model.location_specific
    original_end = Weights.from_learner(learner)
    previous = initial
    try:
        for entry in entries:
            snapshot = root / entry["file"]
            if sha256(snapshot) != entry["sha256"]:
                raise RuntimeError(f"snapshot changed: {snapshot}")
            key = str(entry["step"])
            state = load_snapshot(root / entry["file"])
            log.info("snapshot analysis step=%s", key)
            decompositions = {layer: Decomposition.fit(state.matrix(layer)) for layer in ("down", "up")}
            if not spectra.contains(key):
                record = {"step": entry["step"], "updates": entry["updates"], "layers": {}}
                for layer in ("down", "up"):
                    matrix = state.matrix(layer)
                    difference = matrix - previous.matrix(layer)
                    cumulative = matrix - initial.matrix(layer)
                    fixed = {str(r): fixed_basis_error(difference, initial_svd[layer], r)
                             for r in ranks}
                    distances = {}
                    for rank in ranks:
                        first = initial_svd[layer].vt[:rank]
                        current = decompositions[layer].vt[:rank]
                        overlap = np.linalg.svd(first @ current.T, compute_uv=False)
                        distances[str(rank)] = {
                            "principal_angles_radians": np.arccos(np.clip(overlap, 0, 1)).tolist(),
                            "projector_distance": float(np.sqrt(max(
                                0., 2 * rank - 2 * np.sum(overlap ** 2)
                            ))),
                        }
                    record["layers"][layer] = {
                        "state": decompositions[layer].summary(),
                        "weekly_update": Decomposition.fit(difference).summary(),
                        "cumulative_update": Decomposition.fit(cumulative).summary(),
                        "weekly_update_norm": float(np.linalg.norm(difference)),
                        "no_update": bool(np.array_equal(matrix, previous.matrix(layer))),
                        "fixed_basis_error": fixed, "subspace_distance": distances,
                    }
                spectra.save(key, record)
            if not interventions.contains(key):
                state.apply(learner)
                baseline, truth = predict(components, dataset, batch_size)
                results = []
                cases = [Intervention(mode, r if mode != "up" else None,
                                      r if mode != "down" else None)
                         for mode in ("down", "up", "both") for r in ranks]
                for case in cases:
                    candidate = candidate_weights(state, case, decompositions)
                    candidate.apply(learner)
                    prediction, _ = predict(components, dataset, batch_size)
                    results.append({
                        **asdict(case),
                        "function": compare_functions(learner, state, candidate, features, batch_size),
                        "prediction": prediction_summary(prediction, truth, baseline),
                    })
                interventions.save(key, {"step": entry["step"], "cases": results})
            previous = state
        spectra.complete()
        interventions.complete()
    finally:
        original_end.apply(learner)
