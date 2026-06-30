"""Tiny novel-class admission runner."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from gons.baselines import (
    BaselineModel,
    fit_baseline,
    predict_dataframe,
    update_baseline_model,
)
from gons.continual import (
    cache_labeled_batch_by_class,
    consolidated_refresh,
    provisional_admit_small_item,
)
from gons.data.ton_iot import load_ton_iot_csv
from gons.evaluation.config import (
    TinyAdmissionExperimentConfig,
    load_tiny_admission_config,
)
from gons.evaluation.open_set import (
    predict_gons_dataframe,
    summarize_open_set_predictions,
)
from gons.fit import fit_gons
from gons.prototype.bank import serialize_projection_meta, serialize_threshold_meta
from gons.runtime import (
    build_run_id,
    current_git_commit,
    find_repo_root,
    python_runtime,
    utc_now_iso,
)
from gons.state.model import GONSModel


@dataclass(frozen=True, slots=True)
class TinyAdmissionRunResult:
    """Compact result returned by the GONS runner."""

    method: str
    metrics: dict[str, object]
    run_dir: Path
    run_id: str


@dataclass(frozen=True, slots=True)
class Episode:
    """One GONS tiny-admission episode."""

    episode_id: str
    novel_label: str
    protocol: str


@dataclass(frozen=True, slots=True)
class EvaluationSnapshot:
    """Compact metadata for one deployed state."""

    calibration_mode: str
    known_labels: tuple[str, ...]
    projection_dim: int
    projection_mode: str
    projection_version: int
    prototype_count: int
    tau_global: float


RunProgressCallback = Callable[[str, str], None]
ModelState = GONSModel | BaselineModel | list[GONSModel]


@dataclass(frozen=True, slots=True)
class _BurstPolicy:
    """Effective GONS policy for one burst under the current promotion state."""

    refresh_every_bursts: int


def run_tiny_admission_experiment(
    config_path: Path,
    *,
    run_id: str | None = None,
    repo_root: Path | None = None,
    seed: int | None = None,
    progress_callback: RunProgressCallback | None = None,
) -> TinyAdmissionRunResult:
    """Load one GONS YAML, execute all tiny-admission episodes, and persist artifacts."""

    root = repo_root or find_repo_root()
    _report_progress(
        progress_callback,
        "resolve_config",
        "Resolving GONS tiny-admission config",
    )
    config = load_tiny_admission_config(config_path, repo_root=root)
    if seed is not None:
        config = config.model_copy(
            update={"model": config.model.model_copy(update={"seed": seed})}
        )
    dataset_config = config.resolve_dataset_config(repo_root=root)

    _report_progress(progress_callback, "load_data", "Loading GONS dataset splits")
    train = load_ton_iot_csv(
        config.resolve_path(config.train_path, repo_root=root),
        dataset_config,
    ).frame
    calibration = load_ton_iot_csv(
        config.resolve_path(config.calibration_path, repo_root=root),
        dataset_config,
    ).frame
    test = load_ton_iot_csv(
        config.resolve_path(config.test_path, repo_root=root),
        dataset_config,
    ).frame
    support = load_ton_iot_csv(
        config.resolve_path(config.support_path, repo_root=root),
        dataset_config,
    ).frame
    label_column = dataset_config.label_column

    _report_progress(progress_callback, "build_episodes", "Planning GONS episodes")
    episodes = _build_episodes(
        config,
        available_labels=tuple(
            sorted(
                {
                    *train[label_column].astype(str).unique().tolist(),
                    *calibration[label_column].astype(str).unique().tolist(),
                    *test[label_column].astype(str).unique().tolist(),
                    *support[label_column].astype(str).unique().tolist(),
                }
            )
        ),
    )
    if not episodes:
        raise ValueError("The GONS runner did not produce any episodes.")

    resolved_run_id = run_id or build_run_id(config.experiment_name)
    run_dir = config.resolve_path(config.output_root, repo_root=root) / resolved_run_id
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    _append_event(
        events_path,
        event="run_start",
        payload={
            "config_path": str(config.resolve_path(config_path, repo_root=root)),
            "experiment_name": config.experiment_name,
            "method": config.method,
            "task": config.task,
        },
    )

    combined_predictions: list[pd.DataFrame] = []
    stage_records: list[dict[str, object]] = []
    episode_records: list[dict[str, object]] = []
    skipped_episodes: list[dict[str, object]] = []

    for episode in episodes:
        episode_train = train.loc[train[label_column] != episode.novel_label].reset_index(drop=True)
        episode_calibration = calibration.loc[
            calibration[label_column] != episode.novel_label
        ].reset_index(drop=True)
        episode_support = support.loc[support[label_column] == episode.novel_label].reset_index(
            drop=True
        )
        skip_reason = _episode_skip_reason(
            config=config,
            episode=episode,
            label_column=label_column,
            test=test,
            train=episode_train,
            calibration=episode_calibration,
            support=episode_support,
        )
        if skip_reason is not None:
            skipped_episodes.append(
                {
                    "episode_id": episode.episode_id,
                    "novel_label": episode.novel_label,
                    "protocol": episode.protocol,
                    "reason": skip_reason,
                }
            )
            _append_event(
                events_path,
                event="episode_skipped",
                payload=skipped_episodes[-1],
            )
            continue

        bursts = _build_support_bursts(
            support_rows=episode_support,
            burst_sizes=config.support_burst_sizes,
        )
        _append_event(
            events_path,
            event="episode_start",
            payload={
                "episode_id": episode.episode_id,
                "novel_label": episode.novel_label,
                "burst_sizes": list(config.support_burst_sizes),
                "protocol": episode.protocol,
            },
        )
        _report_progress(
            progress_callback,
            "fit",
            f"Fitting {config.method} for {episode.episode_id}",
        )
        model_state = _fit_initial_model(
            config=config,
            label_column=label_column,
            train=episode_train,
            calibration=episode_calibration,
        )
        initial_snapshot = _snapshot_state(model_state)
        initial_known_labels = initial_snapshot.known_labels

        episode_stage_records: list[dict[str, object]] = []
        episode_predictions: list[pd.DataFrame] = []
        staged_item_ids: list[str] = []
        refresh_event_count = 0
        promotion_burst_index: int | None = None
        support_rows_seen = 0
        stage_order = 0

        _report_progress(
            progress_callback,
            "predict",
            f"Scoring pre-admission state for {episode.episode_id}",
        )
        pre_predictions = _predict_dataframe_with_status(
            model_state,
            test,
            ensemble_aggregation=config.ensemble_aggregation,
        )
        pre_record = _build_stage_record(
            predictions=pre_predictions,
            episode=episode,
            initial_known_labels=initial_known_labels,
            novel_label=episode.novel_label,
            snapshot=_snapshot_state(model_state),
            stage_id="pre_admission",
            stage_kind="pre_admission",
            stage_order=stage_order,
            support_rows_seen=support_rows_seen,
            evaluation_known_labels=initial_known_labels,
        )
        episode_stage_records.append(pre_record)
        episode_predictions.append(_prepare_predictions_output(pre_predictions, pre_record))
        stage_order += 1

        for burst_index, burst in enumerate(bursts, start=1):
            batch_id = f"{episode.episode_id}-b{burst_index:03d}"
            support_rows_seen += int(len(burst))
            _report_progress(
                progress_callback,
                "update",
                f"Applying GONS burst {burst_index} for {episode.episode_id}",
            )
            burst_policy = _resolve_burst_policy(
                config=config,
                model_state=model_state,
                novel_label=episode.novel_label,
            )
            if isinstance(model_state, list):
                updated_models: list[GONSModel] = []
                burst_item_ids: list[str] = []
                for member_index, member_state in enumerate(model_state):
                    member_state, member_item_ids = _apply_bc_tiny_burst(
                        member_state,
                        burst,
                        batch_id=batch_id,
                        enable_provisional_admission=config.enable_provisional_admission,
                        t_now=float(burst_index),
                    )
                    updated_models.append(member_state)
                    if member_index == 0:
                        burst_item_ids = member_item_ids
                model_state = updated_models
                staged_item_ids.extend(burst_item_ids)
            elif isinstance(model_state, GONSModel):
                model_state, burst_item_ids = _apply_bc_tiny_burst(
                    model_state,
                    burst,
                    batch_id=batch_id,
                    enable_provisional_admission=config.enable_provisional_admission,
                    t_now=float(burst_index),
                )
                staged_item_ids.extend(burst_item_ids)
            else:
                model_state = update_baseline_model(model_state, burst)
                if promotion_burst_index is None and episode.novel_label in model_state.seen_labels:
                    promotion_burst_index = burst_index

            admitted_labels = tuple(sorted({*initial_known_labels, episode.novel_label}))
            _report_progress(
                progress_callback,
                "predict",
                f"Scoring post-admission state for {episode.episode_id} burst {burst_index}",
            )
            post_admission_predictions = _predict_dataframe_with_status(
                model_state,
                test,
                ensemble_aggregation=config.ensemble_aggregation,
            )
            post_admission_record = _build_stage_record(
                predictions=post_admission_predictions,
                episode=episode,
                initial_known_labels=initial_known_labels,
                novel_label=episode.novel_label,
                snapshot=_snapshot_state(model_state),
                stage_id=f"burst_{burst_index:03d}_post_admission",
                stage_kind="post_admission",
                stage_order=stage_order,
                support_rows_seen=support_rows_seen,
                evaluation_known_labels=admitted_labels,
            )
            episode_stage_records.append(post_admission_record)
            episode_predictions.append(
                _prepare_predictions_output(post_admission_predictions, post_admission_record)
            )
            stage_order += 1

            should_refresh = False
            if _is_bc_state(model_state) and staged_item_ids:
                should_refresh = (
                    burst_index % burst_policy.refresh_every_bursts == 0
                    or (burst_index == len(bursts) and config.run_final_refresh)
                )

            if should_refresh and _is_bc_state(model_state):
                _report_progress(
                    progress_callback,
                    "update",
                    f"Refreshing GONS state for {episode.episode_id} burst {burst_index}",
                )
                if isinstance(model_state, list):
                    model_state = [
                        consolidated_refresh(member_state, staged_item_ids, t_now=float(burst_index))
                        for member_state in model_state
                    ]
                else:
                    model_state = consolidated_refresh(
                        model_state,
                        staged_item_ids,
                        t_now=float(burst_index),
                    )
                staged_item_ids = []
                refresh_event_count += 1
                if (
                    promotion_burst_index is None
                    and _state_has_label(model_state, episode.novel_label)
                ):
                    promotion_burst_index = burst_index

                _report_progress(
                    progress_callback,
                    "predict",
                    f"Scoring post-refresh state for {episode.episode_id} burst {burst_index}",
                )
                post_refresh_predictions = _predict_dataframe_with_status(
                    model_state,
                    test,
                    ensemble_aggregation=config.ensemble_aggregation,
                )
                post_refresh_record = _build_stage_record(
                    predictions=post_refresh_predictions,
                    episode=episode,
                    initial_known_labels=initial_known_labels,
                    novel_label=episode.novel_label,
                    snapshot=_snapshot_state(model_state),
                    stage_id=f"burst_{burst_index:03d}_post_refresh",
                    stage_kind="post_refresh",
                    stage_order=stage_order,
                    support_rows_seen=support_rows_seen,
                    evaluation_known_labels=admitted_labels,
                )
                episode_stage_records.append(post_refresh_record)
                episode_predictions.append(
                    _prepare_predictions_output(post_refresh_predictions, post_refresh_record)
                )
                stage_order += 1

        episode_dir = episodes_dir / episode.episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        bursts_dir = episode_dir / "support_bursts"
        bursts_dir.mkdir(parents=True, exist_ok=True)
        for burst_index, burst in enumerate(bursts, start=1):
            burst.to_csv(bursts_dir / f"burst_{burst_index:03d}.csv", index=False)

        stages_dir = episode_dir / "stages"
        stages_dir.mkdir(parents=True, exist_ok=True)
        for stage_prediction, stage_record in zip(
            episode_predictions,
            episode_stage_records,
            strict=True,
        ):
            stage_dir = stages_dir / str(stage_record["stage_id"])
            stage_dir.mkdir(parents=True, exist_ok=True)
            stage_prediction.to_csv(stage_dir / "predictions.csv", index=False)
            _write_json(stage_dir / "metrics.json", stage_record)

        episode_summary = _build_episode_record(
            episode=episode,
            initial_snapshot=initial_snapshot,
            promotion_burst_index=promotion_burst_index,
            refresh_event_count=refresh_event_count,
            stage_records=episode_stage_records,
            support_burst_sizes=config.support_burst_sizes,
        )
        stage_frame = pd.DataFrame.from_records(episode_stage_records)
        prediction_frame = pd.concat(episode_predictions, ignore_index=True)
        stage_frame.to_csv(episode_dir / "stage_metrics.csv", index=False)
        prediction_frame.to_csv(episode_dir / "predictions.csv", index=False)
        _write_json(episode_dir / "episode_summary.json", episode_summary)
        _write_json(
            episode_dir / "episode_manifest.json",
            {
                "burst_sizes": list(config.support_burst_sizes),
                "episode_id": episode.episode_id,
                "novel_label": episode.novel_label,
                "protocol": episode.protocol,
            },
        )
        _write_json(episode_dir / "state_metadata.json", _serialize_state_metadata(model_state))

        episode_records.append(episode_summary)
        stage_records.extend(episode_stage_records)
        combined_predictions.append(prediction_frame)
        _append_event(
            events_path,
            event="episode_complete",
            payload={
                "episode_id": episode.episode_id,
                "summary": episode_summary,
            },
        )

    if not episode_records:
        raise ValueError(
            "The GONS runner skipped every episode. "
            "Adjust the holdout groups, support bursts, or fixture data."
        )

    _report_progress(
        progress_callback,
        "write_artifacts",
        "Writing GONS run artifacts",
    )
    episode_metrics_frame = pd.DataFrame.from_records(episode_records)
    stage_metrics_frame = pd.DataFrame.from_records(stage_records)
    combined_predictions_frame = pd.concat(combined_predictions, ignore_index=True)
    aggregate_metrics = _aggregate_metrics(
        episode_metrics=episode_metrics_frame,
        planned_episode_count=len(episodes),
        skipped_episode_count=len(skipped_episodes),
    )
    resolved_config = config.model_copy(
        update={"model": config.model.model_copy(update={"label_column": label_column})}
    )
    _write_json(run_dir / "config_resolved.json", resolved_config.model_dump(mode="json"))
    _write_json(
        run_dir / "run_manifest.json",
        {
            "config_path": str(config.resolve_path(config_path, repo_root=root)),
            "experiment_name": config.experiment_name,
            "git_commit": current_git_commit(root),
            "method": config.method,
            "python_version": python_runtime(),
            "run_id": resolved_run_id,
            "seed": config.model.seed,
            "task": config.task,
            "timestamp": utc_now_iso(),
        },
    )
    combined_predictions_frame.to_csv(run_dir / "predictions.csv", index=False)
    stage_metrics_frame.to_csv(run_dir / "stage_metrics.csv", index=False)
    episode_metrics_frame.to_csv(run_dir / "episode_metrics.csv", index=False)
    _write_json(run_dir / "episodes.json", episode_records)
    _write_json(run_dir / "metrics.json", aggregate_metrics)
    _write_json(run_dir / "summary_overview.json", aggregate_metrics)
    summary = _render_summary(
        aggregate_metrics=aggregate_metrics,
        episode_metrics=episode_metrics_frame,
        method=config.method,
        run_id=resolved_run_id,
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    (run_dir / "summary_overview.md").write_text(summary, encoding="utf-8")
    _append_event(
        events_path,
        event="run_complete",
        payload={
            "episode_count": len(episode_records),
            "method": config.method,
            "run_dir": str(run_dir),
        },
    )
    _report_progress(progress_callback, "complete", "GONS experiment completed")
    return TinyAdmissionRunResult(
        method=config.method,
        metrics=aggregate_metrics,
        run_dir=run_dir,
        run_id=resolved_run_id,
    )


def _build_episodes(
    config: TinyAdmissionExperimentConfig,
    *,
    available_labels: tuple[str, ...],
) -> list[Episode]:
    label_set = set(available_labels)
    groups: list[tuple[str, ...]] = []
    if config.holdout_groups:
        groups.extend(tuple(group) for group in config.holdout_groups)
    elif config.include_leave_one_class_out:
        groups.extend((label,) for label in available_labels)

    episodes: list[Episode] = []
    seen_groups: set[tuple[str, ...]] = set()
    for group in groups:
        normalized_group = tuple(sorted(group))
        if normalized_group in seen_groups:
            continue
        missing = [label for label in normalized_group if label not in label_set]
        if missing:
            raise ValueError(
                "The GONS config references holdout labels that do not exist in the "
                f"dataset: {missing}"
            )
        seen_groups.add(normalized_group)
        episodes.append(
            Episode(
                episode_id=f"track-a-{'-'.join(normalized_group)}",
                novel_label=normalized_group[0],
                protocol="single_tiny_novel_class",
            )
        )
        if config.max_episodes is not None and len(episodes) >= config.max_episodes:
            break
    return episodes


def _episode_skip_reason(
    *,
    config: TinyAdmissionExperimentConfig,
    episode: Episode,
    label_column: str,
    test: pd.DataFrame,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    support: pd.DataFrame,
) -> str | None:
    known_labels = train[label_column].astype(str).unique().tolist()
    if len(known_labels) < config.minimum_known_classes:
        return (
            "Holdout leaves fewer than the configured minimum known classes: "
            f"{len(known_labels)} < {config.minimum_known_classes}."
        )
    if train.empty:
        return "Holdout leaves no training rows."
    if calibration.empty:
        return "Holdout leaves no calibration rows."
    if not test[label_column].isin([episode.novel_label]).any():
        return "Novel label is absent from the evaluation split."
    if len(support) < sum(int(size) for size in config.support_burst_sizes):
        return (
            "Support split does not contain enough rows for the configured GONS bursts: "
            f"{len(support)} < {sum(int(size) for size in config.support_burst_sizes)}."
        )
    return None


def _build_support_bursts(
    *,
    support_rows: pd.DataFrame,
    burst_sizes: list[int],
) -> list[pd.DataFrame]:
    bursts: list[pd.DataFrame] = []
    offset = 0
    for burst_size in burst_sizes:
        next_offset = offset + int(burst_size)
        bursts.append(support_rows.iloc[offset:next_offset].copy().reset_index(drop=True))
        offset = next_offset
    return bursts


def _fit_initial_model(
    *,
    config: TinyAdmissionExperimentConfig,
    label_column: str,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
) -> ModelState:
    if config.method == "gons":
        if int(config.ensemble_size) > 1:
            models: list[GONSModel] = []
            base_seed = int(config.model.seed)
            seed_offset = int(config.ensemble_seed_offset)
            for member_index in range(int(config.ensemble_size)):
                member_seed = base_seed + member_index * seed_offset
                member_config = config.model.model_copy(
                    update={"label_column": label_column, "seed": member_seed}
                )
                models.append(fit_gons(train, calibration, member_config))
            return models
        model_config = config.model.model_copy(update={"label_column": label_column})
        return fit_gons(train, calibration, model_config)
    baseline_config = config.to_baseline_fit_config()
    return fit_baseline(
        train,
        calibration,
        config=baseline_config,
        label_column=label_column,
    )


def _is_bc_state(model_state: ModelState) -> bool:
    return isinstance(model_state, GONSModel) or isinstance(model_state, list)


def _state_has_label(model_state: ModelState, label: str) -> bool:
    if isinstance(model_state, list):
        return any(label in member_state.class_registry for member_state in model_state)
    if isinstance(model_state, GONSModel):
        return label in model_state.class_registry
    return label in model_state.seen_labels


def _snapshot_state(model_state: ModelState) -> EvaluationSnapshot:
    if isinstance(model_state, list):
        member_snapshots = [_snapshot_state(member_state) for member_state in model_state]
        known_labels = tuple(
            sorted(
                {
                    label
                    for snapshot in member_snapshots
                    for label in snapshot.known_labels
                }
            )
        )
        return EvaluationSnapshot(
            calibration_mode="ensemble",
            known_labels=known_labels,
            projection_dim=member_snapshots[0].projection_dim,
            projection_mode="ensemble",
            projection_version=max(snapshot.projection_version for snapshot in member_snapshots),
            prototype_count=int(sum(snapshot.prototype_count for snapshot in member_snapshots)),
            tau_global=float(np.mean([snapshot.tau_global for snapshot in member_snapshots])),
        )
    if isinstance(model_state, GONSModel):
        return EvaluationSnapshot(
            calibration_mode=model_state.calibration_state.mode or "insufficient",
            known_labels=tuple(sorted(model_state.class_registry)),
            projection_dim=0 if model_state.W_proj is None else int(model_state.W_proj.shape[1]),
            projection_mode=(
                "unset"
                if model_state.projection_meta is None
                else model_state.projection_meta.mode
            ),
            projection_version=int(model_state.projection_version),
            prototype_count=int(len(model_state.proto_bank)),
            tau_global=float(model_state.tau_global),
        )
    return EvaluationSnapshot(
        calibration_mode=model_state.calibration_state.mode or "insufficient",
        known_labels=tuple(sorted(model_state.seen_labels)),
        projection_dim=0 if model_state.W_proj is None else int(model_state.W_proj.shape[1]),
        projection_mode=(
            "unset"
            if model_state.projection_meta is None
            else model_state.projection_meta.mode
        ),
        projection_version=0,
        prototype_count=int(len(model_state.proto_bank)),
        tau_global=float(model_state.tau_global),
    )


def _predict_dataframe_with_status(
    model_state: ModelState,
    frame: pd.DataFrame,
    *,
    ensemble_aggregation: str,
) -> pd.DataFrame:
    if isinstance(model_state, list):
        return _predict_bc_ensemble_dataframe(
            model_state,
            frame,
            aggregation=ensemble_aggregation,
        )
    if isinstance(model_state, GONSModel):
        predictions = predict_gons_dataframe(model_state, frame)
        predictions["best_prototype_status"] = predictions["best_prototype_id"].map(
            lambda proto_id: model_state.proto_bank[str(proto_id)].status
        )
        return predictions
    predictions = predict_dataframe(model_state, frame)
    predictions["projection_version"] = 0
    predictions["best_prototype_status"] = predictions["best_prototype_id"].map(
        lambda proto_id: model_state.proto_bank[str(proto_id)].status
        if str(proto_id) in model_state.proto_bank
        else "unknown"
    )
    return predictions


def _predict_bc_ensemble_dataframe(
    models: list[GONSModel],
    frame: pd.DataFrame,
    *,
    aggregation: str,
) -> pd.DataFrame:
    """Aggregate predictions across independent GONS members."""

    member_predictions = [predict_gons_dataframe(model, frame) for model in models]
    if not member_predictions:
        return pd.DataFrame()

    score_matrix = np.column_stack(
        [prediction["score"].to_numpy(dtype=np.float64) for prediction in member_predictions]
    )
    threshold_matrix = np.column_stack(
        [prediction["threshold"].to_numpy(dtype=np.float64) for prediction in member_predictions]
    )
    accepted_matrix = np.column_stack(
        [prediction["accepted"].to_numpy(dtype=bool) for prediction in member_predictions]
    )

    if aggregation == "average":
        final_scores = np.mean(score_matrix, axis=1)
        final_thresholds = np.mean(threshold_matrix, axis=1)
        final_accepted = final_scores <= final_thresholds
    elif aggregation == "max":
        final_scores = np.max(score_matrix, axis=1)
        final_thresholds = np.mean(threshold_matrix, axis=1)
        final_accepted = final_scores <= final_thresholds
    elif aggregation == "majority_vote":
        reject_votes = np.sum(~accepted_matrix, axis=1)
        final_scores = reject_votes.astype(np.float64)
        final_thresholds = np.full(len(frame), len(models) // 2, dtype=np.float64)
        final_accepted = reject_votes <= (len(models) // 2)
    else:
        raise ValueError(f"Unsupported GONS ensemble aggregation: {aggregation}")

    true_labels = member_predictions[0]["true_label"].tolist()
    projection_versions = np.column_stack(
        [
            prediction["projection_version"].to_numpy(dtype=np.int64)
            for prediction in member_predictions
        ]
    )

    records: list[dict[str, object]] = []
    for row_index in range(len(frame)):
        row_predictions = [prediction.iloc[row_index] for prediction in member_predictions]
        indexed_row_predictions = list(enumerate(row_predictions))
        accepted_indices = [
            member_index
            for member_index, row_prediction in indexed_row_predictions
            if bool(row_prediction["accepted"])
        ]
        candidate_indices = (
            accepted_indices if final_accepted[row_index] and accepted_indices else list(range(len(row_predictions)))
        )
        candidate_rows = [row_predictions[member_index] for member_index in candidate_indices]
        label_votes = Counter(
            str(row_prediction["predicted_label"])
            for row_prediction in candidate_rows
            if str(row_prediction["predicted_label"]) != "unknown"
        )
        if final_accepted[row_index] and label_votes:
            predicted_label = label_votes.most_common(1)[0][0]
        else:
            predicted_label = "unknown"

        if predicted_label == "unknown":
            representative_index, representative = min(
                indexed_row_predictions,
                key=lambda item: float(item[1]["score"]),
            )
        else:
            representative_index, representative = next(
                (
                    (member_index, row_prediction)
                    for member_index, row_prediction in indexed_row_predictions
                    if member_index in candidate_indices
                    and str(row_prediction["predicted_label"]) == predicted_label
                ),
                (candidate_indices[0], row_predictions[candidate_indices[0]]),
            )

        member_model = models[representative_index]
        proto_id = str(representative["best_prototype_id"])
        records.append(
            {
                "row_index": row_index,
                "true_label": true_labels[row_index],
                "predicted_label": predicted_label,
                "accepted": bool(final_accepted[row_index]),
                "best_prototype_id": proto_id,
                "best_class_label": str(representative["best_class_label"]),
                "best_prototype_status": member_model.proto_bank[proto_id].status,
                "score": float(final_scores[row_index]),
                "threshold": float(final_thresholds[row_index]),
                "projection_version": int(np.max(projection_versions[row_index])),
            }
        )
    return pd.DataFrame.from_records(records)


def _apply_bc_tiny_burst(
    model_state: GONSModel,
    burst: pd.DataFrame,
    *,
    batch_id: str,
    enable_provisional_admission: bool,
    t_now: float,
) -> tuple[GONSModel, list[str]]:
    label_column = model_state.config.label_column
    item_ids: list[str] = []
    for class_label, class_frame in burst.groupby(label_column, sort=True):
        normalized_class = str(class_label)
        model_state = cache_labeled_batch_by_class(
            model_state,
            class_frame.reset_index(drop=True),
            batch_id=batch_id,
            pathway="singleton" if len(class_frame) == 1 else "tiny",
            t_now=t_now,
        )
        item_id = f"{batch_id}:{normalized_class}"
        item_ids.append(item_id)
        if enable_provisional_admission:
            model_state = provisional_admit_small_item(model_state, item_id, t_now=t_now)
    model_state.pending_batch_manifest[batch_id] = tuple(item_ids)
    return model_state, item_ids


def _resolve_burst_policy(
    *,
    config: TinyAdmissionExperimentConfig,
    model_state: ModelState,
    novel_label: str,
) -> _BurstPolicy:
    """Apply post-promotion GONS overrides and return the effective burst policy."""

    refresh_every_bursts = int(config.refresh_every_bursts)
    if isinstance(model_state, list):
        promoted = _state_has_label(model_state, novel_label)
        for member_state in model_state:
            _apply_policy_overrides(
                config=config,
                model_state=member_state,
                promoted=promoted,
            )
        if promoted and config.post_promotion_refresh_every_bursts is not None:
            refresh_every_bursts = int(config.post_promotion_refresh_every_bursts)
        return _BurstPolicy(refresh_every_bursts=refresh_every_bursts)
    if not isinstance(model_state, GONSModel):
        return _BurstPolicy(refresh_every_bursts=refresh_every_bursts)

    promoted = novel_label in model_state.class_registry
    _apply_policy_overrides(
        config=config,
        model_state=model_state,
        promoted=promoted,
    )
    if promoted and config.post_promotion_refresh_every_bursts is not None:
        refresh_every_bursts = int(config.post_promotion_refresh_every_bursts)

    return _BurstPolicy(refresh_every_bursts=refresh_every_bursts)


def _apply_policy_overrides(
    *,
    config: TinyAdmissionExperimentConfig,
    model_state: GONSModel,
    promoted: bool,
) -> None:
    model_state.config.enable_local_affine_screen = config.model.enable_local_affine_screen
    model_state.config.refresh_mode = config.model.refresh_mode
    model_state.config.support_trim_strategy = config.model.support_trim_strategy

    if not promoted:
        return
    if config.post_promotion_enable_local_affine_screen:
        model_state.config.enable_local_affine_screen = True
    if config.post_promotion_refresh_mode is not None:
        model_state.config.refresh_mode = config.post_promotion_refresh_mode
    if config.post_promotion_support_trim_strategy is not None:
        model_state.config.support_trim_strategy = config.post_promotion_support_trim_strategy


def _build_stage_record(
    *,
    predictions: pd.DataFrame,
    episode: Episode,
    initial_known_labels: tuple[str, ...],
    novel_label: str,
    snapshot: EvaluationSnapshot,
    stage_id: str,
    stage_kind: Literal["pre_admission", "post_admission", "post_refresh"],
    stage_order: int,
    support_rows_seen: int,
    evaluation_known_labels: tuple[str, ...],
) -> dict[str, object]:
    record: dict[str, object] = {
        "episode_id": episode.episode_id,
        "novel_label": novel_label,
        "protocol": episode.protocol,
        "stage_id": stage_id,
        "stage_kind": stage_kind,
        "stage_order": stage_order,
        "support_rows_seen": int(support_rows_seen),
        "calibration_mode": snapshot.calibration_mode,
        "projection_dim": snapshot.projection_dim,
        "projection_mode": snapshot.projection_mode,
        "projection_version": snapshot.projection_version,
        "prototype_count": snapshot.prototype_count,
        "tau_global": snapshot.tau_global,
        "active_class_count": len(snapshot.known_labels),
        "is_promoted_to_canonical": novel_label in snapshot.known_labels,
    }
    record.update(
        summarize_open_set_predictions(
            predictions,
            known_labels=evaluation_known_labels,
        )
    )

    true_labels = predictions["true_label"].astype(str)
    predicted_labels = predictions["predicted_label"].astype(str)
    accepted_mask = predictions["accepted"].astype(bool).to_numpy(dtype=bool)
    best_status = predictions["best_prototype_status"].astype(str)
    correct_label_mask = (predicted_labels == true_labels).to_numpy(dtype=bool)
    reference_known_mask = true_labels.isin(initial_known_labels).to_numpy(dtype=bool)
    novel_mask = (true_labels == novel_label).to_numpy(dtype=bool)
    predicted_novel_mask = (predicted_labels == novel_label).to_numpy(dtype=bool)
    provisional_mask = best_status.isin({"provisional", "singleton_provisional"}).to_numpy(
        dtype=bool
    )
    fragile_mask = (best_status == "fragile_stable").to_numpy(dtype=bool)

    record["served_by_provisional_count"] = int(np.sum(accepted_mask & provisional_mask))
    record["served_by_fragile_count"] = int(np.sum(accepted_mask & fragile_mask))
    record["provisional_prediction_rate"] = float(np.mean(accepted_mask & provisional_mask))
    record["fragile_prediction_rate"] = float(np.mean(accepted_mask & fragile_mask))
    record["admitted_known_label_count"] = len(evaluation_known_labels)

    if np.any(reference_known_mask):
        record["known_reference_accuracy"] = float(
            np.mean(correct_label_mask[reference_known_mask])
        )
        record["known_reference_macro_f1"] = float(
            f1_score(
                true_labels[reference_known_mask],
                predicted_labels[reference_known_mask],
                labels=sorted(initial_known_labels),
                average="macro",
                zero_division=0,
            )
        )
    if np.any(novel_mask):
        record["novel_test_rows"] = int(np.sum(novel_mask))
        record["new_class_recall"] = float(np.mean(predicted_novel_mask[novel_mask]))
    if np.any(predicted_novel_mask):
        record["predicted_novel_rows"] = int(np.sum(predicted_novel_mask))
        record["new_class_precision"] = float(
            np.mean((true_labels[predicted_novel_mask] == novel_label).to_numpy(dtype=bool))
        )
    if np.any(novel_mask) or np.any(predicted_novel_mask):
        record["new_class_f1"] = float(
            f1_score(
                (true_labels == novel_label).to_numpy(dtype=np.int64),
                predicted_novel_mask.astype(np.int64),
                zero_division=0,
            )
        )
    return record


def _prepare_predictions_output(
    predictions: pd.DataFrame,
    stage_record: dict[str, object],
) -> pd.DataFrame:
    output = predictions.copy()
    output.insert(0, "episode_id", str(stage_record["episode_id"]))
    output.insert(1, "stage_id", str(stage_record["stage_id"]))
    output.insert(2, "stage_kind", str(stage_record["stage_kind"]))
    output.insert(3, "stage_order", cast(int, stage_record["stage_order"]))
    output.insert(4, "novel_label", str(stage_record["novel_label"]))
    output.insert(5, "support_rows_seen", cast(int, stage_record["support_rows_seen"]))
    return output


def _build_episode_record(
    *,
    episode: Episode,
    initial_snapshot: EvaluationSnapshot,
    promotion_burst_index: int | None,
    refresh_event_count: int,
    stage_records: list[dict[str, object]],
    support_burst_sizes: list[int],
) -> dict[str, object]:
    pre_stage = next(
        record for record in stage_records if str(record["stage_kind"]) == "pre_admission"
    )
    final_stage = stage_records[-1]
    refresh_stages = [
        record for record in stage_records if str(record["stage_kind"]) == "post_refresh"
    ]
    provisional_stages = [
        record for record in stage_records if str(record["stage_kind"]) == "post_admission"
    ]
    known_accuracy_damage = None
    if (
        isinstance(pre_stage.get("known_reference_accuracy"), (int, float))
        and isinstance(final_stage.get("known_reference_accuracy"), (int, float))
    ):
        final_known_accuracy = cast(float | int, final_stage["known_reference_accuracy"])
        pre_known_accuracy = cast(float | int, pre_stage["known_reference_accuracy"])
        known_accuracy_damage = float(final_known_accuracy) - float(
            pre_known_accuracy
        )
    record: dict[str, object] = {
        "burst_count": len(support_burst_sizes),
        "episode_id": episode.episode_id,
        "final_active_class_count": cast(int, final_stage["active_class_count"]),
        "final_false_known_rate": final_stage.get("false_known_rate"),
        "final_known_reference_accuracy": final_stage.get("known_reference_accuracy"),
        "final_new_class_f1": final_stage.get("new_class_f1"),
        "final_new_class_precision": final_stage.get("new_class_precision"),
        "final_new_class_recall": final_stage.get("new_class_recall"),
        "initial_active_class_count": len(initial_snapshot.known_labels),
        "initial_known_reference_accuracy": pre_stage.get("known_reference_accuracy"),
        "known_class_accuracy_damage": known_accuracy_damage,
        "max_provisional_prediction_rate": _max_metric(
            provisional_stages,
            "provisional_prediction_rate",
        ),
        "mean_provisional_prediction_rate": _mean_metric(
            provisional_stages,
            "provisional_prediction_rate",
        ),
        "novel_label": episode.novel_label,
        "post_refresh_new_class_f1": _last_metric(refresh_stages, "new_class_f1"),
        "promoted_to_canonical": promotion_burst_index is not None,
        "promotion_burst_index": promotion_burst_index,
        "protocol": episode.protocol,
        "refresh_event_count": refresh_event_count,
        "stage_count": len(stage_records),
        "support_burst_sizes": list(support_burst_sizes),
        "support_rows_total": int(sum(support_burst_sizes)),
    }
    return record


def _aggregate_metrics(
    *,
    episode_metrics: pd.DataFrame,
    planned_episode_count: int,
    skipped_episode_count: int,
) -> dict[str, object]:
    aggregate: dict[str, object] = {
        "episode_count": int(len(episode_metrics)),
        "executed_episode_count": int(len(episode_metrics)),
        "planned_episode_count": int(planned_episode_count),
        "skipped_episode_count": int(skipped_episode_count),
    }
    if episode_metrics.empty:
        return aggregate

    mean_columns = (
        "final_false_known_rate",
        "final_known_reference_accuracy",
        "final_new_class_f1",
        "final_new_class_precision",
        "final_new_class_recall",
        "known_class_accuracy_damage",
        "mean_provisional_prediction_rate",
        "post_refresh_new_class_f1",
    )
    for column in mean_columns:
        values = pd.to_numeric(episode_metrics[column], errors="coerce").dropna()
        if not values.empty:
            aggregate[f"mean_{column}"] = float(values.mean())

    promotion_values = pd.to_numeric(
        episode_metrics["promotion_burst_index"],
        errors="coerce",
    ).dropna()
    if not promotion_values.empty:
        aggregate["mean_promotion_burst_index"] = float(promotion_values.mean())
    aggregate["promoted_episode_count"] = int(
        pd.to_numeric(
            episode_metrics["promoted_to_canonical"].astype(int),
            errors="coerce",
        ).fillna(0).sum()
    )
    aggregate["total_refresh_event_count"] = int(
        pd.to_numeric(episode_metrics["refresh_event_count"], errors="coerce").fillna(0).sum()
    )
    aggregate["total_support_rows"] = int(
        pd.to_numeric(episode_metrics["support_rows_total"], errors="coerce").fillna(0).sum()
    )
    return aggregate


def _serialize_state_metadata(model_state: ModelState) -> dict[str, object]:
    if isinstance(model_state, list):
        return {
            "calibration_mode": "ensemble",
            "ensemble_size": len(model_state),
            "known_labels": sorted(
                {
                    label
                    for member_state in model_state
                    for label in member_state.class_registry
                }
            ),
            "members": [_serialize_state_metadata(member_state) for member_state in model_state],
            "projection_meta": {},
            "projection_version": max(
                (int(member_state.projection_version) for member_state in model_state),
                default=0,
            ),
            "threshold_meta": {},
            "unknown_buffer_size": int(
                sum(len(member_state.unknown_buffer) for member_state in model_state)
            ),
        }
    if isinstance(model_state, GONSModel):
        return {
            "calibration_mode": model_state.calibration_state.mode or "insufficient",
            "known_labels": sorted(model_state.class_registry),
            "projection_meta": (
                {}
                if model_state.projection_meta is None
                else serialize_projection_meta(model_state.projection_meta)
            ),
            "projection_version": int(model_state.projection_version),
            "threshold_meta": (
                {}
                if model_state.threshold_meta is None
                else serialize_threshold_meta(model_state.threshold_meta)
            ),
            "unknown_buffer_size": len(model_state.unknown_buffer),
        }
    return {
        "calibration_mode": model_state.calibration_state.mode or "insufficient",
        "known_labels": sorted(model_state.seen_labels),
        "projection_meta": (
            {}
            if model_state.projection_meta is None
            else serialize_projection_meta(model_state.projection_meta)
        ),
        "projection_version": 0,
        "threshold_meta": (
            {}
            if model_state.threshold_meta is None
            else serialize_threshold_meta(model_state.threshold_meta)
        ),
        "unknown_buffer_size": 0,
    }


def _mean_metric(records: list[dict[str, object]], key: str) -> float | None:
    values = [
        float(cast(float | int, record[key]))
        for record in records
        if isinstance(record.get(key), (int, float))
    ]
    if not values:
        return None
    return float(np.mean(values))


def _max_metric(records: list[dict[str, object]], key: str) -> float | None:
    values = [
        float(cast(float | int, record[key]))
        for record in records
        if isinstance(record.get(key), (int, float))
    ]
    if not values:
        return None
    return float(max(values))


def _last_metric(records: list[dict[str, object]], key: str) -> object:
    if not records:
        return None
    return records[-1].get(key)


def _append_event(
    path: Path,
    *,
    event: str,
    payload: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": event,
        "payload": payload,
        "timestamp": utc_now_iso(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable.")


def _render_summary(
    *,
    aggregate_metrics: dict[str, object],
    episode_metrics: pd.DataFrame,
    method: str,
    run_id: str,
) -> str:
    lines = [
        "# GONS Tiny Admission Summary",
        "",
        f"- Method: `{method}`",
        f"- Run ID: `{run_id}`",
        f"- Episodes: `{aggregate_metrics['episode_count']}`",
        f"- Promoted Episodes: `{aggregate_metrics.get('promoted_episode_count', 0)}`",
        "",
        "## Aggregate Metrics",
        "",
    ]
    for key, value in sorted(aggregate_metrics.items()):
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Episodes",
            "",
            (
                "| Episode | Novel Label | Bursts | Promoted | Final New F1 | "
                "Known Damage | Mean Provisional Rate |"
            ),
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in episode_metrics.sort_values("episode_id").iterrows():
        lines.append(
            "| "
            f"{row['episode_id']} | "
            f"{row['novel_label']} | "
            f"{row['burst_count']} | "
            f"{int(bool(row['promoted_to_canonical']))} | "
            f"{_fmt_metric(row.get('final_new_class_f1'))} | "
            f"{_fmt_metric(row.get('known_class_accuracy_damage'))} | "
            f"{_fmt_metric(row.get('mean_provisional_prediction_rate'))} |"
        )
    return "\n".join(lines) + "\n"


def _fmt_metric(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _report_progress(
    progress_callback: RunProgressCallback | None,
    stage: str,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, message)


__all__ = [
    "TinyAdmissionRunResult",
    "run_tiny_admission_experiment",
]
