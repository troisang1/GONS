"""GONS FSCIL runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from bcmrnfst.baselines import update_baseline_model
from bcmrnfst.continual import consolidated_refresh
from bcmrnfst.data.ton_iot import load_ton_iot_csv
from bcmrnfst.evaluation.config import TrackAFscilExperimentConfig, load_track_a_fscil_config
from bcmrnfst.preprocess import transform_rows
from bcmrnfst.runtime import (
    build_run_id,
    current_git_commit,
    find_repo_root,
    python_runtime,
    utc_now_iso,
)
from bcmrnfst.runners.track_a_tiny_admission import (
    TrackAModelState,
    _append_event,
    _apply_bc_tiny_burst,
    _fit_initial_model as _fit_track_a_initial_model,
    _predict_dataframe_with_status,
    _report_progress,
    _serialize_state_metadata,
    _snapshot_state,
    _write_json,
)
from bcmrnfst.state.model import BCMRNFSTModel


@dataclass(frozen=True, slots=True)
class FscilSession:
    """One FSCIL session."""

    session_id: int
    session_name: str
    novel_label: str | None
    is_base: bool


@dataclass(frozen=True, slots=True)
class TrackAFscilRunResult:
    """Compact result returned by the FSCIL runner."""

    method: str
    metrics: dict[str, object]
    run_dir: Path
    run_id: str


def run_track_a_fscil_experiment(
    config_path: Path,
    *,
    run_id: str | None = None,
    repo_root: Path | None = None,
    seed: int | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
) -> TrackAFscilRunResult:
    """Execute the GONS FSCIL protocol."""

    root = repo_root or find_repo_root()
    _report_progress(progress_callback, "resolve_config", "Resolving GONS FSCIL config")
    config = load_track_a_fscil_config(config_path, repo_root=root)
    if seed is not None:
        config = config.model_copy(
            update={"model": config.model.model_copy(update={"seed": seed})}
        )
    dataset_config = config.resolve_dataset_config(repo_root=root)
    label_column = dataset_config.label_column

    _report_progress(progress_callback, "load_data", "Loading GONS FSCIL dataset splits")
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

    _report_progress(progress_callback, "build_sessions", "Planning FSCIL sessions")
    available_labels = tuple(
        sorted(
            {
                *train[label_column].astype(str).unique().tolist(),
                *calibration[label_column].astype(str).unique().tolist(),
                *test[label_column].astype(str).unique().tolist(),
            }
        )
    )
    base_classes, novel_classes, sessions = _plan_fscil_sessions(
        config,
        available_labels=available_labels,
    )
    _validate_fscil_labels(
        train=train,
        calibration=calibration,
        label_column=label_column,
        base_classes=base_classes,
        novel_classes=novel_classes,
    )

    # Binary collapse: relabel all non-benign to 'attack' for training/inference
    # while keeping the original multiclass session schedule.
    # The session plan still uses original labels to decide WHICH attacks arrive
    # when, but the model only sees {benign, attack}.
    binary_collapse = getattr(config, "binary_collapse", False)
    _benign_labels: set[str] = set()
    if binary_collapse:
        _benign_labels = set(getattr(config, "binary_benign_labels", ["benign", "normal", "benigntraffic"]))
        # Keep original labels for row selection, add collapsed column
        train = train.copy()
        calibration = calibration.copy()
        test = test.copy()
        train["_orig_label"] = train[label_column].astype(str)
        calibration["_orig_label"] = calibration[label_column].astype(str)
        test["_orig_label"] = test[label_column].astype(str)
        for frame in [train, calibration, test]:
            frame[label_column] = frame[label_column].apply(
                lambda x: "benign" if str(x) in _benign_labels else "attack"
            )

    if binary_collapse:
        # Base train: rows whose ORIGINAL label is in base_classes
        base_train = train.loc[train["_orig_label"].isin(base_classes)].reset_index(drop=True)
        base_calibration = calibration.loc[calibration["_orig_label"].isin(base_classes)].reset_index(drop=True)
    else:
        base_train = train.loc[train[label_column].isin(base_classes)].reset_index(drop=True)
        base_calibration = calibration.loc[calibration[label_column].isin(base_classes)].reset_index(drop=True)

    resolved_run_id = run_id or build_run_id(config.experiment_name)
    run_dir = config.resolve_path(config.output_root, repo_root=root) / resolved_run_id
    sessions_dir = run_dir / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "events.jsonl"
    _append_event(
        events_path,
        event="run_start",
        payload={
            "base_class_ratio": config.base_class_ratio,
            "base_class_seed": config.base_class_seed,
            "base_classes": base_classes,
            "config_path": str(config.resolve_path(config_path, repo_root=root)),
            "experiment_name": config.experiment_name,
            "method": config.method,
            "novel_class_order": config.novel_class_order,
            "novel_classes": novel_classes,
            "novel_support_fraction": config.novel_support_fraction,
            "task": config.task,
        },
    )

    combined_predictions: list[pd.DataFrame] = []
    session_records: list[dict[str, object]] = []
    model_state: TrackAModelState | None = None

    for session in sessions:
        session_dir = sessions_dir / session.session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        support_rows = 0
        refresh_applied = False

        if session.is_base:
            _report_progress(
                progress_callback,
                "fit",
                f"Fitting FSCIL base session on {len(base_classes)} classes",
            )
            model_state = _fit_track_a_initial_model(
                config=cast(Any, config),
                label_column=label_column,
                train=base_train,
                calibration=base_calibration,
            )
        else:
            if model_state is None:
                raise ValueError("FSCIL model state was not initialized before novel admission.")
            if session.novel_label is None:
                raise ValueError("Novel FSCIL sessions must declare a novel label.")

            _report_progress(
                progress_callback,
                "update",
                f"Applying FSCIL support for {session.novel_label}",
            )
            # For binary collapse, select support by ORIGINAL label but feed
            # collapsed labels to the model.
            _support_label_col = "_orig_label" if binary_collapse else label_column
            _support_novel_label = session.novel_label
            support = _sample_novel_support(
                df_all=train,
                novel_label=_support_novel_label,
                label_column=_support_label_col,
                fraction=config.novel_support_fraction,
                seed=config.base_class_seed + session.session_id,
                max_k=getattr(config, "novel_support_max_k", None),
            )
            support_rows = int(len(support))
            support.to_csv(session_dir / "support.csv", index=False)
            model_state, item_ids = _apply_fscil_support(
                model_state=model_state,
                support=support,
                session=session,
                config=config,
            )
            model_state = _remember_refresh_validation_support(
                model_state=model_state,
                support=support,
            )

            if config.refresh_after_session and _is_bc_fscil_state(model_state) and item_ids:
                _report_progress(
                    progress_callback,
                    "update",
                    f"Refreshing FSCIL state after {session.session_name}",
                )
                # Force-accept the novel class so the affine conflict review
                # does not block it.  In FSCIL the protocol guarantees that
                # the support is correctly labelled.
                force_accept = {session.novel_label} if session.novel_label else set()
                if isinstance(model_state, list):
                    model_state = [
                        consolidated_refresh(
                            member_state,
                            item_ids,
                            t_now=float(session.session_id),
                            force_accept_classes=force_accept,
                        )
                        for member_state in model_state
                    ]
                else:
                    model_state = consolidated_refresh(
                        model_state,
                        item_ids,
                        t_now=float(session.session_id),
                        force_accept_classes=force_accept,
                    )
                refresh_applied = True

        if model_state is None:
            raise ValueError("FSCIL model state was not initialized.")

        _report_progress(
            progress_callback,
            "predict",
            f"Scoring {session.session_name} on the full evaluation split",
        )
        predictions = _predict_dataframe_with_status(
            model_state,
            test,
            ensemble_aggregation=config.ensemble_aggregation,
        )
        snapshot = _snapshot_state(model_state)
        session_record = _build_fscil_session_record(
            session=session,
            predictions=predictions,
            test=test,
            label_column=label_column,
            base_classes=base_classes,
            novel_classes_so_far=novel_classes[: session.session_id],
            known_labels=snapshot.known_labels,
            support_rows=support_rows,
            refresh_applied=refresh_applied,
        )
        session_prediction_output = _prepare_session_predictions_output(
            predictions,
            session_record=session_record,
        )
        session_prediction_output.to_csv(session_dir / "predictions.csv", index=False)
        _write_json(session_dir / "session_metrics.json", session_record)
        _write_json(session_dir / "state_metadata.json", _serialize_state_metadata(model_state))
        combined_predictions.append(session_prediction_output)
        session_records.append(session_record)
        _append_event(
            events_path,
            event="session_complete",
            payload={
                "session_name": session.session_name,
                "session_summary": session_record,
            },
        )

    if not session_records:
        raise ValueError("The FSCIL runner did not produce any session records.")

    _report_progress(progress_callback, "write_artifacts", "Writing GONS FSCIL artifacts")
    session_metrics_frame = pd.DataFrame.from_records(session_records)
    combined_predictions_frame = pd.concat(combined_predictions, ignore_index=True)
    run_summary = _build_fscil_run_summary(
        session_records=session_records,
        base_classes=base_classes,
        novel_classes=novel_classes,
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
    session_metrics_frame.to_csv(run_dir / "session_metrics.csv", index=False)
    _write_json(run_dir / "sessions.json", session_records)
    _write_json(run_dir / "run_summary.json", run_summary)
    _write_json(run_dir / "metrics.json", run_summary)
    _write_json(run_dir / "summary_overview.json", run_summary)
    summary = _render_summary(
        method=config.method,
        run_id=resolved_run_id,
        run_summary=run_summary,
        session_metrics=session_metrics_frame,
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    (run_dir / "summary_overview.md").write_text(summary, encoding="utf-8")
    _append_event(
        events_path,
        event="run_complete",
        payload={
            "method": config.method,
            "run_dir": str(run_dir),
            "session_count": len(session_records),
        },
    )
    _report_progress(progress_callback, "complete", "GONS FSCIL experiment completed")
    return TrackAFscilRunResult(
        method=config.method,
        metrics=run_summary,
        run_dir=run_dir,
        run_id=resolved_run_id,
    )


def _plan_fscil_sessions(
    config: TrackAFscilExperimentConfig,
    *,
    available_labels: tuple[str, ...],
) -> tuple[list[str], list[str], list[FscilSession]]:
    """Split classes into base and novel groups and plan the session order."""

    if len(available_labels) < 2:
        raise ValueError("FSCIL requires at least two labels to form base and novel sessions.")

    all_labels = sorted(available_labels)
    n_base = max(1, int(len(all_labels) * config.base_class_ratio))
    n_base = min(n_base, len(all_labels) - 1)

    import random

    rng = random.Random(config.base_class_seed)
    base_classes = sorted(rng.sample(all_labels, k=n_base))
    novel_classes = [label for label in all_labels if label not in base_classes]
    if config.novel_class_order == "random":
        rng.shuffle(novel_classes)
    else:
        novel_classes = sorted(novel_classes)

    sessions = [
        FscilSession(
            session_id=0,
            session_name="session_0_base",
            novel_label=None,
            is_base=True,
        )
    ]
    for session_id, novel_label in enumerate(novel_classes, start=1):
        sessions.append(
            FscilSession(
                session_id=session_id,
                session_name=f"session_{session_id}_{novel_label}",
                novel_label=novel_label,
                is_base=False,
            )
        )

    return base_classes, novel_classes, sessions


def _validate_fscil_labels(
    *,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    label_column: str,
    base_classes: list[str],
    novel_classes: list[str],
) -> None:
    """Validate that the planned FSCIL split is supported by the data."""

    train_labels = set(train[label_column].astype(str).unique().tolist())
    calibration_labels = set(calibration[label_column].astype(str).unique().tolist())
    missing_base_train = sorted(label for label in base_classes if label not in train_labels)
    if missing_base_train:
        raise ValueError(
            "Base FSCIL classes are missing from the training split: "
            f"{missing_base_train}."
        )
    missing_base_calibration = sorted(
        label for label in base_classes if label not in calibration_labels
    )
    if missing_base_calibration:
        raise ValueError(
            "Base FSCIL classes are missing from the calibration split: "
            f"{missing_base_calibration}."
        )
    missing_novel_support = sorted(label for label in novel_classes if label not in train_labels)
    if missing_novel_support:
        raise ValueError(
            "Novel FSCIL classes are missing from the training split, so support sampling is "
            f"impossible: {missing_novel_support}."
        )


def _sample_novel_support(
    *,
    df_all: pd.DataFrame,
    novel_label: str,
    label_column: str,
    fraction: float,
    seed: int,
    max_k: int | None = None,
) -> pd.DataFrame:
    """Sample a few-shot support set for one novel class."""

    novel_rows = df_all.loc[df_all[label_column] == novel_label]
    if novel_rows.empty:
        raise ValueError(f"No train rows are available for FSCIL novel label {novel_label!r}.")
    n_support = max(1, int(len(novel_rows) * fraction))
    n_support = min(n_support, int(len(novel_rows)))
    if max_k is not None:
        n_support = min(n_support, max_k)
    return novel_rows.sample(n=n_support, random_state=seed, replace=False).reset_index(drop=True)


def _apply_fscil_support(
    *,
    model_state: TrackAModelState,
    support: pd.DataFrame,
    session: FscilSession,
    config: TrackAFscilExperimentConfig,
) -> tuple[TrackAModelState, list[str]]:
    """Apply one novel-class support batch without rebuilding the base state.

    FSCIL support batches can be larger than ``k_tiny_max`` (default 4).
    We split the support into ``k_tiny_max``-sized mini-batches so that each
    chunk goes through ``_apply_bc_tiny_burst`` → ``provisional_admit_small_item``
    which successfully admits novel classes without being blocked by the global
    affine conflict review.
    """

    if isinstance(model_state, list):
        updated_models: list[BCMRNFSTModel] = []
        all_item_ids: list[str] = []
        for member_index, member_state in enumerate(model_state):
            member_state, item_ids = _apply_fscil_support_chunked(
                model_state=member_state,
                support=support,
                session=session,
                config=config,
            )
            updated_models.append(member_state)
            if member_index == 0:
                all_item_ids = item_ids
        return updated_models, all_item_ids
    if isinstance(model_state, BCMRNFSTModel):
        return _apply_fscil_support_chunked(
            model_state=model_state,
            support=support,
            session=session,
            config=config,
        )
    return update_baseline_model(model_state, support), []


def _apply_fscil_support_chunked(
    *,
    model_state: BCMRNFSTModel,
    support: pd.DataFrame,
    session: FscilSession,
    config: TrackAFscilExperimentConfig,
) -> tuple[BCMRNFSTModel, list[str]]:
    """Split FSCIL support into k_tiny_max-sized chunks and apply each via the tiny-burst path."""

    chunk_size = int(model_state.config.k_tiny_max)
    all_item_ids: list[str] = []
    n_rows = len(support)

    for chunk_start in range(0, n_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_rows)
        chunk = support.iloc[chunk_start:chunk_end].reset_index(drop=True)
        batch_id = f"fscil-{session.session_name}-c{chunk_start:04d}"
        model_state, item_ids = _apply_bc_tiny_burst(
            model_state,
            chunk,
            batch_id=batch_id,
            enable_provisional_admission=config.enable_provisional_admission,
            t_now=float(session.session_id),
        )
        all_item_ids.extend(item_ids)

    return model_state, all_item_ids


def _remember_refresh_validation_support(
    *,
    model_state: TrackAModelState,
    support: pd.DataFrame,
) -> TrackAModelState:
    """Retain FSCIL support exemplars for refresh validation when enabled."""

    if isinstance(model_state, list):
        return [
            _remember_refresh_validation_support(model_state=member_state, support=support)
            for member_state in model_state
        ]
    if not isinstance(model_state, BCMRNFSTModel):
        return model_state
    if not model_state.config.enable_refresh_validation or support.empty:
        return model_state

    X_support, _ = transform_rows(
        model_state.schema,
        model_state.preprocessor,
        support,
        label_column=model_state.config.label_column,
    )
    phi_support = np.asarray(model_state.phi.transform(X_support), dtype=np.float64)
    class_labels = support[model_state.config.label_column].astype(str).unique().tolist()
    if len(class_labels) != 1:
        raise ValueError("FSCIL support exemplars must contain exactly one class.")
    if model_state.validation_exemplars is None:
        model_state.validation_exemplars = {}
    model_state.validation_exemplars[class_labels[0]] = phi_support
    return model_state


def _is_bc_fscil_state(model_state: TrackAModelState) -> bool:
    """Return whether the state uses the GONS refresh path."""

    return isinstance(model_state, BCMRNFSTModel) or isinstance(model_state, list)


def _build_fscil_session_record(
    *,
    session: FscilSession,
    predictions: pd.DataFrame,
    test: pd.DataFrame,
    label_column: str,
    base_classes: list[str],
    novel_classes_so_far: list[str],
    known_labels: tuple[str, ...],
    support_rows: int,
    refresh_applied: bool,
) -> dict[str, object]:
    """Compute FSCIL metrics for one session."""

    protocol_known_class_count = len(base_classes) + len(novel_classes_so_far)
    true_labels = (
        predictions["true_label"].astype(str).to_numpy()
        if "true_label" in predictions.columns
        else test[label_column].astype(str).to_numpy()
    )
    predicted_labels = predictions["predicted_label"].astype(str).to_numpy()

    overall_acc = float(np.mean(true_labels == predicted_labels))

    base_mask = np.isin(true_labels, np.asarray(base_classes, dtype=object))
    base_acc = (
        float(np.mean(true_labels[base_mask] == predicted_labels[base_mask]))
        if base_mask.any()
        else 0.0
    )

    novel_acc = 0.0
    if novel_classes_so_far:
        novel_mask = np.isin(true_labels, np.asarray(novel_classes_so_far, dtype=object))
        if novel_mask.any():
            novel_acc = float(
                np.mean(true_labels[novel_mask] == predicted_labels[novel_mask])
            )

    harmonic_mean = 0.0
    if novel_classes_so_far and (base_acc + novel_acc) > 0.0:
        harmonic_mean = float((2.0 * base_acc * novel_acc) / (base_acc + novel_acc))

    all_labels = sorted(set(true_labels.tolist()))
    f1_macro = float(
        f1_score(
            true_labels,
            predicted_labels,
            labels=all_labels,
            average="macro",
            zero_division=0,
        )
    )
    base_f1 = float(
        f1_score(
            true_labels,
            predicted_labels,
            labels=base_classes,
            average="macro",
            zero_division=0,
        )
    )
    novel_f1 = 0.0
    if novel_classes_so_far:
        novel_f1 = float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=novel_classes_so_far,
                average="macro",
                zero_division=0,
            )
        )

    return {
        "session_id": session.session_id,
        "session_name": session.session_name,
        "novel_label": session.novel_label,
        "is_base": session.is_base,
        "n_known_classes": max(len(known_labels), protocol_known_class_count),
        "state_known_class_count": len(known_labels),
        "support_rows": int(support_rows),
        "refresh_applied": bool(refresh_applied),
        "overall_acc": overall_acc,
        "base_acc": base_acc,
        "novel_acc": novel_acc,
        "harmonic_mean": harmonic_mean,
        "f1_macro": f1_macro,
        "base_f1": base_f1,
        "novel_f1": novel_f1,
    }


def _prepare_session_predictions_output(
    predictions: pd.DataFrame,
    *,
    session_record: dict[str, object],
) -> pd.DataFrame:
    """Annotate predictions with FSCIL session metadata."""

    output = predictions.copy()
    output.insert(0, "session_id", cast(int, session_record["session_id"]))
    output.insert(1, "session_name", str(session_record["session_name"]))
    output.insert(
        2,
        "novel_label",
        "" if session_record["novel_label"] is None else str(session_record["novel_label"]),
    )
    output.insert(3, "support_rows", cast(int, session_record["support_rows"]))
    return output


def _build_fscil_run_summary(
    *,
    session_records: list[dict[str, object]],
    base_classes: list[str],
    novel_classes: list[str],
) -> dict[str, object]:
    """Aggregate FSCIL results across sessions."""

    base_record = session_records[0]
    final_record = session_records[-1]
    novel_records = session_records[1:]

    base_accuracy_delta = float(final_record["base_acc"]) - float(base_record["base_acc"])
    performance_dropping_rate = float(base_record["base_acc"]) - float(final_record["base_acc"])
    avg_novel_acc = (
        float(np.mean([float(record["novel_acc"]) for record in novel_records]))
        if novel_records
        else 0.0
    )
    avg_harmonic_mean = (
        float(np.mean([float(record["harmonic_mean"]) for record in novel_records]))
        if novel_records
        else 0.0
    )

    return {
        "base_classes": base_classes,
        "novel_classes": novel_classes,
        "n_sessions": len(session_records),
        "base_session_acc": float(base_record["overall_acc"]),
        "base_session_base_acc": float(base_record["base_acc"]),
        "final_overall_acc": float(final_record["overall_acc"]),
        "final_base_acc": float(final_record["base_acc"]),
        "final_novel_acc": float(final_record["novel_acc"]),
        "final_harmonic_mean": float(final_record["harmonic_mean"]),
        "final_f1_macro": float(final_record["f1_macro"]),
        "final_novel_f1": float(final_record["novel_f1"]),
        "base_accuracy_delta": base_accuracy_delta,
        "performance_dropping_rate": performance_dropping_rate,
        "avg_novel_acc": avg_novel_acc,
        "avg_harmonic_mean": avg_harmonic_mean,
        "refresh_event_count": int(
            sum(int(bool(record["refresh_applied"])) for record in session_records)
        ),
    }


def _render_summary(
    *,
    method: str,
    run_id: str,
    run_summary: dict[str, object],
    session_metrics: pd.DataFrame,
) -> str:
    """Render a compact markdown summary for one FSCIL run."""

    lines = [
        "# GONS FSCIL Summary",
        "",
        f"- Method: `{method}`",
        f"- Run ID: `{run_id}`",
        f"- Sessions: `{run_summary['n_sessions']}`",
        f"- Base Classes: `{run_summary['base_classes']}`",
        f"- Novel Classes: `{run_summary['novel_classes']}`",
        "",
        "## Aggregate Metrics",
        "",
    ]
    for key, value in sorted(run_summary.items()):
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Sessions",
            "",
            "| Session | Novel Label | Known Classes | Overall Acc | Base Acc | Novel Acc | Harmonic Mean |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in session_metrics.sort_values("session_id").iterrows():
        lines.append(
            "| "
            f"{row['session_name']} | "
            f"{row['novel_label'] or 'base'} | "
            f"{int(row['n_known_classes'])} | "
            f"{_fmt_metric(row.get('overall_acc'))} | "
            f"{_fmt_metric(row.get('base_acc'))} | "
            f"{_fmt_metric(row.get('novel_acc'))} | "
            f"{_fmt_metric(row.get('harmonic_mean'))} |"
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


__all__ = [
    "FscilSession",
    "TrackAFscilRunResult",
    "run_track_a_fscil_experiment",
]
