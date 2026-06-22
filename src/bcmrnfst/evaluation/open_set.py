"""Shared open-set evaluation helpers."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, roc_auc_score

from bcmrnfst.continual.inference import predict_many

if TYPE_CHECKING:
    from bcmrnfst.state.model import BCMRNFSTModel


def predict_bc_mrnfst_dataframe(model: BCMRNFSTModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Run GONS open-set inference on a labeled dataframe."""

    results = predict_many(model, frame)
    true_labels = (
        frame[model.config.label_column].astype(str).tolist()
        if model.config.label_column in frame.columns
        else [None] * len(frame)
    )
    records: list[dict[str, object]] = []
    for row_index, result in enumerate(results):
        accepted = result.class_label != "Unknown"
        records.append(
            {
                "row_index": row_index,
                "true_label": true_labels[row_index],
                "predicted_label": (
                    result.class_label.lower() if not accepted else result.class_label
                ),
                "accepted": accepted,
                "best_prototype_id": result.proto_id,
                "best_class_label": model.proto_bank[result.proto_id].class_label,
                "score": float(result.score),
                "threshold": float(result.tau_effective),
                "projection_version": int(result.projection_version),
            }
        )
    return pd.DataFrame.from_records(records)


def summarize_open_set_predictions(
    predictions: pd.DataFrame,
    *,
    known_labels: Collection[str],
) -> dict[str, float | int]:
    """Compute method-agnostic open-set classification metrics."""

    if predictions.empty:
        return {
            "accepted_count": 0,
            "rejected_count": 0,
            "total_rows": 0,
            "overall_open_set_accuracy": 0.0,
        }

    known_label_set = {str(label) for label in known_labels}
    true_labels = predictions["true_label"].astype(str)
    predicted_labels = predictions["predicted_label"].astype(str)
    accepted_mask = predictions["accepted"].astype(bool).to_numpy(dtype=bool)
    known_mask = true_labels.isin(known_label_set).to_numpy(dtype=bool)
    unknown_mask = ~known_mask
    correct_label_mask = (predicted_labels == true_labels).to_numpy(dtype=bool)
    open_set_correct = (known_mask & accepted_mask & correct_label_mask) | (
        unknown_mask & ~accepted_mask
    )

    metrics: dict[str, float | int] = {
        "accepted_count": int(np.sum(accepted_mask)),
        "rejected_count": int(np.sum(~accepted_mask)),
        "total_rows": int(len(predictions)),
        "overall_open_set_accuracy": float(np.mean(open_set_correct)),
    }
    if np.any(accepted_mask):
        metrics["accepted_accuracy"] = float(np.mean(correct_label_mask[accepted_mask]))
    if np.any(known_mask):
        metrics["known_class_accuracy"] = float(np.mean(correct_label_mask[known_mask]))
        metrics["known_macro_f1"] = float(
            f1_score(
                true_labels[known_mask],
                predicted_labels[known_mask],
                labels=sorted(known_label_set),
                average="macro",
                zero_division=0,
            )
        )
        metrics["known_mcc"] = float(
            matthews_corrcoef(
                true_labels[known_mask],
                predicted_labels[known_mask],
            )
        )
    if np.any(unknown_mask):
        metrics["unknown_rejection_rate"] = float(np.mean(~accepted_mask[unknown_mask]))
        metrics["false_known_rate"] = float(np.mean(accepted_mask[unknown_mask]))
    rejection_scores = _rejection_scores(predictions)
    if np.any(known_mask) and np.any(unknown_mask):
        targets = unknown_mask.astype(np.int64)
        metrics["unknown_rejection_auroc"] = float(roc_auc_score(targets, rejection_scores))
        metrics["unknown_rejection_aupr"] = float(
            average_precision_score(targets, rejection_scores)
        )
    return metrics


def aggregate_static_open_set_metrics(episode_metrics: pd.DataFrame) -> dict[str, object]:
    """Aggregate per-episode S0 metrics into one run-level summary."""

    aggregate: dict[str, object] = {
        "episode_count": int(len(episode_metrics)),
    }
    if episode_metrics.empty:
        return aggregate

    mean_columns = (
        "accepted_accuracy",
        "false_known_rate",
        "known_class_accuracy",
        "known_mcc",
        "known_macro_f1",
        "overall_open_set_accuracy",
        "projection_dim",
        "prototype_count",
        "tau_global",
        "unknown_rejection_aupr",
        "unknown_rejection_auroc",
        "unknown_rejection_rate",
    )
    for column in mean_columns:
        if column not in episode_metrics.columns:
            continue
        values = pd.to_numeric(episode_metrics[column], errors="coerce").dropna()
        if not values.empty:
            aggregate[f"mean_{column}"] = float(values.mean())

    for column in ("accepted_count", "rejected_count", "total_rows"):
        if column in episode_metrics.columns:
            aggregate[f"sum_{column}"] = int(
                pd.to_numeric(episode_metrics[column], errors="coerce").fillna(0).sum()
            )

    for column in ("calibration_mode", "projection_mode", "protocol"):
        if column not in episode_metrics.columns:
            continue
        counts = (
            episode_metrics[column]
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )
        aggregate[f"{column}_counts"] = {str(key): int(value) for key, value in counts.items()}

    aggregate["max_holdout_size"] = int(
        pd.to_numeric(episode_metrics["holdout_size"], errors="coerce").fillna(0).max()
    )
    return aggregate


def _rejection_scores(predictions: pd.DataFrame) -> np.ndarray:
    score = pd.to_numeric(predictions["score"], errors="coerce").to_numpy(dtype=np.float64)
    threshold = (
        pd.to_numeric(predictions["threshold"], errors="coerce")
        .replace(0.0, np.finfo(np.float64).eps)
        .to_numpy(dtype=np.float64)
    )
    return np.asarray(score / threshold, dtype=np.float64)


__all__ = [
    "aggregate_static_open_set_metrics",
    "predict_bc_mrnfst_dataframe",
    "summarize_open_set_predictions",
]
