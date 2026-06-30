"""Helpers for building reproducible ToN-IoT subset splits."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from gons.data.ton_iot import normalize_label_series, read_ton_iot_frame

DEFAULT_IDENTIFIER_COLUMNS = (
    "src_ip",
    "dst_ip",
    "src_mac",
    "dst_mac",
)
DEFAULT_AUXILIARY_LABEL_COLUMNS = (
    "label",
    "type",
)


@dataclass(frozen=True, slots=True)
class GeneratedSubset:
    """Summary of a generated balanced subset split."""

    calibration_path: Path
    config_dir: Path
    output_dir: Path
    requested_samples_per_class: int
    run_output_root: Path
    selected_counts: dict[str, int]
    source_path: Path
    test_path: Path
    test_rows: int
    train_path: Path
    train_rows: int
    calibration_rows: int


def split_counts_by_ratio(
    total_rows: int,
    *,
    train_ratio: float,
    calibration_ratio: float,
) -> tuple[int, int, int]:
    """Split a class count by ratio using the largest-remainder method."""

    test_ratio = 1.0 - train_ratio - calibration_ratio
    exact = (
        total_rows * train_ratio,
        total_rows * calibration_ratio,
        total_rows * test_ratio,
    )
    counts = [math.floor(value) for value in exact]
    remainder = total_rows - sum(counts)
    order = sorted(
        range(len(exact)),
        key=lambda index: (exact[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def build_balanced_ton_iot_subset(
    *,
    source_path: Path,
    output_dir: Path,
    config_dir: Path,
    run_output_root: Path,
    samples_per_class: int,
    seed: int,
    label_source_column: str = "type",
    label_column: str = "label",
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    identifier_columns: tuple[str, ...] = DEFAULT_IDENTIFIER_COLUMNS,
    auxiliary_label_columns: tuple[str, ...] = DEFAULT_AUXILIARY_LABEL_COLUMNS,
) -> GeneratedSubset:
    """Materialize a balanced subset with capped per-class sampling."""

    frame = read_ton_iot_frame(source_path)
    if label_source_column not in frame.columns:
        raise KeyError(
            f"Could not find the label source column {label_source_column!r} in {source_path}."
        )

    frame[label_column] = normalize_label_series(frame[label_source_column])
    drop_columns = {
        column
        for column in (*identifier_columns, *auxiliary_label_columns)
        if column in frame.columns and column != label_column
    }
    if drop_columns:
        frame = frame.drop(columns=sorted(drop_columns))

    train_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    available_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    split_counts: dict[str, dict[str, int]] = {}

    for label_value, group in frame.groupby(label_column, sort=True):
        label = str(label_value)
        available_count = int(len(group))
        selected_count = min(samples_per_class, available_count)
        sampled = group.sample(n=selected_count, random_state=seed, replace=False)
        shuffled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        train_count, calibration_count, test_count = split_counts_by_ratio(
            selected_count,
            train_ratio=train_ratio,
            calibration_ratio=calibration_ratio,
        )
        train_parts.append(shuffled.iloc[:train_count].copy())
        calibration_parts.append(
            shuffled.iloc[train_count : train_count + calibration_count].copy()
        )
        test_parts.append(shuffled.iloc[train_count + calibration_count :].copy())
        available_counts[label] = available_count
        selected_counts[label] = selected_count
        split_counts[label] = {
            "train": train_count,
            "calibration": calibration_count,
            "test": test_count,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    run_output_root.mkdir(parents=True, exist_ok=True)

    train = _concat_or_empty(train_parts, frame.columns)
    calibration = _concat_or_empty(calibration_parts, frame.columns)
    test = _concat_or_empty(test_parts, frame.columns)

    train_path = output_dir / "network_train.csv"
    calibration_path = output_dir / "network_cal.csv"
    test_path = output_dir / "network_test.csv"
    train.to_csv(train_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    test.to_csv(test_path, index=False)

    metadata = {
        "available_counts": available_counts,
        "calibration_ratio": calibration_ratio,
        "capped_classes": {
            label: available
            for label, available in available_counts.items()
            if selected_counts[label] < samples_per_class
        },
        "config_dir": str(config_dir),
        "dropped_auxiliary_label_columns": sorted(
            column for column in auxiliary_label_columns if column in drop_columns
        ),
        "dropped_identifier_columns": sorted(
            column for column in identifier_columns if column in drop_columns
        ),
        "label_column": label_column,
        "label_source_column": label_source_column,
        "requested_samples_per_class": samples_per_class,
        "run_output_root": str(run_output_root),
        "seed": seed,
        "selected_counts": selected_counts,
        "selection_policy": "min(requested_samples_per_class, available_class_rows)",
        "source_path": str(source_path),
        "split_counts": split_counts,
        "split_totals": {
            "train": int(len(train)),
            "calibration": int(len(calibration)),
            "test": int(len(test)),
        },
        "test_ratio": 1.0 - train_ratio - calibration_ratio,
        "train_ratio": train_ratio,
    }
    (output_dir / "subset_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return GeneratedSubset(
        calibration_path=calibration_path,
        calibration_rows=int(len(calibration)),
        config_dir=config_dir,
        output_dir=output_dir,
        requested_samples_per_class=samples_per_class,
        run_output_root=run_output_root,
        selected_counts=selected_counts,
        source_path=source_path,
        test_path=test_path,
        test_rows=int(len(test)),
        train_path=train_path,
        train_rows=int(len(train)),
    )


def _concat_or_empty(parts: list[pd.DataFrame], columns: pd.Index) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)


__all__ = [
    "DEFAULT_AUXILIARY_LABEL_COLUMNS",
    "DEFAULT_IDENTIFIER_COLUMNS",
    "GeneratedSubset",
    "build_balanced_ton_iot_subset",
    "split_counts_by_ratio",
]
