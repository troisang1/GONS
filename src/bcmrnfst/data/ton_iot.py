"""ToN-IoT dataset loading helpers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bcmrnfst.config.data import ToNIoTDatasetConfig
from bcmrnfst.data.manifests import DatasetSplitManifest
from bcmrnfst.preprocess.schema import infer_feature_groups
from bcmrnfst.runtime import find_repo_root


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Summary metadata extracted from a labeled tabular dataset."""

    row_count: int
    feature_count: int
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    label_distribution: dict[str, int]


@dataclass(frozen=True, slots=True)
class LoadedTabularDataset:
    """Loaded dataset split with normalized labels and metadata."""

    frame: pd.DataFrame
    label_column: str
    metadata: DatasetMetadata


def normalize_csv_column_name(column: object) -> str:
    """Normalize CSV headers so public ToN-IoT exports map cleanly to config keys."""

    return str(column).replace("\ufeff", "").strip()


def read_ton_iot_frame(csv_path: Path) -> pd.DataFrame:
    """Read a ToN-IoT CSV and normalize header formatting quirks."""

    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    return frame.rename(columns=normalize_csv_column_name)


def normalize_label_value(value: object) -> str:
    """Normalize a raw label value into a stable lowercase token."""

    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower()).strip("_")
    return normalized or "unknown"


def normalize_label_series(series: pd.Series) -> pd.Series:
    """Normalize an entire label series."""

    return series.map(normalize_label_value)


def detect_label_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    """Find the first matching label column from the configured candidates."""

    column_lookup: dict[str, str] = {
        str(column).casefold(): str(column) for column in frame.columns
    }
    for candidate in candidates:
        match = column_lookup.get(candidate.casefold())
        if match is not None:
            return match
    raise KeyError(f"Could not find a label column. Candidates: {candidates}")


def extract_dataset_metadata(
    frame: pd.DataFrame,
    *,
    label_column: str,
    numeric_overrides: list[str] | None = None,
    categorical_overrides: list[str] | None = None,
) -> DatasetMetadata:
    """Extract row counts, feature counts, and feature groups."""

    feature_order, numeric_cols, categorical_cols = infer_feature_groups(
        frame,
        label_column=label_column,
        numeric_overrides=numeric_overrides or [],
        categorical_overrides=categorical_overrides or [],
    )
    label_distribution = {
        str(label): int(count)
        for label, count in frame[label_column].value_counts(dropna=False).sort_index().items()
    }
    return DatasetMetadata(
        row_count=int(len(frame)),
        feature_count=len(feature_order),
        numeric_features=tuple(numeric_cols),
        categorical_features=tuple(categorical_cols),
        label_distribution=label_distribution,
    )


def load_ton_iot_csv(csv_path: Path, config: ToNIoTDatasetConfig) -> LoadedTabularDataset:
    """Load a ToN-IoT CSV file and normalize its label contract."""

    frame = read_ton_iot_frame(csv_path)
    source_label_column = detect_label_column(frame, config.label_candidates())
    if source_label_column != config.label_column:
        frame = frame.rename(columns={source_label_column: config.label_column})

    drop_candidates = [column for column in config.drop_columns if column in frame.columns]
    if drop_candidates:
        frame = frame.drop(columns=drop_candidates)

    frame[config.label_column] = normalize_label_series(frame[config.label_column])
    metadata = extract_dataset_metadata(
        frame,
        label_column=config.label_column,
        numeric_overrides=config.numeric_overrides,
        categorical_overrides=config.categorical_overrides,
    )
    return LoadedTabularDataset(frame=frame, label_column=config.label_column, metadata=metadata)


def load_manifest_split(
    manifest: DatasetSplitManifest,
    split_name: str,
    *,
    config: ToNIoTDatasetConfig,
    repo_root: Path | None = None,
) -> LoadedTabularDataset:
    """Load a dataset split described by a manifest."""

    root = repo_root or find_repo_root()
    split_path = manifest.resolve_split_path(split_name, repo_root=root)
    return load_ton_iot_csv(split_path, config)
