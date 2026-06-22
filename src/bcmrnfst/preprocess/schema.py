"""Frozen schema helpers and alignment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

FrameLike: TypeAlias = pd.DataFrame | pd.Series | Mapping[str, object]


@dataclass(frozen=True, slots=True)
class FrozenSchema:
    """Frozen input schema captured at initialization."""

    feature_order: tuple[str, ...]
    numeric_cols: tuple[str, ...]
    categorical_cols: tuple[str, ...]
    dropped_extra_cols_policy: str = "drop_and_log"
    post_fit_input_dim: int | None = None

    def with_post_fit_input_dim(self, width: int) -> FrozenSchema:
        """Return a copy of the schema with the fitted output width attached."""

        return FrozenSchema(
            feature_order=self.feature_order,
            numeric_cols=self.numeric_cols,
            categorical_cols=self.categorical_cols,
            dropped_extra_cols_policy=self.dropped_extra_cols_policy,
            post_fit_input_dim=width,
        )


@dataclass(frozen=True, slots=True)
class SchemaAlignmentReport:
    """Observed schema drift during alignment."""

    missing_columns: tuple[str, ...]
    extra_columns: tuple[str, ...]


def coerce_frame(data: FrameLike) -> pd.DataFrame:
    """Coerce row-like inputs into a dataframe."""

    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pd.Series):
        return data.to_frame().T
    return pd.DataFrame([dict(data)])


def infer_feature_groups(
    frame: pd.DataFrame,
    *,
    label_column: str,
    numeric_overrides: list[str] | None = None,
    categorical_overrides: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Infer feature order plus numeric and categorical partitions."""

    feature_frame = frame.drop(columns=[label_column], errors="ignore")
    feature_order = list(feature_frame.columns)

    numeric_override_set = set(numeric_overrides or [])
    categorical_override_set = set(categorical_overrides or [])
    overlap = numeric_override_set & categorical_override_set
    if overlap:
        raise ValueError(f"Feature overrides overlap across numeric/categorical groups: {overlap}")

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for column in feature_order:
        if column in categorical_override_set:
            categorical_cols.append(column)
            continue
        if column in numeric_override_set or is_numeric_dtype(feature_frame[column]):
            numeric_cols.append(column)
            continue
        categorical_cols.append(column)
    return feature_order, numeric_cols, categorical_cols


def align_to_schema(
    schema: FrozenSchema,
    data: FrameLike,
    *,
    label_column: str = "label",
) -> tuple[pd.DataFrame, SchemaAlignmentReport]:
    """Align rows to the frozen schema, filling missing columns and dropping extras."""

    frame = coerce_frame(data)
    if label_column in frame.columns:
        frame = frame.drop(columns=[label_column])

    missing_columns = [column for column in schema.feature_order if column not in frame.columns]
    for column in missing_columns:
        if column in schema.numeric_cols:
            frame[column] = float("nan")
        else:
            frame[column] = pd.Series(
                [np.nan] * len(frame),
                index=frame.index,
                dtype="object",
            )

    extra_columns = [column for column in frame.columns if column not in schema.feature_order]
    if extra_columns:
        frame = frame.drop(columns=extra_columns)

    aligned = frame.loc[:, list(schema.feature_order)].copy()
    report = SchemaAlignmentReport(
        missing_columns=tuple(missing_columns),
        extra_columns=tuple(extra_columns),
    )
    return aligned, report
