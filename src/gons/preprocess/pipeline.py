"""Bounded preprocessing pipeline and serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from gons.config.data import PreprocessingConfig
from gons.preprocess.schema import (
    FrameLike,
    FrozenSchema,
    SchemaAlignmentReport,
    align_to_schema,
    infer_feature_groups,
)


class PreprocessingWidthError(ValueError):
    """Raised when the fitted preprocessing width exceeds the configured budget."""


@dataclass(frozen=True, slots=True)
class FittedPreprocessorBundle:
    """Serializable bundle containing schema, preprocessor, and feature names."""

    schema: FrozenSchema
    preprocessor: ColumnTransformer
    config: PreprocessingConfig
    output_features: tuple[str, ...]


def build_preprocessor(schema: FrozenSchema, config: PreprocessingConfig) -> ColumnTransformer:
    """Build the bounded preprocessing pipeline."""

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if schema.numeric_cols:
        if config.numeric_scaler == "robust":
            scaler = RobustScaler()
        elif config.numeric_scaler == "standard":
            scaler = StandardScaler()
        elif config.numeric_scaler == "minmax":
            scaler = MinMaxScaler()
        elif config.numeric_scaler == "power":
            scaler = PowerTransformer(method="yeo-johnson", standardize=True)
        else:
            scaler = QuantileTransformer(
                output_distribution="normal",
                random_state=0,
            )
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", scaler),
            ]
        )
        transformers.append(("num", numeric_pipeline, list(schema.numeric_cols)))

    if schema.categorical_cols:
        if config.bound_categorical_cardinality:
            encoder = OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=config.ohe_min_frequency,
                max_categories=config.ohe_max_categories,
                sparse_output=False,
            )
        else:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", encoder),
            ]
        )
        transformers.append(("cat", categorical_pipeline, list(schema.categorical_cols)))

    if not transformers:
        raise ValueError("At least one numeric or categorical feature is required.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def fit_schema_and_preprocessor(
    frame: pd.DataFrame,
    *,
    label_column: str,
    config: PreprocessingConfig,
    numeric_overrides: list[str] | None = None,
    categorical_overrides: list[str] | None = None,
) -> FittedPreprocessorBundle:
    """Fit the frozen schema and bounded preprocessing pipeline."""

    feature_order, numeric_cols, categorical_cols = infer_feature_groups(
        frame,
        label_column=label_column,
        numeric_overrides=numeric_overrides or [],
        categorical_overrides=categorical_overrides or [],
    )
    schema = FrozenSchema(
        feature_order=tuple(feature_order),
        numeric_cols=tuple(numeric_cols),
        categorical_cols=tuple(categorical_cols),
    )
    aligned_frame, _ = align_to_schema(schema, frame, label_column=label_column)
    preprocessor = build_preprocessor(schema, config)
    preprocessor.fit(aligned_frame)

    output_features = tuple(str(name) for name in preprocessor.get_feature_names_out())
    if len(output_features) > config.d_input_max:
        raise PreprocessingWidthError(
            "Fitted preprocessing width exceeds d_input_max: "
            f"{len(output_features)} > {config.d_input_max}"
        )

    return FittedPreprocessorBundle(
        schema=schema.with_post_fit_input_dim(len(output_features)),
        preprocessor=preprocessor,
        config=config,
        output_features=output_features,
    )


def transform_rows(
    schema: FrozenSchema,
    preprocessor: ColumnTransformer,
    data: FrameLike,
    *,
    label_column: str = "label",
) -> tuple[np.ndarray, SchemaAlignmentReport]:
    """Align and transform rows through the fitted preprocessor."""

    aligned, report = align_to_schema(schema, data, label_column=label_column)
    transformed = preprocessor.transform(aligned)
    transformed = np.asarray(transformed, dtype=float)
    # QuantileTransformer can produce NaN/inf on degenerate columns under
    # certain seeds; replace with finite values so downstream RFF/RBFSampler
    # never sees missing data.
    if not np.all(np.isfinite(transformed)):
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
    return transformed, report


def save_preprocessor_bundle(bundle: FittedPreprocessorBundle, path: Path) -> None:
    """Persist the fitted schema and preprocessor bundle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": bundle.schema,
        "preprocessor": bundle.preprocessor,
        "config": bundle.config.model_dump(mode="python"),
        "output_features": bundle.output_features,
    }
    joblib.dump(payload, path)


def load_preprocessor_bundle(path: Path) -> FittedPreprocessorBundle:
    """Load a previously persisted schema and preprocessor bundle."""

    payload = joblib.load(path)
    return FittedPreprocessorBundle(
        schema=payload["schema"],
        preprocessor=payload["preprocessor"],
        config=PreprocessingConfig.model_validate(payload["config"]),
        output_features=tuple(payload["output_features"]),
    )
