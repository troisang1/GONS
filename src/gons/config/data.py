"""Dataset configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, PositiveInt, field_validator

from gons.runtime import find_repo_root

PRACTICAL_UNBOUNDED_PREPROCESSING_LIMIT = 1_000_000_000


class PreprocessingConfig(BaseModel):
    """Preprocessing settings for bounded tabular inputs."""

    numeric_scaler: Literal["robust", "quantile", "standard", "minmax", "power"] = "robust"
    bound_categorical_cardinality: bool = False
    ohe_min_frequency: int | float = 2
    ohe_max_categories: PositiveInt = PRACTICAL_UNBOUNDED_PREPROCESSING_LIMIT
    d_input_max: PositiveInt = PRACTICAL_UNBOUNDED_PREPROCESSING_LIMIT


class ToNIoTLayoutConfig(BaseModel):
    """Filesystem layout for the local ToN-IoT dataset copy."""

    processed_dir: str = "processed"
    train_test_dir: str = "train_test"
    metadata_dir: str = "metadata"
    description_files: list[str] = Field(default_factory=list)


class ToNIoTSubsetConfig(BaseModel):
    """Named subset policy for a ToN-IoT experiment family."""

    name: str
    description: str
    manifest: Path
    preferred_sources: list[str] = Field(default_factory=list)


class ToNIoTDatasetConfig(BaseModel):
    """Typed dataset contract for ToN-IoT ingestion."""

    name: str
    version: str
    description: str
    raw_root: Path
    manifest_root: Path
    default_subset: str
    label_column: str = "label"
    label_aliases: list[str] = Field(default_factory=list)
    drop_columns: list[str] = Field(default_factory=list)
    numeric_overrides: list[str] = Field(default_factory=list)
    categorical_overrides: list[str] = Field(default_factory=list)
    subsets: list[ToNIoTSubsetConfig] = Field(default_factory=list)
    layout: ToNIoTLayoutConfig = Field(default_factory=ToNIoTLayoutConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    source_path: Path | None = Field(default=None, exclude=True, repr=False)

    @field_validator(
        "label_aliases",
        "drop_columns",
        "numeric_overrides",
        "categorical_overrides",
        mode="before",
    )
    @classmethod
    def deduplicate_strings(cls, value: object) -> list[str]:
        """Deduplicate string lists while preserving order."""

        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("Expected a list of strings.")
        deduplicated: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise TypeError("Expected a list of strings.")
            if item not in deduplicated:
                deduplicated.append(item)
        return deduplicated

    def label_candidates(self) -> list[str]:
        """Return the canonical label column candidates in search order."""

        ordered = [self.label_column, *self.label_aliases]
        deduplicated: list[str] = []
        for candidate in ordered:
            if candidate not in deduplicated:
                deduplicated.append(candidate)
        return deduplicated

    def resolve_raw_root(self, repo_root: Path | None = None) -> Path:
        """Resolve the dataset raw root against the repository root."""

        root = repo_root or find_repo_root()
        return self.raw_root if self.raw_root.is_absolute() else root / self.raw_root

    def resolve_manifest_root(self, repo_root: Path | None = None) -> Path:
        """Resolve the manifest root against the repository root."""

        root = repo_root or find_repo_root()
        return self.manifest_root if self.manifest_root.is_absolute() else root / self.manifest_root

    def subset(self, name: str | None = None) -> ToNIoTSubsetConfig:
        """Return a configured subset by name."""

        target = name or self.default_subset
        for subset in self.subsets:
            if subset.name == target:
                return subset
        raise KeyError(f"Unknown ToN-IoT subset: {target}")

    def resolve_subset_manifest(
        self, name: str | None = None, repo_root: Path | None = None
    ) -> Path:
        """Resolve a subset manifest path against the repository root."""

        subset = self.subset(name)
        manifest_path = subset.manifest
        root = repo_root or find_repo_root()
        return manifest_path if manifest_path.is_absolute() else root / manifest_path


def load_ton_iot_config(
    path: Path | None = None, *, repo_root: Path | None = None
) -> ToNIoTDatasetConfig:
    """Load the repository ToN-IoT dataset configuration."""

    root = repo_root or find_repo_root()
    config_path = path or (root / "configs" / "data" / "ton_iot.yaml")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    config = ToNIoTDatasetConfig.model_validate(payload)
    return config.model_copy(update={"source_path": config_path})


def ensure_ton_iot_layout(
    config: ToNIoTDatasetConfig, *, repo_root: Path | None = None
) -> dict[str, Path]:
    """Return the expected ToN-IoT layout paths for the current config."""

    root = repo_root or find_repo_root()
    raw_root = config.resolve_raw_root(root)
    manifest_root = config.resolve_manifest_root(root)
    return {
        "raw_root": raw_root,
        "manifest_root": manifest_root,
        "processed_dir": raw_root / config.layout.processed_dir,
        "train_test_dir": raw_root / config.layout.train_test_dir,
        "metadata_dir": raw_root / config.layout.metadata_dir,
    }
