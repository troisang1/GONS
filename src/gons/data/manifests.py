"""Dataset manifest models and loaders."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, PositiveInt, model_validator

from gons.runtime import find_repo_root


class SplitDefinition(BaseModel):
    """Single dataset split definition."""

    path: Path
    purpose: str
    description: str = ""
    known_labels: list[str] = Field(default_factory=list)
    novel_labels: list[str] = Field(default_factory=list)


class StreamEpisodeDefinition(BaseModel):
    """Minimal stream construction specification."""

    name: str
    split: str
    batch_size: PositiveInt
    label_latency: str = "immediate"
    description: str = ""


class DatasetSplitManifest(BaseModel):
    """Manifest describing how a dataset subset maps to train/cal/test files."""

    dataset: str
    subset: str
    version: str
    label_column: str = "label"
    splits: dict[str, SplitDefinition]
    stream_episodes: list[StreamEpisodeDefinition] = Field(default_factory=list)
    notes: str = ""
    source_path: Path | None = Field(default=None, exclude=True, repr=False)

    @model_validator(mode="after")
    def validate_stream_episode_splits(self) -> DatasetSplitManifest:
        """Ensure stream episodes reference known split names."""

        for episode in self.stream_episodes:
            if episode.split not in self.splits:
                raise ValueError(
                    f"Unknown split '{episode.split}' in stream episode '{episode.name}'."
                )
        return self

    def resolve_split_path(self, split_name: str, repo_root: Path | None = None) -> Path:
        """Resolve a split path relative to the manifest location or repo root."""

        if split_name not in self.splits:
            raise KeyError(f"Unknown split: {split_name}")

        split_path = self.splits[split_name].path
        if split_path.is_absolute():
            return split_path
        if self.source_path is not None:
            return (self.source_path.parent / split_path).resolve()
        root = repo_root or find_repo_root()
        return (root / split_path).resolve()


def load_dataset_split_manifest(
    path: Path, *, repo_root: Path | None = None
) -> DatasetSplitManifest:
    """Load a dataset split manifest from YAML."""

    resolved_path = path if path.is_absolute() else (repo_root or find_repo_root()) / path
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {resolved_path}")
    manifest = DatasetSplitManifest.model_validate(payload)
    return manifest.model_copy(update={"source_path": resolved_path})
