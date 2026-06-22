"""Helpers for building continual batches from manifests."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bcmrnfst.config.data import ToNIoTDatasetConfig
from bcmrnfst.data.manifests import DatasetSplitManifest
from bcmrnfst.data.ton_iot import load_manifest_split


@dataclass(frozen=True, slots=True)
class SessionBatch:
    """One ordered labeled continual-learning batch."""

    batch_id: str
    episode_name: str
    frame: pd.DataFrame
    label_latency: str
    order: int
    split_name: str


def _build_batches_from_manifest(
    manifest: DatasetSplitManifest,
    *,
    dataset_config: ToNIoTDatasetConfig,
    allowed_label_latencies: Collection[str] | None,
    repo_root: Path | None = None,
    selected_names: Collection[str] | None = None,
    max_batches: int | None = None,
) -> list[SessionBatch]:
    """Expand manifest stream episodes into ordered labeled batches."""

    normalized_names = None if selected_names is None else {str(name) for name in selected_names}
    allowed_latencies = None if allowed_label_latencies is None else {
        str(latency) for latency in allowed_label_latencies
    }
    batches: list[SessionBatch] = []
    for episode in manifest.stream_episodes:
        if normalized_names is not None and episode.name not in normalized_names:
            continue
        if allowed_latencies is not None and episode.label_latency not in allowed_latencies:
            raise ValueError(
                f"Unsupported label latency '{episode.label_latency}' for episode "
                f"'{episode.name}'. Allowed values: {sorted(allowed_latencies)}."
            )
        split_frame = load_manifest_split(
            manifest,
            episode.split,
            config=dataset_config,
            repo_root=repo_root,
        ).frame.reset_index(drop=True)
        for order, start in enumerate(range(0, len(split_frame), episode.batch_size)):
            batch_frame = split_frame.iloc[start : start + episode.batch_size].copy()
            if batch_frame.empty:
                continue
            batches.append(
                SessionBatch(
                    batch_id=f"{episode.name}-s{order:03d}",
                    episode_name=episode.name,
                    frame=batch_frame,
                    label_latency=episode.label_latency,
                    order=len(batches),
                    split_name=episode.split,
                )
            )
            if max_batches is not None and len(batches) >= max_batches:
                return batches
    return batches


def build_session_batches_from_manifest(
    manifest: DatasetSplitManifest,
    *,
    dataset_config: ToNIoTDatasetConfig,
    repo_root: Path | None = None,
    session_names: Collection[str] | None = None,
    max_sessions: int | None = None,
) -> list[SessionBatch]:
    """Expand manifest stream episodes into ordered immediate-label session batches."""

    return _build_batches_from_manifest(
        manifest,
        dataset_config=dataset_config,
        allowed_label_latencies={"immediate"},
        repo_root=repo_root,
        selected_names=session_names,
        max_batches=max_sessions,
    )


def build_stream_batches_from_manifest(
    manifest: DatasetSplitManifest,
    *,
    dataset_config: ToNIoTDatasetConfig,
    repo_root: Path | None = None,
    stream_names: Collection[str] | None = None,
    max_batches: int | None = None,
) -> list[SessionBatch]:
    """Expand manifest stream episodes into ordered prequential stream batches."""

    return _build_batches_from_manifest(
        manifest,
        dataset_config=dataset_config,
        allowed_label_latencies={"immediate", "delayed"},
        repo_root=repo_root,
        selected_names=stream_names,
        max_batches=max_batches,
    )


__all__ = [
    "SessionBatch",
    "build_session_batches_from_manifest",
    "build_stream_batches_from_manifest",
]
