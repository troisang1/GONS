"""Dispatch experiment configs to the GONS FSCIL runner.

This release supports only the `track_a_fscil` task: the GONS (gate-OFF) path.
The competing baselines are evaluated elsewhere and are not part of this release.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from bcmrnfst.runners.track_a_fscil import run_track_a_fscil_experiment
from bcmrnfst.runners.track_a_tiny_admission import run_track_a_tiny_admission_experiment
from bcmrnfst.runtime import find_repo_root

KNOWN_TASKS = {"track_a_fscil", "track_a_tiny_admission"}


@dataclass(frozen=True, slots=True)
class ExperimentRunResult:
    """Generic result returned by CLI-dispatched experiment runs."""

    method: str
    metrics: dict[str, object]
    run_dir: Path
    run_id: str


def detect_experiment_kind(
    config_path: Path,
    *,
    repo_root: Path | None = None,
) -> str:
    """Inspect a YAML config and choose the matching experiment runner."""

    root = repo_root or find_repo_root()
    resolved_path = config_path if config_path.is_absolute() else root / config_path
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {resolved_path}")

    task = payload.get("task")
    if isinstance(task, str) and task in KNOWN_TASKS:
        return task
    raise ValueError(
        f"Unsupported task {task!r} in {resolved_path}. "
        "The GONS release supports only "
        + ", ".join(sorted(KNOWN_TASKS))
        + ". Baselines and the optional gate mechanism are not part of this release."
    )


def run_experiment_config(
    config_path: Path,
    *,
    run_id: str | None = None,
    repo_root: Path | None = None,
    seed: int | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
) -> ExperimentRunResult:
    """Dispatch one config to the GONS FSCIL runner."""

    kind = detect_experiment_kind(config_path, repo_root=repo_root)
    if kind == "track_a_fscil":
        result = run_track_a_fscil_experiment(
            config_path,
            run_id=run_id,
            repo_root=repo_root,
            seed=seed,
            progress_callback=progress_callback,
        )
        return ExperimentRunResult(
            method=result.method,
            metrics=result.metrics,
            run_dir=result.run_dir,
            run_id=result.run_id,
        )
    track_a_result = run_track_a_tiny_admission_experiment(
        config_path,
        run_id=run_id,
        repo_root=repo_root,
        seed=seed,
        progress_callback=progress_callback,
    )
    return ExperimentRunResult(
        method=track_a_result.method,
        metrics=track_a_result.metrics,
        run_dir=track_a_result.run_dir,
        run_id=track_a_result.run_id,
    )


__all__ = [
    "ExperimentRunResult",
    "detect_experiment_kind",
    "run_experiment_config",
]
