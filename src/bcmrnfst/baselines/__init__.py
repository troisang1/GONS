"""Baseline placeholders — EXCLUDED from the GONS release.

The competing open-set FSCIL baselines (EWC-NCM, NC-FSCIL, OpenMax, MSP, DOC,
BiC, RFS) are NOT part of this release: GONS is the only method here.

The GONS (gate-OFF) code path imports these symbols only for type annotations
(`BaselineModel` participates in the `TrackAModelState` union) and for the
never-taken baseline branch in the FSCIL runners. The bodies below are
intentionally inert: if any baseline code path were reached, it raises so the
omission is loud rather than silent.

`bcmrnfst.baselines.config` (baseline config *schema* only — no method code) is
retained intact because `bcmrnfst.evaluation.config` imports its dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bcmrnfst.baselines.config import (
    BaselineExperimentConfig,
    BaselineKind,
    load_baseline_experiment_config,
)

_EXCLUDED = (
    "Baseline methods are excluded from the GONS release. "
    "This release contains only GONS (gate-OFF)."
)


@dataclass
class BaselineModel:
    """Inert placeholder so the `TrackAModelState` type union resolves.

    A real BaselineModel is never constructed on the GONS path.
    """

    method: str = "unsupported_baseline"
    seen_labels: tuple[str, ...] = ()
    proto_bank: dict[str, Any] = field(default_factory=dict)


def fit_baseline(*_args: Any, **_kwargs: Any) -> BaselineModel:  # noqa: D401
    raise NotImplementedError(_EXCLUDED)


def predict_dataframe(*_args: Any, **_kwargs: Any) -> Any:
    raise NotImplementedError(_EXCLUDED)


def update_baseline_model(*_args: Any, **_kwargs: Any) -> BaselineModel:
    raise NotImplementedError(_EXCLUDED)


__all__ = [
    "BaselineExperimentConfig",
    "BaselineKind",
    "BaselineModel",
    "fit_baseline",
    "load_baseline_experiment_config",
    "predict_dataframe",
    "update_baseline_model",
]
