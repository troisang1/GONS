"""Experiment runners (GONS FSCIL only)."""

from bcmrnfst.runners.track_a_fscil import (
    TrackAFscilRunResult,
    run_track_a_fscil_experiment,
)
from bcmrnfst.runners.track_a_tiny_admission import (
    TrackATinyAdmissionRunResult,
    run_track_a_tiny_admission_experiment,
)

__all__ = [
    "TrackAFscilRunResult",
    "TrackATinyAdmissionRunResult",
    "run_track_a_fscil_experiment",
    "run_track_a_tiny_admission_experiment",
]
