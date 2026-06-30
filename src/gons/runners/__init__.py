"""Experiment runners (GONS FSCIL only)."""

from gons.runners.fscil import (
    FscilRunResult,
    run_fscil_experiment,
)
from gons.runners.tiny_admission import (
    TinyAdmissionRunResult,
    run_tiny_admission_experiment,
)

__all__ = [
    "FscilRunResult",
    "TinyAdmissionRunResult",
    "run_fscil_experiment",
    "run_tiny_admission_experiment",
]
