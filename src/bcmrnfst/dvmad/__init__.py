"""Spectral primitive for the optional gate mechanism.

Two-tailed eigenvector selection on the generalized scatter eigen-problem,
augmented with a single artificial reference point. Provides a closed-form,
no-gradient anomaly / open-set scoring backbone. This primitive belongs to the
optional gate mechanism, which is disabled in GONS.
"""

from bcmrnfst.dvmad.core import DVMADCore

__all__ = ["DVMADCore"]
