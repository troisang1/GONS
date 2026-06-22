"""Adapter — two-tailed projection for open-set FSCIL (optional gate mechanism).

Stub. Replaces (or augments) `compute_soft_null_projection` in the support /
known-class projection step. The min-tail directions recover the existing
NFST null-space; the max-tail directions add the structural-saturated
directions. Part of the optional gate mechanism, disabled in GONS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bcmrnfst.dvmad import DVMADCore


@dataclass(slots=True)
class TwoTailedProjection:
    """Result bundle for the two-tailed projection swap-in."""

    npd: NDArray[np.float64]
    filtered_eigvals: NDArray[np.float64]
    basepoint: NDArray[np.float32]
    n_low: int
    n_high: int


def compute_two_tailed_projection(
    X: NDArray[np.float64],
    y: NDArray[np.int64] | None = None,
    *,
    eps: float = 0.1,
    artificial_mode: str = "farthest",
) -> TwoTailedProjection:
    """Fit DVM-AD with `mode='both'` and split low / high tail counts.

    `artificial_mode='farthest'` is the FSCIL-tuned default (Theorem D2).
    """
    core = DVMADCore(
        mode="both",
        eps=eps,
        artificial_mode=artificial_mode,
        use_faiss=False,
    ).fit(X, y)

    eigs = core.filtered_eigvals_
    n_low = int((eigs < eps).sum()) if eigs is not None else 0
    n_high = int((eigs > 1.0 - eps).sum()) if eigs is not None else 0

    return TwoTailedProjection(
        npd=core.npd_,
        filtered_eigvals=eigs,
        basepoint=core.basepoint_X_,
        n_low=n_low,
        n_high=n_high,
    )


def two_tailed_score(
    X: NDArray[np.float64],
    projection: TwoTailedProjection,
) -> NDArray[np.float64]:
    """NN distance in the two-tailed discriminant subspace."""
    Z = X @ projection.npd
    return DVMADCore._min_l2_to_set_chunked(Z.astype(np.float32), projection.basepoint)
