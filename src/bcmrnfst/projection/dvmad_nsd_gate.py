"""NSD-Gate — Null-Space-Duality re-expression of the spectral detector.

Part of the optional gate mechanism (disabled in GONS). Re-expresses the
Sw-whitened Fisher-omega-weighted EWM gate (``dvmad_ewm.py``
``ewm_score_per_class`` min-over-classes, ``whiten_mode='sw'``,
``dvmad_gate_mode='both'``) as an NFST null-subspace score in a derived
non-uniform Fisher metric (Omega_N) plus a derived off-null residual (Omega_H).

Key facts:
  * The gate scores
        g_EWM(x) = min_q sqrt( Sum_{j=1..D} w_j (W_j^T (phi_w(x) - mu_{w,q}))^2 )
    over the FULL whitened eigenbasis W_full of T, weights w_j = (1-lam_j)/(lam_j+delta).
  * NSD-Gate splits the spectrum into N = {j: lam_j < eps} (null block, span = V,
    r = K-1) and H = {j: lam_j > 1-eps} (saturated block, span = V_perp), and scores
        g_NSD(x)^2 = ||V^T delta_phi||^2_{Omega_N} + psi * ||(I-VV^T) delta_phi||^2_{Omega_H}
    Omega_N = diag(w_j)_{j in N} (NON-uniform; measured kappa_N ~ 445), Omega_H likewise.
  * psi = 1 (two-tail) == the both-tail EWM gate, bit-equal when N U H exhausts
    the spectrum (binary saturation at K<D).
  * psi = 0 (ablation only) drops the off-null Omega_H tail — the Omega_N-weighted
    null-subspace gate (not the identity-metric d_*).

This module computes its score via the package code path: it reuses the EWM
full-spectrum cache (W_full, eigvals, omega from ``fit_ewm``) and splits it into
N/H index sets. It does **NOT** import or construct ``DVMADCore`` — the module
``dvmad/core.py`` stays untouched.  ``compute_soft_null_projection`` is used only
as a validation reference (V == W_full[:, N] up to basis), never in the live
score path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from bcmrnfst.projection.dvmad_ewm import EWMState, _maybe_whiten, fit_ewm

# Deployed-gate constants (read off dvmad_ewm.py, Lemma B0').
OMEGA_CAP = 1e6  # dvmad_ewm.py:200 hard clip on omega
DEFAULT_EPS = 1e-3  # null/saturated split tolerance on lam_j in [0,1]


@dataclass(slots=True)
class NSDGateState:
    """Cached NSD-Gate fit: null/saturated split of the EWM full spectrum + tau."""

    # Inherited EWM spectrum (full eigenbasis + Fisher weights), reused verbatim.
    ewm: EWMState = field(default_factory=EWMState)
    # Index sets of the null (N) and saturated (H) blocks over the full spectrum.
    null_idx: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    sat_idx: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    mid_idx: NDArray[np.int64] = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    psi: int = 1  # default (two-tail); 0 = ablation only.
    eps_split: float = DEFAULT_EPS
    tau: float = 0.0
    # Derived diagnostics (Lemma B0' / PF-B0).
    kappa_n: float = 1.0  # condition number of Omega_N (max/min over N).
    n_capped: int = 0     # #{j: omega_j == OMEGA_CAP}.
    delta_floor: float = 0.0


def _split_spectrum(
    eigvals: NDArray[np.float64], eps: float
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Split T's spectrum into null (lam<eps), saturated (lam>1-eps), mid (rest)."""
    lam = np.asarray(eigvals, dtype=np.float64)
    null_idx = np.flatnonzero(lam < eps).astype(np.int64)
    sat_idx = np.flatnonzero(lam > 1.0 - eps).astype(np.int64)
    mid_mask = ~np.isin(np.arange(lam.size), np.concatenate([null_idx, sat_idx]))
    mid_idx = np.flatnonzero(mid_mask).astype(np.int64)
    return null_idx, sat_idx, mid_idx


def fit_nsd_gate(model: "object") -> NSDGateState:
    """Fit the NSD-Gate by re-using the EWM full spectrum + splitting it N/H.

    The gate path computes everything from OUR EWM full-spectrum cache (W_full,
    eigvals, omega) — it does NOT construct a DVMADCore.  Returns an empty
    NSDGateState when the support_store has < 2 classes.
    """
    state = NSDGateState()
    ewm = fit_ewm(model)  # OUR full-spectrum fit; no DVMADCore constructed.
    state.ewm = ewm
    if ewm.W_full.size == 0:
        return state

    psi = 1 if bool(getattr(model.config, "nsd_keep_residual", True)) else 0
    eps = float(getattr(model.config, "nsd_eps_split", DEFAULT_EPS))
    state.psi = psi
    state.eps_split = eps
    state.delta_floor = float(ewm.delta_floor)

    null_idx, sat_idx, mid_idx = _split_spectrum(ewm.eigvals, eps)
    state.null_idx = null_idx
    state.sat_idx = sat_idx
    state.mid_idx = mid_idx

    # PF-B0 diagnostics (Lemma B0'): Omega_N non-uniform + cap binding.
    omega = np.asarray(ewm.omega, dtype=np.float64)
    state.n_capped = int(np.count_nonzero(omega >= OMEGA_CAP))
    if null_idx.size:
        wn = omega[null_idx]
        wn_min = float(np.min(wn))
        wn_max = float(np.max(wn))
        state.kappa_n = wn_max / wn_min if wn_min > 0 else float("inf")
    else:
        state.kappa_n = 1.0

    # Calibration threshold tau from g_NSD on the calibration features, when present.
    X_calib = getattr(model, "_nsd_calib_phi", None)
    if X_calib is not None and np.asarray(X_calib).size:
        scores = nsd_gate_score(state, np.asarray(X_calib, dtype=np.float64))
        if scores.size:
            q = float(getattr(model.config, "dvmad_quantile", 0.95))
            state.tau = float(np.quantile(scores, q))
    return state


def _nsd_class_score_sq(
    state: NSDGateState,
    phi_w: NDArray[np.float64],
    mu_w_q: NDArray[np.float64],
) -> NDArray[np.float64]:
    """g_NSD(x)^2 for one class (whitened): Omega_N null term + psi*Omega_H residual.

    Bit-faithful re-expression of the deployed g_EWM^2 for that class: when
    psi=1 and N U H exhaust the spectrum (binary K<D), the sum
        Sum_{j in N} w_j (W_j^T d)^2 + Sum_{j in H} w_j (W_j^T d)^2
    equals the deployed Sum_{j} w_j (W_j^T d)^2 EXACTLY (same W_full, same omega).
    """
    ewm = state.ewm
    diff = np.asarray(phi_w, dtype=np.float64) - np.asarray(mu_w_q, dtype=np.float64)
    Z = diff @ ewm.W_full  # (n, D) projection onto the full eigenbasis.
    Zsq = Z * Z
    omega = np.asarray(ewm.omega, dtype=np.float64)
    n = Zsq.shape[0]

    # NULL TERM: ||V^T d||^2_{Omega_N} = Sum_{j in N} w_j (W_j^T d)^2.
    if state.null_idx.size:
        out = Zsq[:, state.null_idx] @ omega[state.null_idx]
    else:
        out = np.zeros(n, dtype=np.float64)

    # OFF-NULL RESIDUAL (psi-gated): Sum_{j in H} w_j (W_j^T d)^2, derived Omega_H.
    if state.psi and state.sat_idx.size:
        out = out + Zsq[:, state.sat_idx] @ omega[state.sat_idx]

    # MID-MASS (finite-eps slack, Thm B1' status note): when psi=1 keep it inside
    # the residual so psi=1 reproduces the FULL deployed sum bit-exactly even if a
    # few directions fall in (eps, 1-eps).  When psi=0 it is dropped with the tail.
    if state.psi and state.mid_idx.size:
        out = out + Zsq[:, state.mid_idx] @ omega[state.mid_idx]

    return np.asarray(out, dtype=np.float64)


def nsd_gate_score(
    state: NSDGateState,
    phi_x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """min-over-classes g_NSD(x) (sqrt'd) — the rejection score.

    Mirrors the deployed gate's min-over-classes reduction
    (``ewm_score_per_class(...).min(axis=1)``).
    """
    ewm = state.ewm
    if ewm.W_full.size == 0 or not ewm.centroid_phi_per_class:
        return np.zeros(np.asarray(phi_x).shape[0], dtype=np.float64)
    phi_w = _maybe_whiten(ewm, np.asarray(phi_x, dtype=np.float64))
    classes = sorted(ewm.centroid_phi_per_class.keys())
    cols = []
    for cls in classes:
        mu_w_q = ewm.centroid_phi_per_class[cls]
        cols.append(np.sqrt(np.maximum(_nsd_class_score_sq(state, phi_w, mu_w_q), 0.0)))
    stacked = np.stack(cols, axis=1)  # (n, n_classes)
    return stacked.min(axis=1)


def nsd_gate_reject(state: NSDGateState, phi_x: NDArray[np.float64]) -> NDArray[np.bool_]:
    """reject(x) <-> g_NSD(x) > tau."""
    return nsd_gate_score(state, phi_x) > state.tau


# --- Ablation / pre-flight helpers (NOT in the shipped score path) -------------

def nsd_residual_term(
    state: NSDGateState,
    phi_x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """e_w(x)^2 = ||(I-VV^T) delta_phi_{q*}||^2_{Omega_H} at the argmin-null class.

    PF-B2 channel: the Omega_H off-null residual energy alone (psi-independent),
    evaluated at the class minimizing the null term.  For diagnostics only.
    """
    ewm = state.ewm
    if ewm.W_full.size == 0 or not ewm.centroid_phi_per_class:
        return np.zeros(np.asarray(phi_x).shape[0], dtype=np.float64)
    phi_w = _maybe_whiten(ewm, np.asarray(phi_x, dtype=np.float64))
    classes = sorted(ewm.centroid_phi_per_class.keys())
    omega = np.asarray(ewm.omega, dtype=np.float64)
    h_idx = np.concatenate([state.sat_idx, state.mid_idx]) if state.mid_idx.size else state.sat_idx

    null_sq_cols = []
    res_sq_cols = []
    for cls in classes:
        diff = phi_w - ewm.centroid_phi_per_class[cls]
        Z = diff @ ewm.W_full
        Zsq = Z * Z
        null_sq_cols.append(
            Zsq[:, state.null_idx] @ omega[state.null_idx]
            if state.null_idx.size else np.zeros(phi_w.shape[0])
        )
        res_sq_cols.append(
            Zsq[:, h_idx] @ omega[h_idx] if h_idx.size else np.zeros(phi_w.shape[0])
        )
    null_stack = np.stack(null_sq_cols, axis=1)
    res_stack = np.stack(res_sq_cols, axis=1)
    qstar = np.argmin(null_stack, axis=1)
    return res_stack[np.arange(res_stack.shape[0]), qstar]
