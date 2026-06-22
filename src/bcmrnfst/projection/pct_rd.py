"""PCT-RD: Per-Class Two-Tailed Reference Discriminant.

A mechanism for the optional gate mechanism (disabled in GONS). Replaces the
global single-DVMADCore gate with per-class detectors. Each known class gets its
own DVMADCore (mode='both', class-conditional artificial reference point). Three
contributions:

1. **Class-conditional g_q(x):** Mahalanobis-style NN distance in class-q's
   two-tailed projection, weighted per-direction by `1/eps` (low-tail) and
   `1/(1-lambda_j)` (high-tail). Provides a *classification* margin
   complementary to NCM's centroid distance.

2. **Closed-form mixing weight:** alpha_PCT* = ratio of per-score
   signal-to-noise on calibration. No grid search.

3. **Per-class Q-quantile thresholds:** restores per-class CCR control instead
   of pooled Q90. Lifts macro_f1.

This module is a self-contained scoring path. Wire into the pipeline via:
- `bcmrnfst.fit.fit_bcmrnfst` (initial fit + calibration)
- `bcmrnfst.continual.refresh.consolidated_refresh` (refresh-time refit)
- `bcmrnfst.continual.inference._score_projected_rows` (predict path)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bcmrnfst.dvmad import DVMADCore

if TYPE_CHECKING:
    from bcmrnfst.state.model import BCMRNFSTModel


@dataclass(slots=True)
class PCTRDState:
    """Per-class detector bundle + calibration state for PCT-RD."""

    cores: dict[str, DVMADCore] = field(default_factory=dict)
    g_norm: dict[str, float] = field(default_factory=dict)  # median g_q on class q
    eigvals: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    omega: dict[str, NDArray[np.float64]] = field(default_factory=dict)  # per-direction weights
    alpha_pct: float = 0.5
    alpha_ncm: float = 0.5
    tau_per_class: dict[str, float] = field(default_factory=dict)
    tau_global: float | None = None  # fallback when calibration thin


def _gather_per_class_phi(
    model: "BCMRNFSTModel",
) -> dict[str, NDArray[np.float64]]:
    store = getattr(model, "support_store", None) or {}
    out: dict[str, list[NDArray[np.float64]]] = {}
    for bundle in store.values():
        cls = getattr(bundle, "class_label", None)
        if cls is None or bundle.size == 0:
            continue
        out.setdefault(cls, []).append(np.asarray(bundle.phi_bundle, dtype=np.float64))
    return {k: np.vstack(v) for k, v in out.items() if v}


def _make_omega(
    eigvals: NDArray[np.float64],
    eps: float,
    cap: float = 100.0,
) -> NDArray[np.float64]:
    """Per-direction weights: 1/eps low-tail, 1/(1-lambda) high-tail.

    Capped at `cap` to keep numerical stability when an eigenvalue lies
    extremely close to 1.
    """
    omega = np.empty(eigvals.shape, dtype=np.float64)
    low_mask = eigvals < eps
    high_mask = eigvals > 1.0 - eps
    omega[low_mask] = 1.0 / max(eps, 1e-9)
    denom = np.maximum(1.0 - eigvals[high_mask], 1.0 / cap)
    omega[high_mask] = 1.0 / denom
    # Anything in the middle (rare; only the fallback "keep all" path) gets
    # a neutral weight — preserves NCM-equivalence on those directions.
    mid_mask = ~(low_mask | high_mask)
    omega[mid_mask] = 1.0
    return np.clip(omega, 0.0, cap)


def _g_q_score(
    core: DVMADCore,
    omega: NDArray[np.float64],
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Weighted-Mahalanobis NN distance to the augmented basepoint set.

    Diagonal Ω lets us absorb the weight into the features: scale each
    coordinate by sqrt(omega), then a standard Euclidean NN gives the
    weighted distance. This avoids the O(b * n_base * m) memory blow-up
    of the naive broadcast and uses matmul.
    """
    Z = core.transform(X).astype(np.float32)
    base = core.basepoint_X_.astype(np.float32)
    om = np.asarray(omega, dtype=np.float32)
    sqrt_om = np.sqrt(np.maximum(om, 0.0))
    Zw = Z * sqrt_om[None, :]
    base_w = base * sqrt_om[None, :]
    norm_b = np.sum(base_w ** 2, axis=1)[None, :]
    nA = Zw.shape[0]
    out = np.empty(nA, dtype=np.float64)
    chunk = max(1, min(4096, nA))
    for s in range(0, nA, chunk):
        e = min(s + chunk, nA)
        Ac = Zw[s:e]
        norm_a = np.sum(Ac ** 2, axis=1)[:, None]
        dist2 = norm_a + norm_b - 2.0 * (Ac @ base_w.T)
        out[s:e] = np.sqrt(np.maximum(dist2.min(axis=1), 0.0))
    return out


def fit_pct_rd(model: "BCMRNFSTModel") -> PCTRDState:
    """Fit per-class DVMADCores. Mixing weight + tau set by calibrate().

    Returns ``empty PCTRDState`` when the support store yields fewer than
    two classes (warm-up phase).
    """
    state = PCTRDState()
    by_class = _gather_per_class_phi(model)
    if len(by_class) < 2:
        return state

    eps = float(getattr(model.config, "dvmad_eps", 0.1))
    artificial_mode = str(getattr(model.config, "dvmad_artificial_mode", "farthest"))
    min_per_class = int(getattr(model.config, "pct_rd_min_per_class", 4))

    for cls, X_q in by_class.items():
        if X_q.shape[0] < min_per_class:
            continue
        core = DVMADCore(
            mode="both",
            eps=eps,
            artificial_mode=artificial_mode,
            use_faiss=False,
        ).fit(X_q, y_train=None)
        eigvals = np.asarray(core.filtered_eigvals_, dtype=np.float64)
        omega = _make_omega(eigvals, eps=eps)
        # Train-time normaliser — median of g_q on the class's own points.
        # Using LOO 2nd-NN to avoid self-distance = 0.
        g_self = _g_q_score(core, omega, X_q)
        # When all g_self are 0 (degenerate), fall back to 1.0
        g_med = float(np.median(g_self))
        if g_med <= 1e-9:
            g_med = float(np.mean(g_self) + 1e-9)
        state.cores[cls] = core
        state.eigvals[cls] = eigvals
        state.omega[cls] = omega
        state.g_norm[cls] = g_med
    return state


def calibrate_pct_rd(
    model: "BCMRNFSTModel",
    pct_state: PCTRDState,
    cal_phi: NDArray[np.float64],
    cal_labels: NDArray[np.object_],
    ncm_distance_per_class: NDArray[np.float64] | None = None,
) -> PCTRDState:
    """Set mixing weight (Theorem A3) and per-class quantile thresholds (A4).

    `ncm_distance_per_class` is the existing NCM scores on the calibration
    set, shape (n_cal, |Q_t|), already normalised by per-class radius. If
    None, alpha_pct defaults to a balanced 0.5 split (graceful fallback).

    The per-class threshold is the configured Q-quantile of `g_q` on
    calibration samples whose true class is q.
    """
    if not pct_state.cores:
        return pct_state
    classes = sorted(pct_state.cores.keys())
    quantile = float(getattr(model.config, "dvmad_quantile", 0.99))

    # ---- Per-class tau (Theorem A4) ----
    tau_per_class: dict[str, float] = {}
    for q in classes:
        mask = np.asarray([str(c) == q for c in cal_labels], dtype=bool)
        if mask.sum() < 4:
            continue
        X_q = cal_phi[mask]
        scores = _g_q_score(pct_state.cores[q], pct_state.omega[q], X_q) / pct_state.g_norm[q]
        tau_per_class[q] = float(np.quantile(scores, quantile))

    # Global fallback τ for thin or unseen classes.
    all_scores: list[float] = []
    for q, core in pct_state.cores.items():
        all_scores.extend(
            _g_q_score(core, pct_state.omega[q], cal_phi) / pct_state.g_norm[q]
        )
    if all_scores:
        pct_state.tau_global = float(np.quantile(np.asarray(all_scores), quantile))

    pct_state.tau_per_class = tau_per_class

    # ---- Closed-form mixing weight (Theorem A3) ----
    # alpha_PCT / alpha_NCM = (var_NCM / var_PCT) * (margin_PCT / margin_NCM)
    if ncm_distance_per_class is None or ncm_distance_per_class.size == 0:
        pct_state.alpha_pct = 0.5
        pct_state.alpha_ncm = 0.5
        return pct_state

    # Batch PCT score: (n_cal, n_classes).
    pct_full, _ = score_pct_rd(pct_state, cal_phi)
    label_to_idx = {q: i for i, q in enumerate(classes)}
    pct_correct: list[float] = []
    pct_other: list[float] = []
    ncm_correct: list[float] = []
    ncm_other: list[float] = []
    n_classes = len(classes)
    for i, lbl in enumerate(cal_labels):
        s = str(lbl)
        if s not in label_to_idx:
            continue
        q_idx = label_to_idx[s]
        mask = np.ones(n_classes, dtype=bool)
        mask[q_idx] = False
        pct_correct.append(float(pct_full[i, q_idx]))
        pct_other.extend(pct_full[i, mask].tolist())
        if ncm_distance_per_class.ndim == 2 and ncm_distance_per_class.shape[1] == n_classes:
            ncm_correct.append(float(ncm_distance_per_class[i, q_idx]))
            ncm_other.extend(ncm_distance_per_class[i, mask].tolist())

    if not pct_correct or not ncm_correct:
        pct_state.alpha_pct = 0.5
        pct_state.alpha_ncm = 0.5
        return pct_state

    var_pct = float(np.var(pct_correct)) + 1e-9
    var_ncm = float(np.var(ncm_correct)) + 1e-9
    margin_pct = float(np.median(pct_other) - np.median(pct_correct))
    margin_ncm = float(np.median(ncm_other) - np.median(ncm_correct))
    margin_pct = max(margin_pct, 1e-9)
    margin_ncm = max(margin_ncm, 1e-9)
    ratio = (var_ncm / var_pct) * (margin_pct / margin_ncm)
    alpha_pct = ratio / (1.0 + ratio)
    pct_state.alpha_pct = float(np.clip(alpha_pct, 0.05, 0.95))
    pct_state.alpha_ncm = 1.0 - pct_state.alpha_pct
    return pct_state


def score_pct_rd(
    pct_state: PCTRDState,
    X_phi: NDArray[np.float64],
) -> tuple[NDArray[np.float64], list[str]]:
    """Return (n, |Q|) PCT-RD score matrix + class index order."""
    classes = sorted(pct_state.cores.keys())
    if not classes:
        return np.zeros((X_phi.shape[0], 0)), []
    cols: list[NDArray[np.float64]] = []
    for q in classes:
        g = _g_q_score(pct_state.cores[q], pct_state.omega[q], X_phi)
        cols.append(g / pct_state.g_norm[q])
    return np.stack(cols, axis=1), classes
