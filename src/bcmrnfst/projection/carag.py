"""CARAG: Class-Anchored Reference-Augmented Gate.

A mechanism for the optional gate mechanism (disabled in GONS). Strict superset
of the gate-only variant.

Three additions over the gate-only variant:

1. **Per-class anchor rows** — replace the single
   global ``x_ref`` from ``DVMADCore.fit`` with K extra rows, one farthest-
   from-class-mean per known class. Within-class scatter unchanged
   (singleton blocks contribute zero); structural scatter gains a rank-K
   class-aware perturbation. Two-tailed eigenstructure now resolves K
   class-specific anti-prior directions in ONE shared projection.

2. **Unknown-buffer phantom-negative class** (§ 2.2) — fold the existing
   ``unknown_buffer`` (FIFO cap 256) in as one extra phantom-class block
   in the augmented training set. Within-class scatter now sees novel-
   class spread for the first time, tightening the gate's known-vs-novel
   contrast each refresh.

3. **Per-class anchor blend in NCDR ambiguity band** (§ 2.3) — at predict
   time, when NCM-NCDR's top-2 ratio is in the ambiguity band, blend a
   class-q anchor-distance signal ``δ_q(x) = ‖W(φ(x) − a_q)‖`` into the
   class score. β set in closed form (Theorem D6) from calibration std +
   margin statistics; β=0 when calibration < 100 samples per class.

When ``carag_buffer_cap=0`` and the anchor blend is disabled (β=0), CARAG
reduces to v1 exactly. This is the "v1 floor" guarantee — at worst, tie v1.
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
class CARAGState:
    """Anchor + calibration state for CARAG."""

    anchor_phi: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    delta_calib_median: dict[str, float] = field(default_factory=dict)
    delta_calib_std: dict[str, float] = field(default_factory=dict)
    beta: float = 0.0
    n_buffer_used: int = 0


def _gather_per_class_phi(model: "BCMRNFSTModel") -> dict[str, NDArray[np.float64]]:
    store = getattr(model, "support_store", None) or {}
    out: dict[str, list[NDArray[np.float64]]] = {}
    for bundle in store.values():
        cls = getattr(bundle, "class_label", None)
        if cls is None or bundle.size == 0:
            continue
        out.setdefault(cls, []).append(np.asarray(bundle.phi_bundle, dtype=np.float64))
    return {k: np.vstack(v) for k, v in out.items() if v}


def _per_class_anchors(
    by_class: dict[str, NDArray[np.float64]],
    min_support: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], dict[str, NDArray[np.float64]]]:
    """Return (anchors_matrix, fresh_int_labels, per_class_anchor_phi).

    Two anchors per class:
      * ``a_q^far`` — farthest-from-class-mean (used as DVMADCore training row)
      * ``a_q^cen`` — class centroid (used at predict time for δ_q signal)

    The "far" row is added to the augmented training matrix to inject the
    class-conditional anti-prior direction into structural scatter. The "cen"
    row is what feeds the predict-time blend so
    that small ``δ_q(x)`` correctly indicates class-q membership (closed-form
    β stays positive). Returning the centroid as ``anchor_phi[cls]`` keeps
    the public API of ``per_class_anchor_signal`` unchanged.

    Skip classes with fewer than ``min_support`` support points.
    """
    anchor_phi: dict[str, NDArray[np.float64]] = {}
    rows_far: list[NDArray[np.float64]] = []
    for cls, X_q in by_class.items():
        if X_q.shape[0] < min_support:
            continue
        mu_q = X_q.mean(axis=0)
        idx = int(np.argmax(np.linalg.norm(X_q - mu_q, axis=1)))
        rows_far.append(X_q[idx])
        anchor_phi[cls] = mu_q  # centroid for predict-time blend
    if not rows_far:
        d = next(iter(by_class.values())).shape[1] if by_class else 0
        return np.empty((0, d), dtype=np.float64), np.empty(0, dtype=np.int64), anchor_phi
    A = np.vstack(rows_far)
    yA = np.arange(A.shape[0], dtype=np.int64)
    return A, yA, anchor_phi


def _gather_buffer_phi(
    model: "BCMRNFSTModel",
    cap: int,
) -> NDArray[np.float64]:
    if cap <= 0:
        return np.empty((0, 0), dtype=np.float64)
    buf = list(getattr(model, "unknown_buffer", []) or [])
    if not buf:
        return np.empty((0, 0), dtype=np.float64)
    buf = buf[-cap:]
    feats: list[NDArray[np.float64]] = []
    for r in buf:
        if getattr(r, "phi_x", None) is None:
            continue
        feats.append(np.asarray(r.phi_x, dtype=np.float64))
    if not feats:
        return np.empty((0, 0), dtype=np.float64)
    return np.vstack(feats)


def fit_carag_gate(
    model: "BCMRNFSTModel",
) -> tuple[DVMADCore | None, float | None, CARAGState]:
    """Fit the CARAG-augmented DVMADCore. Returns (core, tau_fallback, state).

    Drop-in for ``fit_dvmad_gate``. The DVMADCore is fitted on a multi-class
    training matrix (K real classes + K anchor singletons + optional
    phantom-negative buffer block). When the support_store has < 2 known
    classes, falls back to (None, None, empty state).
    """
    state = CARAGState()
    by_class = _gather_per_class_phi(model)
    if len(by_class) < 2:
        return None, None, state

    eps = float(getattr(model.config, "dvmad_eps", 0.10))
    artificial_mode = str(getattr(model.config, "dvmad_artificial_mode", "farthest"))
    quantile = float(getattr(model.config, "dvmad_quantile", 0.99))
    gate_mode = str(getattr(model.config, "dvmad_gate_mode", "both"))
    buffer_cap = int(getattr(model.config, "carag_buffer_cap", 256))
    min_anchor = int(getattr(model.config, "carag_min_anchor_support", 10))

    # Build base (X, y) from support_store with int labels.
    label_map: dict[str, int] = {}
    feats: list[NDArray[np.float64]] = []
    labels: list[int] = []
    for cls, X_q in sorted(by_class.items()):
        if cls not in label_map:
            label_map[cls] = len(label_map)
        feats.append(X_q)
        labels.extend([label_map[cls]] * X_q.shape[0])
    X = np.vstack(feats)
    y = np.asarray(labels, dtype=np.int64)
    K = len(label_map)

    # 2.1 — class-conditional anchors.
    A, _, anchor_phi = _per_class_anchors(by_class, min_anchor)
    if A.shape[0] > 0:
        anchor_labels = np.arange(K, K + A.shape[0], dtype=np.int64)
        X = np.vstack([X, A])
        y = np.concatenate([y, anchor_labels])
        state.anchor_phi = anchor_phi

    # 2.2 — unknown-buffer phantom-negative anchors.
    B = _gather_buffer_phi(model, buffer_cap)
    if B.size > 0 and B.shape[1] == X.shape[1]:
        unk_label = int(y.max()) + 1
        X = np.vstack([X, B])
        y = np.concatenate([y, np.full(B.shape[0], unk_label, dtype=np.int64)])
        state.n_buffer_used = int(B.shape[0])

    # Single shared DVMADCore — same surface as v1.
    core = DVMADCore(
        mode=gate_mode,
        eps=eps,
        artificial_mode=artificial_mode,
        use_faiss=False,
    ).fit(X, y)

    # Train-time fallback Q-quantile (the proper Q is set later from cal data).
    train_scores = np.asarray(core.predict(X[: sum(X_q.shape[0] for X_q in by_class.values())]), dtype=np.float64)
    tau_fallback = float(np.quantile(train_scores, quantile))
    return core, tau_fallback, state


def calibrate_carag_beta(
    model: "BCMRNFSTModel",
    state: CARAGState,
    cal_phi: NDArray[np.float64],
    cal_labels: NDArray[np.object_],
    ncm_distance_per_class: NDArray[np.float64] | None = None,
    class_order: list[str] | None = None,
) -> CARAGState:
    """Compute β (Theorem D6) + per-class δ-calibration stats.

    Returns the updated state. β is clipped to [0, 1] and zeroed when
    calibration data per class < 100 samples (Failure mode F1 mitigation).
    """
    core: DVMADCore | None = getattr(model, "dvmad_core", None)
    if core is None or not state.anchor_phi:
        return state

    # δ_q on calibration points, per class
    delta_per_class: dict[str, NDArray[np.float64]] = {}
    npd = core.npd_
    Z_cal = cal_phi @ npd
    for cls, a in state.anchor_phi.items():
        Z_a = (a @ npd).reshape(1, -1)
        delta_per_class[cls] = np.linalg.norm(Z_cal - Z_a, axis=1)

    # Per-class median + std on calibration samples whose true label is q.
    for cls in state.anchor_phi:
        mask = np.asarray([str(c) == cls for c in cal_labels], dtype=bool)
        if mask.sum() < 4:
            state.delta_calib_median[cls] = float(np.median(delta_per_class[cls]))
            state.delta_calib_std[cls] = float(np.std(delta_per_class[cls]) + 1e-9)
            continue
        d_self = delta_per_class[cls][mask]
        state.delta_calib_median[cls] = float(np.median(d_self))
        state.delta_calib_std[cls] = float(np.std(d_self) + 1e-9)

    # Closed-form β (Theorem D6).
    if (
        ncm_distance_per_class is None
        or class_order is None
        or len(class_order) != ncm_distance_per_class.shape[1]
    ):
        state.beta = 0.0
        return state

    cls_to_idx = {c: i for i, c in enumerate(class_order)}
    ncm_correct: list[float] = []
    ncm_other: list[float] = []
    delta_correct: list[float] = []
    delta_other: list[float] = []
    for i, lbl in enumerate(cal_labels):
        c = str(lbl)
        if c not in cls_to_idx or c not in state.anchor_phi:
            continue
        q_idx = cls_to_idx[c]
        ncm_correct.append(float(ncm_distance_per_class[i, q_idx]))
        # next-best NCM
        row = ncm_distance_per_class[i].copy()
        row[q_idx] = np.inf
        ncm_other.append(float(np.min(row)))
        delta_correct.append(float(delta_per_class[c][i]))
        # next-best δ over other classes
        other_d = [delta_per_class[c2][i] for c2 in state.anchor_phi if c2 != c]
        if other_d:
            delta_other.append(float(np.min(other_d)))

    if len(ncm_correct) < 100:
        state.beta = 0.0
        return state

    sig_ncm2 = float(np.var(ncm_correct)) + 1e-9
    sig_delta2 = float(np.var(delta_correct)) + 1e-9
    delta_margin = float(np.median(np.asarray(delta_other) - np.asarray(delta_correct)))
    ncm_margin = float(np.median(np.asarray(ncm_other) - np.asarray(ncm_correct)))
    if delta_margin <= 0 or ncm_margin <= 0:
        state.beta = 0.0
        return state

    num = sig_ncm2 * delta_margin
    den = sig_ncm2 * delta_margin + sig_delta2 * ncm_margin
    state.beta = float(np.clip(num / max(den, 1e-12), 0.0, 1.0))
    return state


def per_class_anchor_signal(
    core: DVMADCore,
    phi_x: NDArray[np.float64],
    anchor_phi: dict[str, NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    """δ_q(x) for each class. Returns dict class → (n,) distances."""
    if core.npd_ is None or not anchor_phi:
        return {}
    npd = core.npd_
    Z_x = np.asarray(phi_x, dtype=np.float64) @ npd
    out: dict[str, NDArray[np.float64]] = {}
    for cls, a in anchor_phi.items():
        Z_a = (np.asarray(a, dtype=np.float64) @ npd).reshape(1, -1)
        out[cls] = np.linalg.norm(Z_x - Z_a, axis=1)
    return out
