"""Eigenvalue-Weighted Mahalanobis (EWM) — optional gate mechanism (disabled in GONS).

Extends the eigen-problem T = D⁻¹ Q_Sᵀ M Q_S by using the eigenvalues
λ_j ∈ [0, 1] as **continuous SNR weights** rather than for binary selection.

Distance: d²_λ(x, μ_q) = Σ_j ω_j · (W_jᵀ (x − μ_q))²
Weight:    ω_j = (1 − λ_j) / (λ_j + δ_floor)  (Fisher SNR)

- λ_j → 0: ω → ∞ — null-space directions dominate (recovers NFST).
- λ_j → 1: ω → 0 — saturated directions ignored.
- Middle: smooth interpolation — **NEW INFO** that the hard-cut discards.

The binary corner case is ω_j = 1[λ_j < ε ∨ λ_j > 1−ε] (binary step).

The DVMADCore returned by `bcmrnfst.dvmad` only stores eigenvectors that
PASSED its mask. To access the full spectrum, we recompute T's full
eigen-decomposition once at fit time and cache it on the model state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from bcmrnfst.state.model import BCMRNFSTModel


@dataclass(slots=True)
class EWMState:
    """Cached full spectrum + class centroids in eigenvalue-weighted space."""

    W_full: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0)))
    eigvals: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    omega: NDArray[np.float64] = field(default_factory=lambda: np.empty(0))
    centroid_phi_per_class: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    delta_floor: float = 0.01
    alpha_mix: float = 1.0
    # PCA-whitening pipeline (v3.5 — feature-extractor adaptation).
    pca_mean: NDArray[np.float64] | None = None
    pca_components: NDArray[np.float64] | None = None  # (out_dim, in_dim)
    pca_explained_var: NDArray[np.float64] | None = None  # (out_dim,)


def _gather_per_class_phi(model: "BCMRNFSTModel") -> dict[str, NDArray[np.float64]]:
    store = getattr(model, "support_store", None) or {}
    out: dict[str, list[NDArray[np.float64]]] = {}
    for bundle in store.values():
        cls = getattr(bundle, "class_label", None)
        if cls is None or bundle.size == 0:
            continue
        out.setdefault(cls, []).append(np.asarray(bundle.phi_bundle, dtype=np.float64))
    return {k: np.vstack(v) for k, v in out.items() if v}


def _pca_whiten_fit(
    X: NDArray[np.float64],
    n_out: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Fit PCA-whitening transform: returns (mean, components, explained_var).

    Whitened transform: Z = (X - mean) @ components.T / sqrt(explained_var).
    """
    mean = X.mean(axis=0)
    Xc = X - mean
    # SVD-based PCA. Use the smaller of (n_out, samples-1, dims).
    n_keep = min(n_out, Xc.shape[0] - 1, Xc.shape[1])
    if n_keep < 1:
        n_keep = 1
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_keep]                                  # (n_out, in_dim)
    explained_var = (S[:n_keep] ** 2) / max(Xc.shape[0] - 1, 1)
    return mean, components, np.maximum(explained_var, 1e-9)


def _pca_whiten_apply(
    X: NDArray[np.float64],
    mean: NDArray[np.float64],
    components: NDArray[np.float64],
    explained_var: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply pre-fitted PCA-whitening to new data."""
    Xc = np.asarray(X, dtype=np.float64) - mean
    Z = Xc @ components.T
    return Z / np.sqrt(explained_var)[None, :]


def fit_ewm(model: "BCMRNFSTModel") -> EWMState:
    """Fit full T spectrum + per-class centroids in φ-space.

    Optional PCA-whitening pre-step (v3.5): when ``ewm_pca_whiten`` config is
    enabled, fit a PCA-whitening transform to the support features first,
    then compute T's spectrum on the whitened features. This unlocks the
    eigenvalue continuum that RFF expansion otherwise destroys.

    Returns empty EWMState when the support_store has < 2 classes.
    """
    state = EWMState()
    by_class = _gather_per_class_phi(model)
    if len(by_class) < 2:
        return state

    # Build int-labeled X, y from support_store.
    label_map: dict[str, int] = {}
    feats: list[NDArray[np.float64]] = []
    labels: list[int] = []
    raw_centroids: dict[str, NDArray[np.float64]] = {}
    for cls, X_q in sorted(by_class.items()):
        if cls not in label_map:
            label_map[cls] = len(label_map)
        feats.append(X_q)
        labels.extend([label_map[cls]] * X_q.shape[0])
        raw_centroids[cls] = X_q.mean(axis=0)
    X_raw = np.vstack(feats)
    y = np.asarray(labels, dtype=np.int64)

    # Optional pre-whitening (PCA or Sw-Mahalanobis).
    pca_mean = pca_components = pca_var = None
    whiten_mode = str(getattr(model.config, "ewm_whiten_mode", "none"))
    if whiten_mode == "pca" or bool(getattr(model.config, "ewm_pca_whiten", False)):
        n_out = int(getattr(model.config, "ewm_pca_dim", 64))
        pca_mean, pca_components, pca_var = _pca_whiten_fit(X_raw, n_out)
        X = _pca_whiten_apply(X_raw, pca_mean, pca_components, pca_var)
        centroids = {cls: _pca_whiten_apply(mu[None, :], pca_mean, pca_components, pca_var)[0]
                     for cls, mu in raw_centroids.items()}
    elif whiten_mode == "sw":
        # Sw-Mahalanobis whitening: multiply by Sw^{-1/2}. Stored as a single
        # linear transform in (mean=0, components=Sw^{-1/2}, var=1) so that
        # _pca_whiten_apply at predict time does the same transform.
        cls_arr = np.unique(y)
        Sw = np.zeros((X_raw.shape[1], X_raw.shape[1]), dtype=np.float64)
        for c in cls_arr:
            Pc = X_raw[y == c]
            mc = Pc.mean(axis=0)
            diff = Pc - mc
            Sw += diff.T @ diff
        Sw /= max(X_raw.shape[0], 1)
        reg = float(getattr(model.config, "ewm_sw_reg", 1e-3))
        Sw_reg = Sw + reg * np.eye(Sw.shape[0])
        ew, Vw = np.linalg.eigh(Sw_reg)
        sw_inv_half = Vw @ np.diag(1.0 / np.sqrt(np.maximum(ew, 1e-9))) @ Vw.T
        pca_mean = np.zeros(X_raw.shape[1], dtype=np.float64)
        pca_components = sw_inv_half.T  # (in_dim, in_dim) — applied as Xc @ comp.T
        pca_var = np.ones(X_raw.shape[1], dtype=np.float64)
        X = X_raw @ sw_inv_half
        centroids = {cls: mu @ sw_inv_half for cls, mu in raw_centroids.items()}
    else:
        X = X_raw
        centroids = raw_centroids
    D = X.shape[1]

    # Augment with single global x_ref (consistent with DVMADCore).
    artificial_mode = str(getattr(model.config, "dvmad_artificial_mode", "farthest"))
    if artificial_mode == "max":
        x_ref = X.max(axis=0)
    elif artificial_mode == "min":
        x_ref = X.min(axis=0)
    else:  # farthest
        mu = X.mean(axis=0)
        idx = int(np.argmax(np.linalg.norm(X - mu, axis=1)))
        x_ref = X[idx]
    X_aug = np.vstack([X, x_ref[None, :]])
    y_aug = np.concatenate([y, [int(y.max()) + 1]])

    # Compute T's full spectrum.
    classes = np.unique(y_aug)
    mean_total = X_aug.mean(axis=0)
    P_W_blocks: list[NDArray[np.float64]] = []
    for c in classes:
        Xc = X_aug[y_aug == c]
        mc = Xc.mean(axis=0)
        P_W_blocks.append((Xc - mc).T)
    P_W = np.hstack(P_W_blocks)
    P_S = (X_aug - mean_total).T
    S_S = P_S @ P_S.T
    eigvals_S, Q_S = np.linalg.eigh(S_S)
    # Floor tiny structural-scatter eigenvalues to avoid divide-by-zero.
    D_floor = max(float(np.median(eigvals_S)) * 1e-6, 1e-12)
    D_pinv_diag = np.where(eigvals_S > D_floor, 1.0 / np.maximum(eigvals_S, D_floor), 0.0)
    M = P_W @ P_W.T
    T = (D_pinv_diag[:, None] * (Q_S.T @ M @ Q_S))
    T = 0.5 * (T + T.T)  # symmetrize numeric drift
    ev, V = np.linalg.eigh(T)
    ev = np.clip(np.real(ev), 0.0, 1.0)
    W_full = Q_S @ V  # (D, D) — full eigenvectors of T in φ-basis

    # Data-driven δ_floor: median of smallest 10% of λ.
    if ev.size:
        n_low = max(int(np.ceil(0.1 * ev.size)), 1)
        delta_floor = float(np.median(np.sort(ev)[:n_low])) + 1e-9
        delta_floor = max(delta_floor, 1e-6)
    else:
        delta_floor = 1e-3

    omega = (1.0 - ev) / (ev + delta_floor)
    # Cap to avoid numerical blow-up in case some λ_j ≈ 0 exactly.
    omega = np.minimum(omega, 1e6)

    state.W_full = np.asarray(W_full, dtype=np.float64)
    state.eigvals = np.asarray(ev, dtype=np.float64)
    state.omega = np.asarray(omega, dtype=np.float64)
    state.centroid_phi_per_class = centroids
    state.delta_floor = delta_floor
    state.alpha_mix = float(getattr(model.config, "ewm_alpha_mix", 1.0))
    state.pca_mean = pca_mean
    state.pca_components = pca_components
    state.pca_explained_var = pca_var
    return state


def _maybe_whiten(state: EWMState, phi_x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Apply state's PCA-whitening to phi_x if it was fit at train time."""
    if state.pca_mean is None or state.pca_components is None or state.pca_explained_var is None:
        return np.asarray(phi_x, dtype=np.float64)
    return _pca_whiten_apply(phi_x, state.pca_mean, state.pca_components, state.pca_explained_var)


def ewm_class_distance(
    state: EWMState,
    phi_x: NDArray[np.float64],
    class_label: str,
) -> NDArray[np.float64] | None:
    """d²_λ(x, μ_q) for one class. Returns (n,) — sqrt'd."""
    if state.W_full.size == 0 or class_label not in state.centroid_phi_per_class:
        return None
    Phi = _maybe_whiten(state, phi_x)
    mu_q = state.centroid_phi_per_class[class_label]
    diff = Phi - mu_q
    Z = diff @ state.W_full
    return np.sqrt(np.maximum(Z * Z @ state.omega, 0.0))


def ewm_score_per_class(
    state: EWMState,
    phi_x: NDArray[np.float64],
) -> tuple[NDArray[np.float64], list[str]]:
    """Return (n_x, n_classes) Fisher-weighted distances + class order."""
    classes = sorted(state.centroid_phi_per_class.keys())
    if not classes:
        return np.zeros((phi_x.shape[0], 0)), []
    cols = []
    for cls in classes:
        cols.append(ewm_class_distance(state, phi_x, cls))
    return np.stack(cols, axis=1), classes
