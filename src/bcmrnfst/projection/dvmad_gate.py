"""Optional novelty gate mechanism (disabled in GONS).

Sits beside the W_proj / NCM / prototype scoring without replacing any of
it. The gate is a separate `DVMADCore` fitted on the
labeled support_store features, configured in `mode='both'`. Its purpose
is to add an extra OPEN-SET REJECTION signal: if a sample's NN distance
in the gate's two-tailed projection exceeds a closed-form Q90 threshold
(Theorem D3), the sample is rejected as unknown — even when the
classifier path admits it.

Combined rejection logic at predict time:
    reject  iff  classifier_says_reject  OR  gate_says_reject

This keeps closed-set CCR untouched (gate never overrides admit→reject in
the closed-set path) and aims to lift TUR / OS-HM by catching novel-class
samples the prototype-NCM path would have absorbed into the nearest base
class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from bcmrnfst.dvmad import DVMADCore

if TYPE_CHECKING:
    from bcmrnfst.state.model import BCMRNFSTModel


def _gather_support_phi(model: "BCMRNFSTModel") -> tuple[NDArray[np.float64], NDArray[np.int64]] | None:
    store = getattr(model, "support_store", None) or {}
    feats: list[NDArray[np.float64]] = []
    labels: list[int] = []
    label_lookup: dict[str, int] = {}
    for bundle in store.values():
        cls = getattr(bundle, "class_label", None)
        if cls is None or bundle.size == 0:
            continue
        if cls not in label_lookup:
            label_lookup[cls] = len(label_lookup)
        feats.append(np.asarray(bundle.phi_bundle, dtype=np.float64))
        labels.extend([label_lookup[cls]] * bundle.size)
    if len(label_lookup) < 2 or not feats:
        return None
    return np.vstack(feats), np.asarray(labels, dtype=np.int64)


def fit_dvmad_gate(model: "BCMRNFSTModel") -> tuple[DVMADCore | None, float | None]:
    """Fit the gate detector. Returns (core, fallback_tau).

    The fallback tau is the Q-quantile of the *training* (support) scores —
    typically too tight, but useful when no calibration data is available.
    The proper tau gets recomputed by :func:`calibrate_dvmad_tau` against
    held-out calibration features; that path is the one wired into the
    main fit/refresh pipeline.
    """
    bundle = _gather_support_phi(model)
    if bundle is None:
        return None, None
    X, y = bundle

    eps = float(getattr(model.config, "dvmad_eps", 0.1))
    artificial_mode = str(getattr(model.config, "dvmad_artificial_mode", "farthest"))
    quantile = float(getattr(model.config, "dvmad_quantile", 0.90))
    gate_mode = str(getattr(model.config, "dvmad_gate_mode", "both"))

    core = DVMADCore(
        mode=gate_mode,
        eps=eps,
        artificial_mode=artificial_mode,
        use_faiss=False,
    ).fit(X, y)

    train_scores = np.asarray(core.predict(X), dtype=np.float64)
    tau_fallback = float(np.quantile(train_scores, quantile))
    return core, tau_fallback


def calibrate_dvmad_tau(
    model: "BCMRNFSTModel",
    calibration_features: NDArray[np.float64],
) -> float | None:
    """Set `model.dvmad_tau` from Q-quantile of gate scores on held-out calibration features.

    Theorem D3 (DKW finite-sample): the empirical Q-quantile of known
    calibration scores is an unbiased estimate of the open-set rejection
    threshold. Replaces the train-set fallback tau computed at fit time.

    Also stores ``model.dvmad_tau_scale`` = tau_calibration / tau_train so
    refresh-time refits (which only see support-store points) can rescale
    their internal Q90 to match the calibration regime.
    """
    if model.dvmad_core is None:
        return None
    if calibration_features.size == 0:
        return model.dvmad_tau
    quantile = float(getattr(model.config, "dvmad_quantile", 0.90))
    scores = np.asarray(
        model.dvmad_core.predict(np.asarray(calibration_features, dtype=np.float64)),
        dtype=np.float64,
    )
    if scores.size == 0:
        return model.dvmad_tau
    tau_cal = float(np.quantile(scores, quantile))

    # Estimate scale = tau_cal / tau_loo. Refresh-time refit recomputes LOO
    # on the new support and multiplies by this scale to stay in the
    # calibration-set regime.
    bundle = _gather_support_phi(model)
    if bundle is not None:
        X_supp, _ = bundle
        if X_supp.shape[0] >= 4:
            tau_loo = _loo_quantile(model.dvmad_core, X_supp, quantile)
            if tau_loo > 1e-9:
                model.dvmad_tau_scale = tau_cal / tau_loo

    model.dvmad_tau = tau_cal
    return tau_cal


def _loo_quantile(core: DVMADCore, X: NDArray[np.float64], quantile: float) -> float:
    """Leave-one-out Q-quantile of NN distances.

    For each support row, take its 2nd-nearest distance in the projected
    basepoint set (skipping the self-distance), then Q-quantile over rows.
    Equivalent to LOO scoring without n_support refits.
    """
    Z = core.transform(X).astype(np.float32)
    base = core.basepoint_X_.astype(np.float32)
    nA = Z.shape[0]
    out = np.empty(nA, dtype=np.float64)
    norm_base = np.sum(base ** 2, axis=1)[None, :]
    chunk = max(1, min(2048, nA))
    for s in range(0, nA, chunk):
        e = min(s + chunk, nA)
        Ac = Z[s:e]
        norm_a = np.sum(Ac ** 2, axis=1)[:, None]
        dist2 = norm_a + norm_base - 2.0 * (Ac @ base.T)
        # k=2 partial sort along axis=1; index [1] is 2nd-smallest.
        partitioned = np.partition(dist2, kth=1, axis=1)[:, 1]
        out[s:e] = np.sqrt(np.maximum(partitioned, 0.0))
    return float(np.quantile(out, quantile))


def refit_gate_with_scale(model: "BCMRNFSTModel") -> tuple[DVMADCore | None, float | None]:
    """Refresh-time refit + self-calibrated tau via leave-one-out Q-quantile.

    The Q-quantile of LOO NN distances on the current support set is used
    as the new tau. This adapts to novel classes admitted since the
    original fit, where the calibration-set tau no longer covers the
    feature manifold of the augmented support.
    """
    core, tau_fallback = fit_dvmad_gate(model)
    if core is None or tau_fallback is None:
        return core, tau_fallback

    bundle = _gather_support_phi(model)
    if bundle is None:
        return core, tau_fallback
    X, _ = bundle
    quantile = float(getattr(model.config, "dvmad_quantile", 0.90))

    if X.shape[0] < 4:
        return core, tau_fallback

    tau_loo = _loo_quantile(core, X, quantile)
    scale = getattr(model, "dvmad_tau_scale", None)
    if scale is None or scale <= 0:
        return core, tau_loo
    return core, tau_loo * float(scale)


def score_dvmad_gate(
    model: "BCMRNFSTModel",
    X: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return per-sample DVM-AD gate score (NN distance in the two-tailed projection)."""
    core: DVMADCore | None = getattr(model, "dvmad_core", None)
    if core is None:
        return np.zeros(int(X.shape[0]), dtype=np.float64)
    return np.asarray(core.predict(np.asarray(X, dtype=np.float64)), dtype=np.float64)


def gate_rejects(
    model: "BCMRNFSTModel",
    scores: NDArray[np.float64],
) -> NDArray[np.bool_]:
    tau: float | None = getattr(model, "dvmad_tau", None)
    if tau is None:
        return np.zeros(int(scores.shape[0]), dtype=bool)
    return scores > tau
