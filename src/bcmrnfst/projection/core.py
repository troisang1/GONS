"""Moment statistics, scatter reconstruction, and soft-null projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from bcmrnfst.config.model import BCMRNFSTConfig
    from bcmrnfst.state.model import BCMRNFSTModel
    from bcmrnfst.state.support import SupportBundle


@dataclass(slots=True)
class MomentStats:
    """Exact moment summaries over the current bounded active support store."""

    n_total: float
    s1_global: NDArray[np.float64]
    S2_global: NDArray[np.float64]
    n_q: dict[str, float]
    s1_q: dict[str, NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class ResolvedThresholds:
    """Current relative thresholds resolved from the active geometry."""

    range_tol: float
    eps_null_refresh: float
    eps_aff_local: float
    eps_aff_global: float
    eta_margin: float
    rho_floor: float
    tau_sep: float


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """Soft-null projection result and relevant metadata."""

    W_proj: NDArray[np.float64]
    mode: str
    small_solver_mode: str
    n_null_dirs: int
    min_lambda_raw: float
    eps_null_used: float
    range_tol_used: float
    range_rank: int
    small_eigvals: NDArray[np.float64]
    range_eigvals: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ThresholdMetadata:
    """Serialized threshold state for the current projection version."""

    range_tol: float
    eps_null_refresh: float
    eps_aff_local: float
    eps_aff_global: float
    eta_margin: float
    rho_floor: float
    tau_sep: float
    tau_global: float
    calibration_mode: str


@dataclass(frozen=True, slots=True)
class EigenSolveResult:
    """Smallest-eigenpair solve output."""

    eigvals: NDArray[np.float64]
    eigvecs: NDArray[np.float64]
    mode: str


class ProjectionError(RuntimeError):
    """Raised when the active support state is too degenerate for projection."""


def scaled_num_tol(matrix: NDArray[np.float64]) -> float:
    """Machine-scaled numerical tolerance for a symmetric matrix."""

    dimension = max(1, int(matrix.shape[0]))
    trace_scale = float(np.trace(np.abs(matrix)) / dimension)
    spectral_scale = float(np.linalg.norm(matrix, ord=2))
    scale = max(1.0, trace_scale, spectral_scale)
    return float(np.finfo(matrix.dtype).eps * scale)


def scaled_num_tol_from_value(value: float) -> float:
    """Machine-scaled tolerance for a scalar reference value."""

    return float(np.finfo(np.float64).eps * max(1.0, abs(value)))


def weighted_mean(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute a stable weighted mean."""

    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("Expected strictly positive total weight.")
    weighted_sum = np.sum(values * weights[:, None], axis=0, dtype=np.float64)
    return np.asarray(weighted_sum / total_weight, dtype=np.float64)


def weighted_quantile(
    values: NDArray[np.float64],
    quantile: float,
    *,
    weights: NDArray[np.float64] | None = None,
) -> float:
    """Compute a 1D weighted quantile."""

    if values.size == 0:
        return 0.0
    if weights is None:
        return float(np.quantile(values, quantile))
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights, dtype=np.float64)
    cutoff = quantile * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    index = min(index, len(sorted_values) - 1)
    return float(sorted_values[index])


def support_radius(bundle: SupportBundle) -> float:
    """Compute the explicit-space support radius for one support bundle."""

    phi_bundle = bundle.phi_bundle
    weights = bundle.w_bundle
    center = weighted_mean(phi_bundle, weights)
    distances = np.linalg.norm(phi_bundle - center, axis=1)
    return weighted_quantile(distances, 0.5, weights=weights)


def median_knn_distance(points: NDArray[np.float64]) -> float:
    """Median nearest-neighbor distance with a safe small-sample fallback."""

    if points.shape[0] <= 1:
        return 1.0
    if points.shape[0] == 2:
        return float(np.linalg.norm(points[0] - points[1]))

    model = NearestNeighbors(n_neighbors=2, metric="euclidean")
    model.fit(points)
    distances = model.kneighbors(points, return_distance=True)[0]
    nearest = distances[:, 1]
    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        return 1.0
    return float(np.median(finite))


def stack_active_supports(
    model: BCMRNFSTModel,
    *,
    projected: bool,
) -> NDArray[np.float64]:
    """Stack all active support rows in explicit or projected space."""

    blocks: list[NDArray[np.float64]] = []
    for bundle in model.support_store.values():
        block = bundle.phi_bundle
        if projected:
            if model.W_proj is None:
                raise ValueError("Projected support requested before the projection matrix is set.")
            block = block @ model.W_proj
        blocks.append(np.asarray(block, dtype=np.float64))
    if not blocks:
        raise ValueError("Expected at least one support bundle.")
    return np.vstack(blocks)


def rebuild_stats_from_support_store(model: BCMRNFSTModel) -> MomentStats:
    """Rebuild exact moment summaries from the retained support store.

    When ``scatter_class_balance_gamma > 0``, each class's contribution is
    re-weighted so that under-represented classes (e.g. novel classes with
    few support samples) have equal influence on the scatter matrices as
    well-represented classes.  ``gamma=1.0`` is full class balancing.
    """

    first_bundle = next(iter(model.support_store.values()), None)
    if first_bundle is None:
        raise ValueError("Support store is empty.")
    feature_dim = int(first_bundle.phi_bundle.shape[1])

    gamma = getattr(model.config, "scatter_class_balance_gamma", 0.0)

    # First pass: accumulate per-class total weight for rebalancing.
    class_weights: dict[str, float] = {}
    if gamma > 0:
        for bundle in model.support_store.values():
            cls = bundle.class_label
            class_weights[cls] = class_weights.get(cls, 0.0) + float(
                np.sum(bundle.w_bundle)
            )

    # Compute per-class scale factors: scale = (n_class)^(-gamma) * C
    class_scale: dict[str, float] = {}
    if gamma > 0 and class_weights:
        n_classes = len(class_weights)
        total_weight = sum(class_weights.values())
        target_per_class = total_weight / n_classes
        for cls, cw in class_weights.items():
            if cw > 0:
                class_scale[cls] = target_per_class / cw
            else:
                class_scale[cls] = 1.0

    s1_global = np.zeros(feature_dim, dtype=np.float64)
    S2_global = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    n_q: dict[str, float] = {}
    s1_q: dict[str, NDArray[np.float64]] = {}
    n_total = 0.0

    for unit_id, bundle in model.support_store.items():
        phi_bundle = np.asarray(bundle.phi_bundle, dtype=np.float64)
        weights = np.asarray(bundle.w_bundle, dtype=np.float64)
        if phi_bundle.shape[0] != weights.shape[0]:
            raise ValueError(f"Support bundle '{unit_id}' has mismatched feature and weight rows.")

        # Apply class-balance scaling.
        if gamma > 0 and bundle.class_label in class_scale:
            scale = class_scale[bundle.class_label] ** gamma
            weights = weights * scale

        n_unit = float(np.sum(weights))
        s1_unit = np.asarray(
            np.sum(phi_bundle * weights[:, None], axis=0, dtype=np.float64),
            dtype=np.float64,
        )
        S2_unit = phi_bundle.T @ (weights[:, None] * phi_bundle)
        n_q[unit_id] = n_unit
        s1_q[unit_id] = s1_unit
        n_total += n_unit
        s1_global += s1_unit
        S2_global += S2_unit

    return MomentStats(
        n_total=n_total,
        s1_global=s1_global,
        S2_global=S2_global,
        n_q=n_q,
        s1_q=s1_q,
    )


def reconstruct_scatters(
    stats: MomentStats,
    *,
    eps: float = 1e-12,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reconstruct within-unit and total scatter matrices."""

    if stats.n_total <= eps:
        raise ValueError("Expected strictly positive total retained support.")
    St = stats.S2_global - np.outer(stats.s1_global, stats.s1_global) / stats.n_total
    Sw = np.array(stats.S2_global, copy=True)
    for unit_id, n_unit in stats.n_q.items():
        if n_unit > eps:
            Sw -= np.outer(stats.s1_q[unit_id], stats.s1_q[unit_id]) / n_unit
    St = 0.5 * (St + St.T)
    Sw = 0.5 * (Sw + Sw.T)
    return Sw.astype(np.float64), St.astype(np.float64)


def solve_smallest_symmetric_eigs(
    A: NDArray[np.float64],
    k_target: int,
    config: BCMRNFSTConfig,
) -> EigenSolveResult:
    """Solve for the smallest eigenpairs of a symmetric matrix."""

    dimension = int(A.shape[0])
    if dimension == 1:
        return EigenSolveResult(
            eigvals=np.array([float(A[0, 0])], dtype=np.float64),
            eigvecs=np.array([[1.0]], dtype=np.float64),
            mode="trivial",
        )

    k_eff = min(max(1, k_target), dimension - 1)
    if config.small_eig_solver == "dense_only":
        eigvals, eigvecs = eigh(A, subset_by_index=[0, k_eff - 1])
        order = np.argsort(eigvals)
        return EigenSolveResult(
            eigvals=np.asarray(eigvals[order], dtype=np.float64),
            eigvecs=np.asarray(eigvecs[:, order], dtype=np.float64),
            mode="dense_only",
        )

    if dimension <= config.n_dense_max or k_eff >= dimension - 1:
        eigvals, eigvecs = eigh(A, subset_by_index=[0, k_eff - 1])
        order = np.argsort(eigvals)
        return EigenSolveResult(
            eigvals=np.asarray(eigvals[order], dtype=np.float64),
            eigvecs=np.asarray(eigvecs[:, order], dtype=np.float64),
            mode="dense",
        )

    try:
        if config.use_shift_invert_small:
            eigvals, eigvecs = eigsh(
                A,
                k=k_eff,
                sigma=0.0,
                which="LM",
                tol=config.eig_tol,
                maxiter=config.eig_maxiter,
            )
            mode = "eigsh_shift_invert"
        else:
            eigvals, eigvecs = eigsh(
                A,
                k=k_eff,
                which="SM",
                tol=config.eig_tol,
                maxiter=config.eig_maxiter,
            )
            mode = "eigsh_sm"
    except ArpackNoConvergence as err:
        if (
            err.eigenvalues is not None
            and err.eigenvectors is not None
            and len(err.eigenvalues) > 0
        ):
            eigvals = np.asarray(err.eigenvalues, dtype=np.float64)
            eigvecs = np.asarray(err.eigenvectors, dtype=np.float64)
            mode = "eigsh_partial"
        else:
            eigvals, eigvecs = eigh(A, subset_by_index=[0, k_eff - 1])
            mode = "dense_fallback"

    order = np.argsort(eigvals)
    return EigenSolveResult(
        eigvals=np.asarray(eigvals[order], dtype=np.float64),
        eigvecs=np.asarray(eigvecs[:, order], dtype=np.float64),
        mode=mode,
    )


def resolve_refresh_scales(
    model: BCMRNFSTModel,
    *,
    optional_small_spectrum: NDArray[np.float64] | None = None,
    optional_range_spectrum: NDArray[np.float64] | None = None,
) -> ResolvedThresholds:
    """Resolve relative thresholds from the current active geometry."""

    aff_ref_candidates = [support_radius(bundle) for bundle in model.support_store.values()]
    aff_ref = float(np.median(aff_ref_candidates)) if aff_ref_candidates else 1.0

    explicit_support = stack_active_supports(model, projected=False)
    projected_support = (
        stack_active_supports(model, projected=True)
        if model.W_proj is not None
        else explicit_support
    )
    knn_ref = median_knn_distance(projected_support)

    stable_radii = [
        prototype.rho
        for prototype in model.proto_bank.values()
        if prototype.status in {"stable", "fragile_stable"}
    ]
    rho_ref = float(np.median(stable_radii)) if stable_radii else knn_ref

    range_spectrum = (
        optional_range_spectrum
        if optional_range_spectrum is not None
        else np.array([knn_ref**2], dtype=np.float64)
    )
    lambda_max_t = float(np.max(range_spectrum)) if range_spectrum.size else knn_ref**2

    positive_small = (
        optional_small_spectrum[optional_small_spectrum > 0.0]
        if optional_small_spectrum is not None
        else np.array([], dtype=np.float64)
    )
    eig_ref = float(np.median(positive_small)) if positive_small.size else lambda_max_t

    if model.config.threshold_mode == "fixed_absolute":
        return ResolvedThresholds(
            range_tol=max(model.config.c_range, scaled_num_tol_from_value(model.config.c_range)),
            eps_null_refresh=max(
                model.config.c_null,
                scaled_num_tol_from_value(model.config.c_null),
            ),
            eps_aff_local=float(model.config.c_aff_local),
            eps_aff_global=float(model.config.c_aff_global),
            eta_margin=float(model.config.c_margin),
            rho_floor=max(model.config.c_rho, 1e-12),
            tau_sep=max(model.config.c_sep, 1e-12),
        )

    return ResolvedThresholds(
        range_tol=max(model.config.c_range * lambda_max_t, scaled_num_tol_from_value(lambda_max_t)),
        eps_null_refresh=max(model.config.c_null * eig_ref, scaled_num_tol_from_value(eig_ref)),
        eps_aff_local=model.config.c_aff_local * max(aff_ref, 1e-12),
        eps_aff_global=model.config.c_aff_global * max(aff_ref, 1e-12),
        eta_margin=model.config.c_margin * max(rho_ref, 1e-12),
        rho_floor=model.config.c_rho * max(knn_ref, 1e-12),
        tau_sep=max(model.tau_global, model.config.c_sep * model.tau_global),
    )


def build_threshold_metadata(
    scales: ResolvedThresholds,
    *,
    tau_global: float,
    calibration_mode: str,
) -> ThresholdMetadata:
    """Attach threshold metadata to the current projection version."""

    return ThresholdMetadata(
        range_tol=scales.range_tol,
        eps_null_refresh=scales.eps_null_refresh,
        eps_aff_local=scales.eps_aff_local,
        eps_aff_global=scales.eps_aff_global,
        eta_margin=scales.eta_margin,
        rho_floor=scales.rho_floor,
        tau_sep=scales.tau_sep,
        tau_global=tau_global,
        calibration_mode=calibration_mode,
    )


def compute_soft_null_projection(model: BCMRNFSTModel) -> ProjectionResult:
    """Compute the soft-null projection over the active bounded support state.

    Theory-ablation extension: when ``model.config.projection_mode`` is
    ``"pca"`` or ``"random"``, replace the null-space projection with the
    top-r eigenvectors of the total scatter (PCA) or a random orthonormal
    basis of the same dimension. Used to validate that the null-space
    projection is uniquely good for open-set rejection.
    """

    if model.stats is None:
        raise ValueError("Model statistics must be available before projection.")
    Sw_raw, St_raw = reconstruct_scatters(model.stats)
    if model.config.c_reg > 0.0:
        feature_dim = int(Sw_raw.shape[0])
        lambda_reg = model.config.c_reg * float(np.trace(Sw_raw)) / max(feature_dim, 1)
        Sw_raw = Sw_raw + lambda_reg * np.eye(feature_dim, dtype=np.float64)
    range_eigvals, range_eigvecs = eigh(St_raw)
    range_eigvals = np.asarray(range_eigvals, dtype=np.float64)
    range_eigvecs = np.asarray(range_eigvecs, dtype=np.float64)

    proj_mode = getattr(model.config, "projection_mode", "null_space")
    if proj_mode == "dvmad_two_tailed":
        return _compute_dvmad_two_tailed_projection(
            model,
            range_eigvals=range_eigvals,
        )
    if proj_mode in ("pca", "random"):
        feature_dim = int(St_raw.shape[0])
        keep_count = min(int(model.config.d_proj_max), feature_dim)
        if proj_mode == "pca":
            # Top-d_proj_max eigenvectors of total scatter (largest variance directions).
            order = np.argsort(range_eigvals)[::-1]
            top_idx = order[:keep_count]
            W_proj = np.asarray(range_eigvecs[:, top_idx], dtype=np.float64)
            sel_eigvals = range_eigvals[top_idx]
        else:  # random
            rng = np.random.default_rng(int(model.config.seed))
            G = rng.standard_normal((feature_dim, keep_count))
            Q, _ = np.linalg.qr(G)
            W_proj = np.asarray(Q[:, :keep_count], dtype=np.float64)
            sel_eigvals = np.asarray(np.diag(W_proj.T @ Sw_raw @ W_proj), dtype=np.float64)
        return ProjectionResult(
            W_proj=W_proj,
            mode=proj_mode,
            small_solver_mode="ablation",
            n_null_dirs=0,
            min_lambda_raw=float(np.min(sel_eigvals)) if sel_eigvals.size else 0.0,
            eps_null_used=0.0,
            range_tol_used=0.0,
            range_rank=int(range_eigvals.size),
            small_eigvals=sel_eigvals,
            range_eigvals=range_eigvals,
        )

    scales_pre = resolve_refresh_scales(model, optional_range_spectrum=range_eigvals)
    keep_range = np.flatnonzero(range_eigvals > scales_pre.range_tol)
    if keep_range.size == 0:
        raise ProjectionError("Degenerate active state: no usable total-scatter range.")

    B = range_eigvecs[:, keep_range]
    M_raw = 0.5 * ((B.T @ Sw_raw @ B) + (B.T @ Sw_raw @ B).T)
    eig_result = solve_smallest_symmetric_eigs(M_raw, model.config.d_proj_max, model.config)
    scales = resolve_refresh_scales(
        model,
        optional_small_spectrum=eig_result.eigvals,
        optional_range_spectrum=range_eigvals,
    )
    null_idx = np.flatnonzero(eig_result.eigvals <= scales.eps_null_refresh)
    if null_idx.size > 0:
        keep_count = min(model.config.d_proj_max, int(null_idx.size))
        keep = np.arange(keep_count, dtype=np.int64)
        mode = "epsilon_null"
    else:
        keep_count = min(model.config.d_proj_max, int(eig_result.eigvals.size))
        keep = np.arange(keep_count, dtype=np.int64)
        mode = "min_scatter_fallback"

    V_small = eig_result.eigvecs[:, keep]
    W_proj = np.asarray(B @ V_small, dtype=np.float64)
    return ProjectionResult(
        W_proj=W_proj,
        mode=mode,
        small_solver_mode=eig_result.mode,
        n_null_dirs=int(null_idx.size),
        min_lambda_raw=float(np.min(eig_result.eigvals)),
        eps_null_used=scales.eps_null_refresh,
        range_tol_used=scales.range_tol,
        range_rank=int(keep_range.size),
        small_eigvals=eig_result.eigvals,
        range_eigvals=range_eigvals,
    )


def _gather_labeled_support(
    model: BCMRNFSTModel,
) -> tuple[NDArray[np.float64], NDArray[np.int64]] | None:
    """Concatenate per-class support features for DVM-AD generalised scatter.

    Returns ``None`` when fewer than two classes contribute support rows or
    the available row count is below the projection budget — caller must
    fall back to the null-space pipeline in that case.
    """
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
    X = np.vstack(feats)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def _compute_dvmad_two_tailed_projection(
    model: BCMRNFSTModel,
    *,
    range_eigvals: NDArray[np.float64],
) -> ProjectionResult:
    """DVM-AD two-tailed eigenvector projection (`projection_mode='dvmad_two_tailed'`).

    Min-tail directions recover the existing null-space projection; max-tail
    directions add the structural-saturated directions.

    Falls back to the null-space pipeline when the support store cannot
    yield labeled feature points (e.g., warm-up before the first refresh).
    """
    from bcmrnfst.dvmad import DVMADCore

    bundle = _gather_labeled_support(model)
    if bundle is None:
        return _compute_null_space_fallback(model, range_eigvals=range_eigvals)
    X, y = bundle

    eps = float(getattr(model.config, "dvmad_eps", 0.1))
    artificial_mode = str(getattr(model.config, "dvmad_artificial_mode", "farthest"))
    keep_max = int(model.config.d_proj_max)

    core = DVMADCore(
        mode="both",
        eps=eps,
        artificial_mode=artificial_mode,
        use_faiss=False,
    )
    n_aug = X.shape[0] + 1
    y_aug = np.hstack([y, [int(y.max()) + 1]])
    X_aug = np.vstack([X, core._construct_reference_point(X)[None, :]])
    npd = core.compute_discriminants(X_aug, y_aug)
    eigvals = core.filtered_eigvals_

    if npd.shape[1] > keep_max:
        order = np.argsort(np.abs(eigvals - 0.5))[::-1]
        keep_idx = np.sort(order[:keep_max])
        npd = npd[:, keep_idx]
        eigvals = eigvals[keep_idx]

    return ProjectionResult(
        W_proj=np.asarray(npd, dtype=np.float64),
        mode="dvmad_two_tailed",
        small_solver_mode=f"dvmad(n_aug={n_aug},eps={eps})",
        n_null_dirs=int((eigvals < eps).sum()),
        min_lambda_raw=float(eigvals.min()) if eigvals.size else 0.0,
        eps_null_used=eps,
        range_tol_used=0.0,
        range_rank=int(range_eigvals.size),
        small_eigvals=np.asarray(eigvals, dtype=np.float64),
        range_eigvals=range_eigvals,
    )


def _compute_null_space_fallback(
    model: BCMRNFSTModel,
    *,
    range_eigvals: NDArray[np.float64],
) -> ProjectionResult:
    """Re-run the standard null-space pipeline (used as DVM-AD warm-up fallback)."""
    Sw_raw, St_raw = reconstruct_scatters(model.stats)
    if model.config.c_reg > 0.0:
        feature_dim = int(Sw_raw.shape[0])
        lambda_reg = model.config.c_reg * float(np.trace(Sw_raw)) / max(feature_dim, 1)
        Sw_raw = Sw_raw + lambda_reg * np.eye(feature_dim, dtype=np.float64)
    range_eigvals_local, range_eigvecs = eigh(St_raw)
    range_eigvals_local = np.asarray(range_eigvals_local, dtype=np.float64)
    range_eigvecs = np.asarray(range_eigvecs, dtype=np.float64)
    scales_pre = resolve_refresh_scales(model, optional_range_spectrum=range_eigvals_local)
    keep_range = np.flatnonzero(range_eigvals_local > scales_pre.range_tol)
    if keep_range.size == 0:
        raise ProjectionError("Degenerate active state: no usable total-scatter range.")
    B = range_eigvecs[:, keep_range]
    M_raw = 0.5 * ((B.T @ Sw_raw @ B) + (B.T @ Sw_raw @ B).T)
    eig_result = solve_smallest_symmetric_eigs(M_raw, model.config.d_proj_max, model.config)
    scales = resolve_refresh_scales(
        model,
        optional_small_spectrum=eig_result.eigvals,
        optional_range_spectrum=range_eigvals_local,
    )
    null_idx = np.flatnonzero(eig_result.eigvals <= scales.eps_null_refresh)
    if null_idx.size > 0:
        keep_count = min(model.config.d_proj_max, int(null_idx.size))
        mode = "epsilon_null"
    else:
        keep_count = min(model.config.d_proj_max, int(eig_result.eigvals.size))
        mode = "min_scatter_fallback"
    keep = np.arange(keep_count, dtype=np.int64)
    V_small = eig_result.eigvecs[:, keep]
    W_proj = np.asarray(B @ V_small, dtype=np.float64)
    return ProjectionResult(
        W_proj=W_proj,
        mode=f"dvmad_warmup_fallback({mode})",
        small_solver_mode=eig_result.mode,
        n_null_dirs=int(null_idx.size),
        min_lambda_raw=float(np.min(eig_result.eigvals)),
        eps_null_used=scales.eps_null_refresh,
        range_tol_used=scales.range_tol,
        range_rank=int(keep_range.size),
        small_eigvals=eig_result.eigvals,
        range_eigvals=range_eigvals_local,
    )


def warm_start_projection(
    W_old: NDArray[np.float64] | None,
    W_fresh: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    """Blend a fresh projection with the null space of the previous one."""

    if W_old is None or alpha <= 0.0:
        return np.asarray(W_fresh, dtype=np.float64)

    old_rank = int(W_old.shape[1])
    if old_rank == 0:
        return np.asarray(W_fresh, dtype=np.float64)

    gram = W_old.T @ W_old
    stabilized_gram = gram + 1e-12 * np.eye(old_rank, dtype=np.float64)
    null_projector = np.eye(W_old.shape[0], dtype=np.float64) - W_old @ np.linalg.inv(
        stabilized_gram
    ) @ W_old.T
    constrained_delta = null_projector @ W_fresh
    delta_norms = np.linalg.norm(constrained_delta, axis=0)
    fresh_norms = np.linalg.norm(W_fresh, axis=0)
    relative_norms = delta_norms / np.maximum(fresh_norms, 1e-12)
    if float(np.median(relative_norms)) < 1e-6:
        return np.asarray(W_fresh, dtype=np.float64)

    constrained_q, _ = np.linalg.qr(constrained_delta, mode="reduced")
    blended = (1.0 - alpha) * W_fresh + alpha * constrained_q
    warmed_q, _ = np.linalg.qr(blended, mode="reduced")
    return np.asarray(warmed_q, dtype=np.float64)
