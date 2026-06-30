"""Prototype bank construction and configurable scoring."""

from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.spatial.distance import pdist

from gons.config.model import ScoringMode
from gons.projection.core import (
    ProjectionResult,
    ResolvedThresholds,
    ThresholdMetadata,
    weighted_mean,
    weighted_quantile,
)

if TYPE_CHECKING:
    from gons.state.model import GONSModel


@dataclass(frozen=True, slots=True)
class Prototype:
    """Projected prototype entry used for novelty scoring."""

    prototype_id: str
    b: NDArray[np.float64]
    rho: float
    class_label: str
    member_q_ids: tuple[str, ...]
    support_weight: float
    status: str
    tau_override: float | None
    sigma_inv: NDArray[np.float64] | None = None
    use_mahalanobis: bool = False
    tau_local: float | None = None
    b_full: NDArray[np.float64] | None = None
    rho_full: float | None = None
    diag_precision: NDArray[np.float64] | None = None


def _stabilize_covariance(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    stabilized = 0.5 * (matrix + matrix.T)
    jitter = 1e-10 * np.eye(matrix.shape[0], dtype=np.float64)
    return np.asarray(stabilized + jitter, dtype=np.float64)


def _weighted_covariance(
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    center: NDArray[np.float64],
) -> NDArray[np.float64]:
    centered = np.asarray(values - center[None, :], dtype=np.float64)
    support_weight = float(np.sum(weights))
    if support_weight <= 0.0:
        return np.eye(values.shape[1], dtype=np.float64)
    covariance = centered.T @ (weights[:, None] * centered)
    return np.asarray(covariance / max(support_weight, 1e-12), dtype=np.float64)


def calibrate_prototype_center(
    b_novel: NDArray[np.float64],
    stable_prototypes: dict[str, Prototype],
    *,
    n_support: float,
    lambda_base: float,
    temperature: float,
    n_ref: float,
    top_k: int,
    novel_class_label: str,
) -> NDArray[np.float64]:
    """Blend a low-support prototype toward nearby stable base-class prototypes."""

    candidates = [
        prototype
        for prototype in stable_prototypes.values()
        if prototype.class_label != novel_class_label and prototype.status == "stable"
    ]
    if not candidates:
        return np.asarray(b_novel, dtype=np.float64)

    centers = np.vstack([prototype.b for prototype in candidates])
    novel_norm = float(np.linalg.norm(b_novel))
    if novel_norm <= 1e-12:
        return np.asarray(b_novel, dtype=np.float64)

    center_norms = np.linalg.norm(centers, axis=1)
    similarities = (centers @ b_novel) / np.maximum(center_norms * novel_norm, 1e-12)
    keep_count = min(max(top_k, 1), len(candidates))
    top_indices = np.argsort(similarities)[-keep_count:]
    top_scores = similarities[top_indices]
    top_centers = centers[top_indices]
    logits = (top_scores - np.max(top_scores)) / max(temperature, 1e-12)
    weights = np.exp(logits)
    weights = weights / np.maximum(np.sum(weights), 1e-12)
    base_blend = np.sum(weights[:, None] * top_centers, axis=0, dtype=np.float64)
    lam = lambda_base / (1.0 + n_support / max(n_ref, 1.0))
    return np.asarray((1.0 - lam) * b_novel + lam * base_blend, dtype=np.float64)


def prototype_scores(
    z: NDArray[np.float64],
    prototype_matrix: NDArray[np.float64],
    *,
    radii_vec: NDArray[np.float64] | None = None,
    sigma_inv_stack: NDArray[np.float64] | None = None,
    use_mahalanobis: bool = False,
) -> NDArray[np.float64]:
    """Score one projected row against all candidate prototypes."""

    if prototype_matrix.size == 0:
        raise ValueError("Prototype bank is empty.")
    diff = np.asarray(prototype_matrix - z[None, :], dtype=np.float64)
    if use_mahalanobis:
        if sigma_inv_stack is None:
            raise ValueError("Mahalanobis scoring requires per-prototype inverse covariance.")
        quadratic = np.einsum("ni,nij,nj->n", diff, sigma_inv_stack, diff)
        return np.asarray(np.sqrt(np.maximum(quadratic, 0.0)), dtype=np.float64)
    if radii_vec is None:
        raise ValueError("Normalized prototype scoring requires prototype radii.")
    raw_dist = np.linalg.norm(diff, axis=1)
    return np.asarray(raw_dist / np.maximum(radii_vec, 1e-12), dtype=np.float64)


def reduce_prototype_scores(
    scores: NDArray[np.float64],
    *,
    scoring_mode: ScoringMode,
    temperature: float,
) -> float:
    """Reduce per-prototype scores into one novelty score."""

    if scores.size == 0:
        raise ValueError("Prototype score array is empty.")
    if scoring_mode == "min_distance":
        return float(np.min(scores))
    temp = max(float(temperature), 1e-12)
    return float(-temp * logsumexp(-np.asarray(scores, dtype=np.float64) / temp))


def nearest_prototype(
    z: NDArray[np.float64],
    prototype_matrix: NDArray[np.float64],
    proto_ids: tuple[str, ...],
    *,
    radii_vec: NDArray[np.float64] | None = None,
    sigma_inv_stack: NDArray[np.float64] | None = None,
    use_mahalanobis: bool = False,
) -> tuple[str, float]:
    """Return the nearest prototype under the configured geometry."""

    if len(proto_ids) == 0:
        raise ValueError("Prototype bank is empty.")
    scores = prototype_scores(
        z,
        prototype_matrix,
        radii_vec=radii_vec,
        sigma_inv_stack=sigma_inv_stack,
        use_mahalanobis=use_mahalanobis,
    )
    best_index = int(np.argmin(scores))
    return proto_ids[best_index], float(scores[best_index])


def nearest_mahalanobis_prototype(
    z: NDArray[np.float64],
    prototype_matrix: NDArray[np.float64],
    proto_ids: tuple[str, ...],
    sigma_inv_stack: NDArray[np.float64],
) -> tuple[str, float]:
    """Return the nearest prototype under Mahalanobis distance."""

    return nearest_prototype(
        z,
        prototype_matrix,
        proto_ids,
        sigma_inv_stack=sigma_inv_stack,
        use_mahalanobis=True,
    )


def merge_two_prototypes(
    left: Prototype,
    right: Prototype,
    *,
    tau_global: float,
    fragile_tau_mult: float,
) -> Prototype:
    """Merge two same-class prototypes into a single replacement entry."""

    total_weight = left.support_weight + right.support_weight
    merged_center = (
        (left.b * left.support_weight + right.b * right.support_weight) / max(total_weight, 1e-12)
    )
    merged_status = "stable" if "stable" in {left.status, right.status} else left.status
    tau_override = None
    if merged_status != "stable":
        tau_override = fragile_tau_mult * tau_global
    merged_sigma_inv = None
    use_mahalanobis = left.use_mahalanobis or right.use_mahalanobis
    if left.sigma_inv is not None and right.sigma_inv is not None:
        left_cov = np.linalg.pinv(left.sigma_inv)
        right_cov = np.linalg.pinv(right.sigma_inv)
        within_cov = (
            left.support_weight * left_cov + right.support_weight * right_cov
        ) / max(total_weight, 1e-12)
        center_diff = left.b - right.b
        between_cov = (
            (left.support_weight * right.support_weight)
            / max(total_weight**2, 1e-12)
        ) * np.outer(center_diff, center_diff)
        merged_cov = within_cov + between_cov
        merged_sigma_inv = np.linalg.pinv(_stabilize_covariance(merged_cov))
    elif left.sigma_inv is not None or right.sigma_inv is not None:
        merged_sigma_inv = left.sigma_inv if left.sigma_inv is not None else right.sigma_inv
    merged_b_full = None
    merged_rho_full = None
    merged_diag_precision = None
    if left.b_full is not None and right.b_full is not None:
        merged_b_full = (
            (left.b_full * left.support_weight + right.b_full * right.support_weight)
            / max(total_weight, 1e-12)
        )
        merged_rho_full = max(
            left.rho_full if left.rho_full is not None else 0.0,
            right.rho_full if right.rho_full is not None else 0.0,
        )
    if left.diag_precision is not None and right.diag_precision is not None:
        left_var = 1.0 / np.maximum(left.diag_precision, 1e-10)
        right_var = 1.0 / np.maximum(right.diag_precision, 1e-10)
        merged_var = (
            left.support_weight * left_var + right.support_weight * right_var
        ) / max(total_weight, 1e-12)
        merged_diag_precision = 1.0 / np.maximum(merged_var, 1e-10)
    elif left.diag_precision is not None or right.diag_precision is not None:
        merged_diag_precision = (
            left.diag_precision if left.diag_precision is not None else right.diag_precision
        )

    return Prototype(
        prototype_id=left.prototype_id,
        b=np.asarray(merged_center, dtype=np.float64),
        rho=max(left.rho, right.rho),
        class_label=left.class_label,
        member_q_ids=tuple(sorted(set(left.member_q_ids + right.member_q_ids))),
        support_weight=total_weight,
        status=merged_status,
        tau_override=tau_override,
        sigma_inv=merged_sigma_inv,
        use_mahalanobis=use_mahalanobis,
        tau_local=None,
        b_full=merged_b_full,
        rho_full=merged_rho_full,
        diag_precision=merged_diag_precision,
    )


def build_stable_prototypes_from_support_store(
    model: GONSModel,
    scales: ResolvedThresholds,
) -> dict[str, Prototype]:
    """Build stable and fragile-stable prototypes from the bounded support store."""

    if model.W_proj is None:
        raise ValueError("Model projection must be available before prototype construction.")

    projected_supports: dict[str, dict[str, object]] = {}
    shared_covariance = np.zeros((model.W_proj.shape[1], model.W_proj.shape[1]), dtype=np.float64)
    shared_weight = 0.0
    raw_bank: dict[str, Prototype] = {}

    for unit_id, bundle in model.support_store.items():
        Z_q = np.asarray(bundle.phi_bundle @ model.W_proj, dtype=np.float64)
        weights = np.asarray(bundle.w_bundle, dtype=np.float64)
        support_weight = float(np.sum(weights))
        center = weighted_mean(Z_q, weights)
        projected_supports[unit_id] = {
            "Z_q": Z_q,
            "weights": weights,
            "support_weight": support_weight,
            "class_label": bundle.class_label,
        }
        shared_covariance += support_weight * _weighted_covariance(Z_q, weights, center)
        shared_weight += support_weight

        distances = np.linalg.norm(Z_q - center, axis=1)
        rho_q = max(
            weighted_quantile(
                distances,
                model.config.radius_quantile,
                weights=weights,
            ),
            scales.rho_floor,
        )
        if model.config.c_squeeze > 0.0 and model.projection_version == 0:
            rho_q = max(rho_q * (1.0 - model.config.c_squeeze), scales.rho_floor)
        low_support = support_weight < model.config.min_microclass_size
        status = "fragile_stable" if low_support else "stable"
        raw_bank[unit_id] = Prototype(
            prototype_id=unit_id,
            b=center,
            rho=float(rho_q),
            class_label=bundle.class_label,
            member_q_ids=(unit_id,),
            support_weight=support_weight,
            status=status,
            tau_override=(
                model.config.fragile_tau_mult * model.tau_global if low_support else None
            ),
        )

    shared_covariance = shared_covariance / max(shared_weight, 1e-12)
    shared_diag = np.diag(np.diag(shared_covariance))
    projected_dim = int(model.W_proj.shape[1])

    proto_bank: dict[str, Prototype] = {}
    for unit_id, bundle in model.support_store.items():
        state = projected_supports[unit_id]
        Z_q = np.asarray(state["Z_q"], dtype=np.float64)
        weights = np.asarray(state["weights"], dtype=np.float64)
        support_weight = float(state["support_weight"])
        class_label = str(state["class_label"])
        low_support = support_weight < model.config.min_microclass_size

        center = raw_bank[unit_id].b
        if model.config.enable_teen_calibration and low_support:
            center = calibrate_prototype_center(
                center,
                raw_bank,
                n_support=support_weight,
                lambda_base=model.config.teen_lambda_base,
                temperature=model.config.teen_temperature,
                n_ref=float(model.config.min_microclass_size),
                top_k=model.config.teen_top_k,
                novel_class_label=class_label,
            )

        distances = np.linalg.norm(Z_q - center, axis=1)
        rho_q = max(
            weighted_quantile(
                distances,
                model.config.radius_quantile,
                weights=weights,
            ),
            scales.rho_floor,
        )
        if model.config.c_squeeze > 0.0 and model.projection_version == 0:
            rho_q = max(rho_q * (1.0 - model.config.c_squeeze), scales.rho_floor)

        # Support radius shrink: tighten acceptance radius for high-support
        # prototypes to reduce the absorber effect.
        if model.config.enable_support_radius_shrink:
            total_support = sum(
                float(np.sum(b.w_bundle)) for b in model.support_store.values()
            )
            mean_support = total_support / max(len(model.support_store), 1)
            support_ratio = support_weight / max(mean_support, 1e-12)
            if support_ratio > 1.0:
                shrink_factor = 1.0 / (
                    1.0 + model.config.radius_shrink_rate * (support_ratio - 1.0)
                )
                rho_q = max(rho_q * shrink_factor, scales.rho_floor)

        sigma_inv = None
        if model.config.use_mahalanobis:
            alpha = 0.0
            if model.config.shrinkage_mode == "adaptive":
                alpha = support_weight / max(support_weight + projected_dim, 1e-12)
            covariance = _weighted_covariance(Z_q, weights, center)
            shrunk_covariance = alpha * covariance + (1.0 - alpha) * shared_diag
            sigma_inv = np.linalg.pinv(_stabilize_covariance(shrunk_covariance))
        diag_precision = None
        if model.config.enable_ghost_threshold:
            centered = Z_q - center[None, :]
            weight_sum = float(np.sum(weights))
            if weight_sum > 1e-12 and Z_q.shape[0] >= 2:
                var_per_dim = np.sum(weights[:, None] * centered**2, axis=0) / weight_sum
                diag_precision = 1.0 / np.maximum(var_per_dim, 1e-10)
            else:
                diag_precision = np.ones(Z_q.shape[1], dtype=np.float64)

        b_full = None
        rho_full = None
        if model.config.enable_dual_space_scoring:
            phi_block = np.asarray(bundle.phi_bundle, dtype=np.float64)
            b_full = weighted_mean(phi_block, weights)
            full_distances = np.linalg.norm(phi_block - b_full[None, :], axis=1)
            rho_full = max(
                weighted_quantile(
                    full_distances,
                    model.config.radius_quantile,
                    weights=weights,
                ),
                scales.rho_floor,
            )

        prototype = raw_bank[unit_id]
        proto_bank[unit_id] = Prototype(
            prototype_id=prototype.prototype_id,
            b=np.asarray(center, dtype=np.float64),
            rho=float(rho_q),
            class_label=prototype.class_label,
            member_q_ids=prototype.member_q_ids,
            support_weight=prototype.support_weight,
            status=prototype.status,
            tau_override=prototype.tau_override,
            sigma_inv=sigma_inv,
            use_mahalanobis=bool(model.config.use_mahalanobis),
            tau_local=None,
            b_full=b_full,
            rho_full=rho_full,
            diag_precision=diag_precision,
        )
    return proto_bank


def cap_prototype_bank(model: GONSModel) -> GONSModel:
    """Merge same-class prototypes until the configured cap is satisfied."""

    if len(model.proto_bank) <= model.config.Q_proto_max:
        return model
    if model.config.Q_proto_max < model.active_class_count:
        raise ValueError("Q_proto_max must be at least the number of active classes.")

    class_members: dict[str, set[str]] = {}
    class_versions: dict[str, int] = {}
    candidate_heap: list[tuple[float, str, str, str, int]] = []

    for prototype_id, prototype in model.proto_bank.items():
        class_members.setdefault(prototype.class_label, set()).add(prototype_id)
    for class_label in class_members:
        class_versions[class_label] = 0

    def push_class_candidate(class_label: str) -> None:
        candidate = _best_class_merge_candidate(model.proto_bank, class_members[class_label])
        if candidate is None:
            return
        distance, left_id, right_id = candidate
        heapq.heappush(
            candidate_heap,
            (distance, left_id, right_id, class_label, class_versions[class_label]),
        )

    for class_label in sorted(class_members):
        push_class_candidate(class_label)

    while len(model.proto_bank) > model.config.Q_proto_max:
        best_pair: tuple[str, str, str] | None = None
        while candidate_heap:
            _, left_id, right_id, class_label, version = heapq.heappop(candidate_heap)
            if version != class_versions.get(class_label):
                continue
            if left_id not in model.proto_bank or right_id not in model.proto_bank:
                continue
            left = model.proto_bank[left_id]
            right = model.proto_bank[right_id]
            if left.class_label != class_label or right.class_label != class_label:
                continue
            best_pair = (class_label, left_id, right_id)
            break
        if best_pair is None:
            raise ValueError("Could not find a safe same-class prototype merge candidate.")
        class_label, left_id, right_id = best_pair
        merged = merge_two_prototypes(
            model.proto_bank[left_id],
            model.proto_bank[right_id],
            tau_global=model.tau_global,
            fragile_tau_mult=model.config.fragile_tau_mult,
        )
        model.proto_bank[left_id] = merged
        del model.proto_bank[right_id]
        class_members[class_label].discard(right_id)
        class_versions[class_label] += 1
        push_class_candidate(class_label)
    return model


def _best_class_merge_candidate(
    proto_bank: dict[str, Prototype],
    prototype_ids: set[str],
) -> tuple[float, str, str] | None:
    """Return the closest mergeable pair within one class.

    Ties follow the same ordering as the previous brute-force scan because the
    IDs are sorted first and `pdist` emits the upper-triangular pairs in
    row-major order.
    """

    ordered_ids = tuple(sorted(prototype_ids))
    if len(ordered_ids) < 2:
        return None
    centers = np.vstack([proto_bank[prototype_id].b for prototype_id in ordered_ids])
    pairwise_distances = pdist(centers, metric="euclidean")
    best_index = int(np.argmin(pairwise_distances))
    left_indices, right_indices = np.triu_indices(len(ordered_ids), k=1)
    left_id = ordered_ids[int(left_indices[best_index])]
    right_id = ordered_ids[int(right_indices[best_index])]
    return float(pairwise_distances[best_index]), left_id, right_id


def rebuild_dense_proto_arrays(model: GONSModel) -> GONSModel:
    """Rebuild dense prototype arrays used by inference and calibration."""

    model.proto_ids = tuple(sorted(model.proto_bank))
    if not model.proto_ids:
        model.prototype_matrix = np.empty((0, 0), dtype=np.float64)
        model.radii_vec = np.empty((0,), dtype=np.float64)
        model.sigma_inv_stack = None
        model.shared_sigma_inv = None
        model.global_mean_proj = None
        model.knn_tree = None
        model.knn_labels = None
        model.reciprocal_matrix = None
        model.reciprocal_radii = None
        if hasattr(model, "full_prototype_matrix"):
            model.full_prototype_matrix = None
            model.full_radii_vec = None
        return model
    model.prototype_matrix = np.vstack(
        [model.proto_bank[proto_id].b for proto_id in model.proto_ids]
    )
    model.radii_vec = np.asarray(
        [max(model.proto_bank[proto_id].rho, 1e-12) for proto_id in model.proto_ids],
        dtype=np.float64,
    )
    if any(model.proto_bank[proto_id].use_mahalanobis for proto_id in model.proto_ids):
        dimension = int(model.prototype_matrix.shape[1])
        model.sigma_inv_stack = np.stack(
            [
                (
                    model.proto_bank[proto_id].sigma_inv
                    if model.proto_bank[proto_id].sigma_inv is not None
                    else np.eye(dimension, dtype=np.float64)
                )
                for proto_id in model.proto_ids
            ],
            axis=0,
        )
    else:
        model.sigma_inv_stack = None

    if hasattr(model, "full_prototype_matrix"):
        if model.config.enable_dual_space_scoring and all(
            model.proto_bank[proto_id].b_full is not None for proto_id in model.proto_ids
        ):
            model.full_prototype_matrix = np.vstack(
                [model.proto_bank[proto_id].b_full for proto_id in model.proto_ids]
            )
            model.full_radii_vec = np.asarray(
                [
                    max(model.proto_bank[proto_id].rho_full or 1e-12, 1e-12)
                    for proto_id in model.proto_ids
                ],
                dtype=np.float64,
            )
        else:
            model.full_prototype_matrix = None
            model.full_radii_vec = None

    if model.config.enable_reciprocal_points:
        reciprocal_matrix = np.zeros_like(model.prototype_matrix)
        reciprocal_radii = np.zeros(len(model.proto_ids), dtype=np.float64)
        for proto_index, proto_id in enumerate(model.proto_ids):
            own_class = model.proto_bank[proto_id].class_label
            other_indices = [
                other_index
                for other_index, other_id in enumerate(model.proto_ids)
                if model.proto_bank[other_id].class_label != own_class
            ]
            if other_indices:
                reciprocal_matrix[proto_index] = np.mean(
                    model.prototype_matrix[other_indices],
                    axis=0,
                    dtype=np.float64,
                )
                reciprocal_radii[proto_index] = float(
                    np.mean(model.radii_vec[other_indices], dtype=np.float64)
                )
            else:
                reciprocal_matrix[proto_index] = model.prototype_matrix[proto_index] + 10.0
                reciprocal_radii[proto_index] = 1.0
        model.reciprocal_matrix = reciprocal_matrix
        model.reciprocal_radii = reciprocal_radii
    else:
        model.reciprocal_matrix = None
        model.reciprocal_radii = None

    projected_blocks: list[NDArray[np.float64]] = []
    projected_weights: list[NDArray[np.float64]] = []
    projected_labels: list[str] = []
    class_blocks: dict[str, list[NDArray[np.float64]]] = {}
    class_weights: dict[str, list[NDArray[np.float64]]] = {}
    if model.W_proj is not None:
        for bundle in model.support_store.values():
            Z_q = np.asarray(bundle.phi_bundle @ model.W_proj, dtype=np.float64)
            if Z_q.size == 0:
                continue
            weights = np.asarray(bundle.w_bundle, dtype=np.float64)
            projected_blocks.append(Z_q)
            projected_weights.append(weights)
            projected_labels.extend([bundle.class_label] * Z_q.shape[0])
            class_blocks.setdefault(bundle.class_label, []).append(Z_q)
            class_weights.setdefault(bundle.class_label, []).append(weights)

    if model.config.scoring_mode == "relative_mahalanobis" and projected_blocks:
        Z_all = np.vstack(projected_blocks)
        w_all = np.concatenate(projected_weights)
        model.global_mean_proj = weighted_mean(Z_all, w_all)
        tied_covariance = np.zeros((Z_all.shape[1], Z_all.shape[1]), dtype=np.float64)
        total_weight = 0.0
        for class_label, blocks in class_blocks.items():
            class_Z = np.vstack(blocks)
            class_w = np.concatenate(class_weights[class_label])
            class_weight = float(np.sum(class_w))
            if class_weight <= 0.0:
                continue
            class_center = weighted_mean(class_Z, class_w)
            tied_covariance += class_weight * _weighted_covariance(class_Z, class_w, class_center)
            total_weight += class_weight
        tied_covariance = tied_covariance / max(total_weight, 1e-12)
        tied_covariance = (
            0.5 * (tied_covariance + tied_covariance.T)
            + 1e-6 * np.eye(tied_covariance.shape[0], dtype=np.float64)
        )
        model.shared_sigma_inv = np.linalg.pinv(tied_covariance)
    else:
        model.shared_sigma_inv = None
        model.global_mean_proj = None

    if model.config.enable_knn_scoring and projected_blocks:
        from sklearn.neighbors import BallTree

        Z_train = np.vstack(projected_blocks)
        model.knn_tree = BallTree(Z_train, leaf_size=40)
        model.knn_labels = np.asarray(projected_labels, dtype=object)
    else:
        model.knn_tree = None
        model.knn_labels = None
    return model


def refresh_prototype_tau_overrides(model: GONSModel) -> GONSModel:
    """Refresh strict tau overrides after global threshold recalibration."""

    updated_bank: dict[str, Prototype] = {}
    for proto_id, prototype in model.proto_bank.items():
        tau_override = prototype.tau_override
        if prototype.status != "stable":
            tau_override = model.config.fragile_tau_mult * model.tau_global
        updated_bank[proto_id] = Prototype(
            prototype_id=prototype.prototype_id,
            b=prototype.b,
            rho=prototype.rho,
            class_label=prototype.class_label,
            member_q_ids=prototype.member_q_ids,
            support_weight=prototype.support_weight,
            status=prototype.status,
            tau_override=tau_override,
            sigma_inv=prototype.sigma_inv,
            use_mahalanobis=prototype.use_mahalanobis,
            tau_local=prototype.tau_local,
            b_full=prototype.b_full,
            rho_full=prototype.rho_full,
            diag_precision=prototype.diag_precision,
        )
    model.proto_bank = updated_bank
    return model


def nearest_normalized_prototype(
    z: NDArray[np.float64],
    prototype_matrix: NDArray[np.float64],
    proto_ids: tuple[str, ...],
    radii_vec: NDArray[np.float64],
) -> tuple[str, float]:
    """Return the nearest prototype under normalized Euclidean distance."""

    return nearest_prototype(
        z,
        prototype_matrix,
        proto_ids,
        radii_vec=radii_vec,
        use_mahalanobis=False,
    )


def serialize_projection_meta(projection_meta: ProjectionResult) -> dict[str, float | int | str]:
    """Convert projection metadata into a JSON-friendly summary."""

    return {
        "eps_null_used": float(projection_meta.eps_null_used),
        "min_lambda_raw": float(projection_meta.min_lambda_raw),
        "mode": projection_meta.mode,
        "n_null_dirs": int(projection_meta.n_null_dirs),
        "projection_dim": int(projection_meta.W_proj.shape[1]),
        "range_rank": int(projection_meta.range_rank),
        "range_tol_used": float(projection_meta.range_tol_used),
        "small_solver_mode": projection_meta.small_solver_mode,
    }


def serialize_threshold_meta(threshold_meta: ThresholdMetadata) -> dict[str, float | str]:
    """Convert threshold metadata into a JSON-friendly mapping."""

    return {
        key: (float(value) if isinstance(value, (float, int)) else value)
        for key, value in asdict(threshold_meta).items()
    }
