"""Calibration reservoirs, threshold selection, and scoring-state fitting."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import chi2, genpareto, weibull_min

from gons.preprocess import transform_rows
from gons.prototype.bank import prototype_scores, reduce_prototype_scores

if TYPE_CHECKING:
    from gons.state.model import GONSModel


@dataclass(slots=True)
class CalibrationState:
    """Bounded calibration reservoirs and operating mode."""

    pos_score_reservoir_by_class: dict[str, list[float]] = field(default_factory=dict)
    imp_score_reservoir: list[float] = field(default_factory=list)
    unknown_score_reservoir: list[float] = field(default_factory=list)
    mode: str | None = None


def trim_reservoir(scores: list[float], max_size: int) -> list[float]:
    """Retain only the most recent bounded slice of a score reservoir."""

    if len(scores) <= max_size:
        return scores
    return scores[-max_size:]


def class_balanced_positive_scores(calibration_state: CalibrationState) -> NDArray[np.float64]:
    """Build a class-balanced positive-score array."""

    available = [
        scores
        for scores in calibration_state.pos_score_reservoir_by_class.values()
        if scores
    ]
    if not available:
        return np.array([], dtype=np.float64)
    per_class = min(len(scores) for scores in available)
    balanced = [np.asarray(scores[-per_class:], dtype=np.float64) for scores in available]
    return np.concatenate(balanced) if balanced else np.array([], dtype=np.float64)


def _prototype_groups(
    model: GONSModel,
    class_label: str,
) -> tuple[list[str], list[str]]:
    stable_statuses = {"stable", "fragile_stable"}
    same_class = [
        proto_id
        for proto_id, prototype in model.proto_bank.items()
        if prototype.class_label == class_label and prototype.status in stable_statuses
    ]
    other_class = [
        proto_id
        for proto_id, prototype in model.proto_bank.items()
        if prototype.class_label != class_label and prototype.status in stable_statuses
    ]
    return same_class, other_class


def _min_prototype_distance(
    z: NDArray[np.float64],
    model: GONSModel,
    proto_ids: list[str],
) -> float:
    return score_projected_row_for_calibration(z, model, proto_ids=proto_ids)


def score_projected_row_for_calibration(
    z: NDArray[np.float64],
    model: GONSModel,
    *,
    proto_ids: list[str] | None = None,
) -> float:
    """Score one projected sample with the model's active scoring method."""

    if model.prototype_matrix is None or not model.proto_ids:
        raise ValueError("Prototype bank must be available for calibration scoring.")
    candidate_proto_ids = proto_ids or list(model.proto_ids)
    if not candidate_proto_ids:
        raise ValueError("Expected at least one prototype id.")

    proto_index = {proto_id: index for index, proto_id in enumerate(model.proto_ids)}
    candidate_indices = [proto_index[proto_id] for proto_id in candidate_proto_ids]
    matrix = np.vstack([model.proto_bank[proto_id].b for proto_id in candidate_proto_ids])
    z_row = np.asarray(z, dtype=np.float64)

    if model.config.enable_knn_scoring and model.knn_tree is not None and model.knn_labels is not None:
        k = min(int(model.config.knn_k), max(int(len(model.knn_labels)), 1))
        distances, _ = model.knn_tree.query(z_row[None, :], k=k)
        return float(distances[0, -1])

    if model.config.enable_ghost_threshold:
        diag_precision = np.stack(
            [
                (
                    model.proto_bank[proto_id].diag_precision
                    if model.proto_bank[proto_id].diag_precision is not None
                    else np.ones(matrix.shape[1], dtype=np.float64)
                )
                for proto_id in candidate_proto_ids
            ],
            axis=0,
        )
        diff = matrix - z_row[None, :]
        chi2_values = np.sum(diff**2 * diag_precision, axis=1)
        max_p = float(np.max(chi2.sf(chi2_values, df=matrix.shape[1])))
        return float(1.0 - max_p)

    if (
        model.config.enable_reciprocal_points
        and model.reciprocal_matrix is not None
        and model.reciprocal_radii is not None
    ):
        radii = np.asarray(
            [max(model.proto_bank[proto_id].rho, 1e-12) for proto_id in candidate_proto_ids],
            dtype=np.float64,
        )
        reciprocal_matrix = np.asarray(model.reciprocal_matrix[candidate_indices], dtype=np.float64)
        reciprocal_radii = np.asarray(
            np.maximum(model.reciprocal_radii[candidate_indices], 1e-12),
            dtype=np.float64,
        )
        positive = np.linalg.norm(matrix - z_row[None, :], axis=1) / radii
        negative = np.linalg.norm(reciprocal_matrix - z_row[None, :], axis=1) / reciprocal_radii
        return float(np.min(positive - float(model.config.reciprocal_lambda) * negative))

    if model.config.enable_hyperspherical_scoring:
        z_hat = z_row / max(float(np.linalg.norm(z_row)), 1e-12)
        proto_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        proto_hat = matrix / np.maximum(proto_norms, 1e-12)
        return float(1.0 - np.max(proto_hat @ z_hat))

    if model.config.scoring_mode == "relative_mahalanobis":
        if model.shared_sigma_inv is None or model.global_mean_proj is None:
            raise ValueError(
                "Relative Mahalanobis calibration requires shared_sigma_inv and global_mean_proj."
            )
        diff_class = matrix - z_row[None, :]
        mahal_class = np.sqrt(
            np.maximum(
                np.einsum("nd,de,ne->n", diff_class, model.shared_sigma_inv, diff_class),
                0.0,
            )
        )
        diff_bg = z_row - model.global_mean_proj
        background_score = float(
            np.sqrt(max(float(diff_bg @ model.shared_sigma_inv @ diff_bg), 0.0))
        )
        return float(np.min(mahal_class) - background_score)

    radii = np.asarray([model.proto_bank[proto_id].rho for proto_id in candidate_proto_ids], dtype=np.float64)
    sigma_inv_stack = None
    if model.config.use_mahalanobis:
        dimension = int(matrix.shape[1])
        sigma_inv_stack = np.stack(
            [
                (
                    model.proto_bank[proto_id].sigma_inv
                    if model.proto_bank[proto_id].sigma_inv is not None
                    else np.eye(dimension, dtype=np.float64)
                )
                for proto_id in candidate_proto_ids
            ],
            axis=0,
        )
    scores = prototype_scores(
        z_row,
        matrix,
        radii_vec=radii,
        sigma_inv_stack=sigma_inv_stack,
        use_mahalanobis=model.config.use_mahalanobis,
    )
    return reduce_prototype_scores(
        scores,
        scoring_mode=model.config.scoring_mode,
        temperature=model.config.energy_temperature,
    )


def fit_evm_weibull_per_class(
    model: GONSModel,
) -> dict[str, tuple[float, float, float]]:
    """Fit one Weibull tail model per class over own-class prototype distances."""

    weibull_params: dict[str, tuple[float, float, float]] = {}
    if model.W_proj is None or model.prototype_matrix is None:
        return weibull_params

    tail_size = int(model.config.evm_tail_size)
    for class_label in model.class_registry:
        class_proto_ids, _ = _prototype_groups(model, class_label)
        if not class_proto_ids:
            continue
        class_scores: list[float] = []
        matrix = np.vstack([model.proto_bank[proto_id].b for proto_id in class_proto_ids])
        radii = np.asarray(
            [model.proto_bank[proto_id].rho for proto_id in class_proto_ids],
            dtype=np.float64,
        )
        sigma_inv_stack = None
        if model.config.use_mahalanobis:
            dimension = int(matrix.shape[1])
            sigma_inv_stack = np.stack(
                [
                    (
                        model.proto_bank[proto_id].sigma_inv
                        if model.proto_bank[proto_id].sigma_inv is not None
                        else np.eye(dimension, dtype=np.float64)
                    )
                    for proto_id in class_proto_ids
                ],
                axis=0,
            )
        for bundle in model.support_store.values():
            if bundle.class_label != class_label:
                continue
            projected = np.asarray(bundle.phi_bundle @ model.W_proj, dtype=np.float64)
            for row in projected:
                own_scores = prototype_scores(
                    row,
                    matrix,
                    radii_vec=radii,
                    sigma_inv_stack=sigma_inv_stack,
                    use_mahalanobis=model.config.use_mahalanobis,
                )
                class_scores.append(float(np.min(own_scores)))
        if len(class_scores) < 3:
            continue
        tail = np.sort(np.asarray(class_scores, dtype=np.float64))[
            -min(tail_size, len(class_scores)) :
        ]
        if tail.size < 3:
            continue
        try:
            shape, loc, scale = weibull_min.fit(tail, floc=0.0)
        except Exception:
            continue
        weibull_params[class_label] = (float(shape), float(loc), float(scale))
    return weibull_params


def update_calibration_state(
    model: GONSModel,
    labeled_eval_df: pd.DataFrame,
    *,
    optional_unknown_records: Iterable[NDArray[np.float64]] | None = None,
) -> GONSModel:
    """Populate bounded positive, impostor, and optional unknown score reservoirs."""

    X_eval, _ = transform_rows(
        model.schema,
        model.preprocessor,
        labeled_eval_df,
        label_column=model.config.label_column,
    )
    Phi_eval = model.phi.transform(X_eval)
    y_eval = labeled_eval_df[model.config.label_column].astype(str).to_numpy(dtype=object)

    for row_index, class_label_obj in enumerate(y_eval):
        z_i = np.asarray(Phi_eval[row_index] @ model.W_proj, dtype=np.float64)
        class_label = str(class_label_obj)
        same_class, other_class = _prototype_groups(model, class_label)
        if same_class:
            score = _min_prototype_distance(z_i, model, same_class)
            reservoir = model.calibration_state.pos_score_reservoir_by_class.setdefault(
                class_label,
                [],
            )
            reservoir.append(float(score))
            model.calibration_state.pos_score_reservoir_by_class[class_label] = trim_reservoir(
                reservoir,
                model.config.calibration_reservoir_size,
            )
        if (
            model.config.enable_impostor_calibration
            and other_class
            and not model.config.enable_knn_scoring
        ):
            score = _min_prototype_distance(z_i, model, other_class)
            model.calibration_state.imp_score_reservoir.append(float(score))

    if optional_unknown_records is not None:
        if (
            model.W_proj is None
            or model.prototype_matrix is None
            or model.radii_vec is None
        ):
            raise ValueError(
                "Unknown-score calibration requires a fitted projection "
                "and prototype bank."
            )
        W_proj = model.W_proj
        for phi_x in optional_unknown_records:
            z_u = np.asarray(phi_x @ W_proj, dtype=np.float64)
            score = score_projected_row_for_calibration(z_u, model)
            model.calibration_state.unknown_score_reservoir.append(float(score))

    model.calibration_state.imp_score_reservoir = trim_reservoir(
        model.calibration_state.imp_score_reservoir,
        model.config.calibration_reservoir_size,
    )
    model.calibration_state.unknown_score_reservoir = trim_reservoir(
        model.calibration_state.unknown_score_reservoir,
        model.config.calibration_reservoir_size,
    )
    return model


def recalibrate_global_threshold_grid_search(model: GONSModel) -> float:
    """Choose the threshold with the legacy constrained quantile grid search."""

    pos_scores = class_balanced_positive_scores(model.calibration_state)
    if pos_scores.size == 0:
        model.calibration_state.mode = "insufficient"
        return float(model.tau_global)

    negatives = np.asarray(
        (
            model.calibration_state.imp_score_reservoir
            + model.calibration_state.unknown_score_reservoir
        ),
        dtype=np.float64,
    )
    if negatives.size > 0:
        combined = np.concatenate([pos_scores, negatives])
        quantiles = np.linspace(0.0, 1.0, model.config.threshold_grid_size)
        candidate_grid = np.unique(np.quantile(combined, quantiles))
        feasible = []
        for tau in candidate_grid:
            false_known_rate = float(np.mean(negatives <= tau))
            true_support_coverage = float(np.mean(pos_scores <= tau))
            if (
                false_known_rate <= model.config.alpha_false_known
                and true_support_coverage >= model.config.beta_true_support
            ):
                feasible.append(float(tau))
        if feasible:
            model.calibration_state.mode = "full"
            return max(feasible)

    model.calibration_state.mode = "pos_only_fallback"
    return float(np.quantile(pos_scores, model.config.threshold_quantile))


def recalibrate_global_threshold_evt(model: GONSModel) -> float:
    """Choose the threshold from GPD tail fits over positive scores."""

    pos_by_class = model.calibration_state.pos_score_reservoir_by_class
    if not pos_by_class:
        model.calibration_state.mode = "insufficient"
        return float(model.tau_global)

    tail_size = int(model.config.evt_tail_size)
    gpd_params: dict[str, tuple[float, float, float]] = {}
    for class_label, scores in pos_by_class.items():
        if len(scores) < tail_size:
            continue
        sorted_scores = np.sort(np.asarray(scores, dtype=np.float64))
        if sorted_scores.size == 0:
            continue
        if sorted_scores.size > tail_size:
            threshold_u = float(sorted_scores[-(tail_size + 1)])
        else:
            threshold_u = float(sorted_scores[0])
        exceedances = sorted_scores[sorted_scores > threshold_u] - threshold_u
        if exceedances.size < max(tail_size // 2, 3):
            continue
        try:
            shape, _, scale = genpareto.fit(exceedances, floc=0)
        except Exception:
            continue
        gpd_params[class_label] = (float(shape), threshold_u, float(scale))

    pos_scores = class_balanced_positive_scores(model.calibration_state)
    if pos_scores.size == 0:
        model.calibration_state.mode = "insufficient"
        return float(model.tau_global)
    if not gpd_params:
        model.calibration_state.mode = "evt_fallback"
        return float(np.quantile(pos_scores, model.config.threshold_quantile))

    target = float(model.config.evt_target_coverage)
    tau_candidates: list[float] = []
    for class_label, (shape, threshold_u, scale) in gpd_params.items():
        total_scores = len(pos_by_class[class_label])
        tail_fraction = tail_size / max(total_scores, 1)
        survival_target = (1.0 - target) / max(tail_fraction, 1e-12)
        survival_target = min(max(survival_target, 0.0), 1.0)
        try:
            exceedance_quantile = float(
                genpareto.isf(survival_target, shape, loc=0, scale=scale)
            )
        except Exception:
            continue
        tau_candidates.append(threshold_u + max(exceedance_quantile, 0.0))

    if tau_candidates:
        model.calibration_state.mode = "evt"
        return float(max(tau_candidates))

    model.calibration_state.mode = "evt_fallback"
    return float(np.quantile(pos_scores, model.config.threshold_quantile))


def recalibrate_global_threshold_loco(model: GONSModel) -> float:
    """Choose the threshold from leave-one-class-out score constraints."""

    pos_by_class = model.calibration_state.pos_score_reservoir_by_class
    if len(pos_by_class) < 2:
        model.calibration_state.mode = "insufficient"
        return float(model.tau_global)

    tau_upper_per_class: list[float] = []
    for held_out_scores in pos_by_class.values():
        if not held_out_scores:
            continue
        tau_upper_per_class.append(
            float(np.quantile(held_out_scores, model.config.beta_true_support))
        )

    if not tau_upper_per_class:
        model.calibration_state.mode = "insufficient"
        return float(model.tau_global)

    tau_max_known = min(tau_upper_per_class)
    negatives = np.asarray(
        model.calibration_state.imp_score_reservoir
        + model.calibration_state.unknown_score_reservoir,
        dtype=np.float64,
    )
    if negatives.size == 0:
        model.calibration_state.mode = "loco_pos_only"
        return float(tau_max_known)

    tau_min_unknown = float(
        np.quantile(negatives, 1.0 - model.config.alpha_false_known)
    )
    tau_global = min(tau_max_known, max(tau_min_unknown, tau_max_known * 0.8))
    model.calibration_state.mode = "loco"
    return float(tau_global)


def fit_per_prototype_thresholds(model: GONSModel) -> dict[str, float]:
    """Fit EVT rejection thresholds for each deployed prototype independently."""

    if model.W_proj is None or not model.proto_ids:
        return {}

    tail_size = int(model.config.per_proto_evt_tail_size)
    if tail_size < 1:
        return {}
    target_coverage = float(model.config.per_proto_evt_coverage)
    fallback_tau = float(model.tau_global) * float(model.config.per_proto_tau_fallback_mult)
    thresholds: dict[str, float] = {}
    stable_statuses = {"stable", "fragile_stable"}

    for proto_id in model.proto_ids:
        prototype = model.proto_bank[proto_id]
        if prototype.status not in stable_statuses:
            thresholds[proto_id] = fallback_tau
            continue

        own_scores: list[float] = []
        for bundle in model.support_store.values():
            if bundle.class_label != prototype.class_label:
                continue
            projected = np.asarray(bundle.phi_bundle @ model.W_proj, dtype=np.float64)
            sigma_inv_stack = None
            if model.config.use_mahalanobis:
                if prototype.sigma_inv is None:
                    own_scores = []
                    break
                sigma_inv_stack = prototype.sigma_inv[None, :, :]
            for row in projected:
                score = prototype_scores(
                    row,
                    prototype.b[None, :],
                    radii_vec=np.asarray([prototype.rho], dtype=np.float64),
                    sigma_inv_stack=sigma_inv_stack,
                    use_mahalanobis=model.config.use_mahalanobis,
                )
                own_scores.append(float(score[0]))

        if len(own_scores) < tail_size:
            thresholds[proto_id] = fallback_tau
            continue

        sorted_scores = np.sort(np.asarray(own_scores, dtype=np.float64))
        if sorted_scores.size > tail_size:
            threshold_u = float(sorted_scores[-(tail_size + 1)])
        else:
            threshold_u = float(sorted_scores[0])
        exceedances = sorted_scores[sorted_scores > threshold_u] - threshold_u
        if exceedances.size < max(tail_size // 2, 3):
            thresholds[proto_id] = fallback_tau
            continue

        try:
            shape, _, scale = genpareto.fit(exceedances, floc=0)
            tail_fraction = tail_size / max(len(own_scores), 1)
            survival_target = (1.0 - target_coverage) / max(tail_fraction, 1e-12)
            survival_target = min(max(survival_target, 0.0), 1.0)
            exceedance_quantile = float(
                genpareto.isf(survival_target, shape, loc=0, scale=scale)
            )
        except Exception:
            thresholds[proto_id] = fallback_tau
            continue

        tau_local = threshold_u + max(exceedance_quantile, 0.0)
        tau_local = max(tau_local, 0.3 * float(model.tau_global))
        tau_local = min(tau_local, 2.0 * float(model.tau_global))
        thresholds[proto_id] = float(tau_local)

    return thresholds


def fit_per_class_quantile_thresholds(model: GONSModel) -> dict[str, float]:
    """Fit one rejection threshold per class from own-class calibration scores."""

    quantile = float(model.config.per_class_threshold_quantile)
    min_samples = 5
    thresholds: dict[str, float] = {}
    tau_global = float(model.tau_global)

    for class_label, scores in model.calibration_state.pos_score_reservoir_by_class.items():
        if len(scores) < min_samples:
            thresholds[class_label] = tau_global
            continue
        tau_class = float(np.quantile(np.asarray(scores, dtype=np.float64), quantile))
        tau_class = max(tau_class, 0.3 * tau_global)
        tau_class = min(tau_class, 3.0 * tau_global)
        thresholds[class_label] = tau_class

    return thresholds


def recalibrate_global_threshold(model: GONSModel) -> float:
    """Choose the current global novelty threshold for the configured mode."""

    if model.config.calibration_mode == "grid_search":
        return recalibrate_global_threshold_grid_search(model)
    if model.config.calibration_mode == "evt":
        return recalibrate_global_threshold_evt(model)
    if model.config.calibration_mode == "loco":
        return recalibrate_global_threshold_loco(model)
    if model.config.calibration_mode == "per_proto_evt":
        return recalibrate_global_threshold_evt(model)
    raise ValueError(f"Unsupported calibration mode: {model.config.calibration_mode}")
