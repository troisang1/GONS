"""Known-vs-unknown inference and bounded unknown-buffer helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import chi2, weibull_min

from bcmrnfst.config.model import UnknownBufferPolicy
from bcmrnfst.preprocess import transform_rows
from bcmrnfst.preprocess.schema import FrameLike, coerce_frame
from bcmrnfst.state.model import update_sanity_metrics

if TYPE_CHECKING:
    from bcmrnfst.state.model import BCMRNFSTModel


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Single-sample inference result in the deployed projection."""

    class_label: str
    score: float
    proto_id: str
    tau_effective: float
    projection_version: int
    phi_x: NDArray[np.float64] | None


@dataclass(frozen=True, slots=True)
class UnknownRecord:
    """Buffered unknown sample retained for later discovery or refresh review."""

    x_raw: dict[str, object]
    phi_x: NDArray[np.float64]
    score: float
    t_now: float
    projection_version: int


def _require_inference_state(
    model: BCMRNFSTModel,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64] | None,
]:
    """Return the fitted projection and dense prototype arrays."""

    if model.W_proj is None:
        raise ValueError("Model projection is not fitted.")
    if model.prototype_matrix is None or model.radii_vec is None or not model.proto_ids:
        raise ValueError("Model prototype bank is not ready for inference.")
    return model.W_proj, model.prototype_matrix, model.radii_vec, model.sigma_inv_stack


def _normalize_single_sample(
    x_raw: FrameLike,
    *,
    label_column: str,
) -> dict[str, object]:
    """Normalize a single raw sample into a serializable mapping."""

    frame = coerce_frame(x_raw)
    if len(frame) != 1:
        raise ValueError("Inference expects exactly one input row.")
    if label_column in frame.columns:
        frame = frame.drop(columns=[label_column])
    return {str(column): frame.iloc[0][column] for column in frame.columns}


def tau_effective(model: BCMRNFSTModel, proto_id: str) -> float:
    """Return the effective acceptance threshold for one deployed prototype."""

    prototype = model.proto_bank[proto_id]
    if model.config.enable_per_class_quantile_threshold:
        class_tau = model.per_class_thresholds.get(prototype.class_label)
        if class_tau is not None:
            if prototype.tau_override is not None:
                return float(min(class_tau, prototype.tau_override))
            return float(class_tau)
    if model.config.enable_per_proto_threshold and prototype.tau_local is not None:
        if prototype.status == "stable":
            return float(prototype.tau_local)
        if prototype.tau_override is not None:
            return float(min(prototype.tau_local, prototype.tau_override))
        return float(prototype.tau_local)
    if prototype.status == "stable":
        return float(model.tau_global)
    if prototype.tau_override is None:
        raise ValueError(f"Prototype '{proto_id}' requires a tau override.")
    return float(min(model.tau_global, prototype.tau_override))


def _prototype_prediction_tables(
    model: BCMRNFSTModel,
) -> tuple[NDArray[np.float64], tuple[str, ...], tuple[str, ...]]:
    """Return per-prototype thresholds and class labels aligned to `model.proto_ids`."""

    tau_by_proto = np.asarray(
        [tau_effective(model, proto_id) for proto_id in model.proto_ids],
        dtype=np.float64,
    )
    class_by_proto = tuple(model.proto_bank[proto_id].class_label for proto_id in model.proto_ids)
    return tau_by_proto, model.proto_ids, class_by_proto


def _nearest_normalized_prototypes_batch(
    projected_rows: NDArray[np.float64],
    prototype_matrix: NDArray[np.float64],
    radii_vec: NDArray[np.float64],
    *,
    block_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return nearest prototype indices and normalized scores for many rows."""

    if block_size < 1:
        raise ValueError("block_size must be at least 1.")
    if projected_rows.ndim != 2:
        raise ValueError("projected_rows must be a 2D array.")
    if projected_rows.shape[0] == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    safe_radii = np.maximum(radii_vec, 1e-12)
    best_indices = np.empty(projected_rows.shape[0], dtype=np.int64)
    best_scores = np.empty(projected_rows.shape[0], dtype=np.float64)

    for start in range(0, projected_rows.shape[0], block_size):
        stop = min(start + block_size, projected_rows.shape[0])
        block = projected_rows[start:stop]
        distances = np.linalg.norm(prototype_matrix[None, :, :] - block[:, None, :], axis=2)
        normalized_scores = distances / safe_radii[None, :]
        block_minima = normalized_scores.min(axis=1, keepdims=True)
        tied_minima = np.isclose(normalized_scores, block_minima, rtol=0.0, atol=1e-12)
        block_indices = np.argmax(tied_minima, axis=1)
        row_offsets = np.arange(stop - start)
        best_indices[start:stop] = block_indices
        best_scores[start:stop] = normalized_scores[row_offsets, block_indices]

    return best_indices, best_scores


def _nearest_mahalanobis_prototypes_batch(
    projected_rows: NDArray[np.float64],
    prototype_matrix: NDArray[np.float64],
    sigma_inv_stack: NDArray[np.float64],
    *,
    block_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return nearest prototype indices and Mahalanobis scores for many rows."""

    if block_size < 1:
        raise ValueError("block_size must be at least 1.")
    if projected_rows.ndim != 2:
        raise ValueError("projected_rows must be a 2D array.")
    if projected_rows.shape[0] == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    best_indices = np.empty(projected_rows.shape[0], dtype=np.int64)
    best_scores = np.empty(projected_rows.shape[0], dtype=np.float64)

    for start in range(0, projected_rows.shape[0], block_size):
        stop = min(start + block_size, projected_rows.shape[0])
        block = projected_rows[start:stop]
        diff = prototype_matrix[None, :, :] - block[:, None, :]
        quadratic = np.einsum("bnd,ndk,bnk->bn", diff, sigma_inv_stack, diff)
        scores = np.sqrt(np.maximum(quadratic, 0.0))
        block_minima = scores.min(axis=1, keepdims=True)
        tied_minima = np.isclose(scores, block_minima, rtol=0.0, atol=1e-12)
        block_indices = np.argmax(tied_minima, axis=1)
        row_offsets = np.arange(stop - start)
        best_indices[start:stop] = block_indices
        best_scores[start:stop] = scores[row_offsets, block_indices]

    return best_indices, best_scores


def _energy_score_batch(
    projected_rows: NDArray[np.float64],
    prototype_matrix: NDArray[np.float64],
    *,
    radii_vec: NDArray[np.float64] | None,
    sigma_inv_stack: NDArray[np.float64] | None,
    use_mahalanobis: bool,
    temperature: float,
    block_size: int,
    per_proto_temperature: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return nearest prototype indices and energy-reduced scores for many rows.

    If *per_proto_temperature* is provided, each prototype uses its own
    temperature in the logsumexp reduction (class-conditional temperature).
    """

    if block_size < 1:
        raise ValueError("block_size must be at least 1.")
    if projected_rows.ndim != 2:
        raise ValueError("projected_rows must be a 2D array.")
    if projected_rows.shape[0] == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    temp = max(float(temperature), 1e-12)
    best_indices = np.empty(projected_rows.shape[0], dtype=np.int64)
    energy_scores = np.empty(projected_rows.shape[0], dtype=np.float64)
    safe_radii = None if radii_vec is None else np.maximum(radii_vec, 1e-12)

    for start in range(0, projected_rows.shape[0], block_size):
        stop = min(start + block_size, projected_rows.shape[0])
        block = projected_rows[start:stop]
        if use_mahalanobis:
            if sigma_inv_stack is None:
                raise ValueError("Mahalanobis inference requires per-prototype inverse covariances.")
            diff = prototype_matrix[None, :, :] - block[:, None, :]
            quadratic = np.einsum("bnd,ndk,bnk->bn", diff, sigma_inv_stack, diff)
            scores = np.sqrt(np.maximum(quadratic, 0.0))
        else:
            if safe_radii is None:
                raise ValueError("Energy scoring requires prototype radii.")
            distances = np.linalg.norm(prototype_matrix[None, :, :] - block[:, None, :], axis=2)
            scores = distances / safe_radii[None, :]
        block_minima = scores.min(axis=1, keepdims=True)
        tied_minima = np.isclose(scores, block_minima, rtol=0.0, atol=1e-12)
        block_indices = np.argmax(tied_minima, axis=1)
        best_indices[start:stop] = block_indices
        if per_proto_temperature is not None:
            # Per-prototype temperature: higher temp for high-support classes
            # makes their energy contribution flatter.
            ppt = np.maximum(per_proto_temperature, 1e-12)[None, :]  # (1, P)
            energy_scores[start:stop] = -temp * logsumexp(-scores / ppt, axis=1)
        else:
            energy_scores[start:stop] = -temp * logsumexp(-scores / temp, axis=1)

    return best_indices, energy_scores


def _relative_mahalanobis_score_batch(
    projected_rows: NDArray[np.float64],
    prototype_matrix: NDArray[np.float64],
    shared_sigma_inv: NDArray[np.float64],
    global_mean: NDArray[np.float64],
    *,
    block_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Score rows with relative Mahalanobis distance."""

    best_indices = np.empty(projected_rows.shape[0], dtype=np.int64)
    best_scores = np.empty(projected_rows.shape[0], dtype=np.float64)

    for start in range(0, projected_rows.shape[0], block_size):
        stop = min(start + block_size, projected_rows.shape[0])
        block = projected_rows[start:stop]
        diff_class = prototype_matrix[None, :, :] - block[:, None, :]
        mahal_class = np.sqrt(
            np.maximum(
                np.einsum("bqd,de,bqe->bq", diff_class, shared_sigma_inv, diff_class),
                0.0,
            )
        )
        block_indices = np.argmin(mahal_class, axis=1)
        row_indices = np.arange(stop - start)
        d_class = mahal_class[row_indices, block_indices]

        diff_bg = block - global_mean[None, :]
        d_background = np.sqrt(
            np.maximum(
                np.einsum("bd,de,be->b", diff_bg, shared_sigma_inv, diff_bg),
                0.0,
            )
        )

        best_indices[start:stop] = block_indices
        best_scores[start:stop] = d_class - d_background

    return best_indices, best_scores


def _ghost_chi2_score_batch(
    projected_rows: NDArray[np.float64],
    prototype_matrix: NDArray[np.float64],
    diag_precision_stack: NDArray[np.float64],
    *,
    block_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Score rows with GHOST chi-squared p-values."""

    best_indices = np.empty(projected_rows.shape[0], dtype=np.int64)
    best_scores = np.empty(projected_rows.shape[0], dtype=np.float64)
    degrees_of_freedom = int(prototype_matrix.shape[1])

    for start in range(0, projected_rows.shape[0], block_size):
        stop = min(start + block_size, projected_rows.shape[0])
        block = projected_rows[start:stop]
        diff = prototype_matrix[None, :, :] - block[:, None, :]
        chi2_values = np.sum(diff**2 * diag_precision_stack[None, :, :], axis=2)
        p_values = chi2.sf(chi2_values, df=degrees_of_freedom)
        block_indices = np.argmax(p_values, axis=1)
        row_indices = np.arange(stop - start)
        best_indices[start:stop] = block_indices
        best_scores[start:stop] = 1.0 - p_values[row_indices, block_indices]

    return best_indices, best_scores


def _knn_score_batch(
    projected_rows: NDArray[np.float64],
    knn_tree: object,
    knn_labels: NDArray[np.object_],
    *,
    k: int,
    class_to_proto_index: dict[str, int],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Score rows with k-nearest-neighbor distances in projected space."""

    k_eff = min(max(int(k), 1), max(int(len(knn_labels)), 1))
    distances, neighbor_indices = knn_tree.query(projected_rows, k=k_eff)
    best_indices = np.empty(projected_rows.shape[0], dtype=np.int64)
    best_scores = np.asarray(distances[:, -1], dtype=np.float64)

    for row_index in range(projected_rows.shape[0]):
        label_counts = Counter(str(label) for label in knn_labels[neighbor_indices[row_index]])
        predicted_label = label_counts.most_common(1)[0][0]
        best_indices[row_index] = int(class_to_proto_index.get(predicted_label, 0))

    return best_indices, best_scores


def _reciprocal_score_batch(
    projected_rows: NDArray[np.float64],
    prototype_matrix: NDArray[np.float64],
    radii_vec: NDArray[np.float64],
    reciprocal_matrix: NDArray[np.float64],
    reciprocal_radii: NDArray[np.float64],
    reciprocal_lambda: float,
    *,
    block_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Score rows against prototypes and reciprocal anti-prototypes."""

    safe_radii = np.maximum(radii_vec, 1e-12)
    safe_reciprocal_radii = np.maximum(reciprocal_radii, 1e-12)
    best_indices = np.empty(projected_rows.shape[0], dtype=np.int64)
    best_scores = np.empty(projected_rows.shape[0], dtype=np.float64)

    for start in range(0, projected_rows.shape[0], block_size):
        stop = min(start + block_size, projected_rows.shape[0])
        block = projected_rows[start:stop]
        d_positive = np.linalg.norm(prototype_matrix[None, :, :] - block[:, None, :], axis=2)
        d_positive = d_positive / safe_radii[None, :]
        d_negative = np.linalg.norm(reciprocal_matrix[None, :, :] - block[:, None, :], axis=2)
        d_negative = d_negative / safe_reciprocal_radii[None, :]
        combined = d_positive - float(reciprocal_lambda) * d_negative
        block_indices = np.argmin(combined, axis=1)
        row_indices = np.arange(stop - start)
        best_indices[start:stop] = block_indices
        best_scores[start:stop] = combined[row_indices, block_indices]

    return best_indices, best_scores


def _hyperspherical_score_batch(
    projected_rows: NDArray[np.float64],
    prototype_matrix: NDArray[np.float64],
    *,
    block_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Score rows with cosine similarity on the unit hypersphere."""

    proto_norms = np.linalg.norm(prototype_matrix, axis=1, keepdims=True)
    proto_hat = prototype_matrix / np.maximum(proto_norms, 1e-12)
    best_indices = np.empty(projected_rows.shape[0], dtype=np.int64)
    best_scores = np.empty(projected_rows.shape[0], dtype=np.float64)

    for start in range(0, projected_rows.shape[0], block_size):
        stop = min(start + block_size, projected_rows.shape[0])
        block = projected_rows[start:stop]
        block_norms = np.linalg.norm(block, axis=1, keepdims=True)
        block_hat = block / np.maximum(block_norms, 1e-12)
        similarities = block_hat @ proto_hat.T
        block_indices = np.argmax(similarities, axis=1)
        row_indices = np.arange(stop - start)
        best_indices[start:stop] = block_indices
        best_scores[start:stop] = 1.0 - similarities[row_indices, block_indices]

    return best_indices, best_scores


def _evm_score_batch(
    projected_rows: NDArray[np.float64],
    prototype_matrix: NDArray[np.float64],
    *,
    class_by_proto: tuple[str, ...],
    evm_weibull_params: dict[str, tuple[float, float, float]],
    radii_vec: NDArray[np.float64] | None,
    sigma_inv_stack: NDArray[np.float64] | None,
    use_mahalanobis: bool,
    block_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Score rows with per-class EVM inclusion probabilities."""

    class_to_indices: dict[str, list[int]] = {}
    for proto_index, class_label in enumerate(class_by_proto):
        class_to_indices.setdefault(class_label, []).append(proto_index)

    if use_mahalanobis:
        if sigma_inv_stack is None:
            raise ValueError("EVM Mahalanobis scoring requires per-prototype inverse covariances.")
    else:
        if radii_vec is None:
            raise ValueError("EVM scoring requires prototype radii.")
        safe_radii = np.maximum(radii_vec, 1e-12)

    fitted_classes = [class_label for class_label in class_by_proto if class_label in evm_weibull_params]
    unique_fitted_classes = tuple(dict.fromkeys(fitted_classes))
    if not unique_fitted_classes:
        raise ValueError("EVM scoring requires at least one fitted class Weibull model.")

    best_indices = np.empty(projected_rows.shape[0], dtype=np.int64)
    best_scores = np.empty(projected_rows.shape[0], dtype=np.float64)

    for start in range(0, projected_rows.shape[0], block_size):
        stop = min(start + block_size, projected_rows.shape[0])
        block = projected_rows[start:stop]
        if use_mahalanobis:
            diff = prototype_matrix[None, :, :] - block[:, None, :]
            raw_scores = np.sqrt(
                np.maximum(
                    np.einsum("bnd,ndk,bnk->bn", diff, sigma_inv_stack, diff),
                    0.0,
                )
            )
        else:
            distances = np.linalg.norm(prototype_matrix[None, :, :] - block[:, None, :], axis=2)
            raw_scores = distances / safe_radii[None, :]

        class_scores = np.empty((stop - start, len(unique_fitted_classes)), dtype=np.float64)
        class_proto_indices = np.empty((stop - start, len(unique_fitted_classes)), dtype=np.int64)
        for class_offset, class_label in enumerate(unique_fitted_classes):
            indices = np.asarray(class_to_indices[class_label], dtype=np.int64)
            class_block_scores = raw_scores[:, indices]
            best_local = np.argmin(class_block_scores, axis=1)
            row_indices = np.arange(stop - start)
            class_scores[:, class_offset] = class_block_scores[row_indices, best_local]
            class_proto_indices[:, class_offset] = indices[best_local]

        inclusion = np.empty_like(class_scores)
        for class_offset, class_label in enumerate(unique_fitted_classes):
            shape, loc, scale = evm_weibull_params[class_label]
            inclusion[:, class_offset] = weibull_min.sf(
                class_scores[:, class_offset],
                shape,
                loc=loc,
                scale=max(scale, 1e-12),
            )

        best_class_offsets = np.argmax(inclusion, axis=1)
        row_indices = np.arange(stop - start)
        best_indices[start:stop] = class_proto_indices[row_indices, best_class_offsets]
        best_scores[start:stop] = 1.0 - inclusion[row_indices, best_class_offsets]

    return best_indices, best_scores


def _score_projected_rows(
    model: BCMRNFSTModel,
    projected_rows: NDArray[np.float64],
    *,
    block_size: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], bool]:
    """Score projected rows with the active phase-2 method and return row thresholds."""

    _, prototype_matrix, radii_vec, sigma_inv_stack = _require_inference_state(model)
    tau_by_proto, _, class_by_proto = _prototype_prediction_tables(model)
    class_to_proto_index: dict[str, int] = {}
    for proto_index, class_label in enumerate(class_by_proto):
        class_to_proto_index.setdefault(class_label, proto_index)

    if model.config.enable_knn_scoring and model.knn_tree is not None and model.knn_labels is not None:
        reject_indices, best_scores = _knn_score_batch(
            projected_rows,
            model.knn_tree,
            model.knn_labels,
            k=model.config.knn_k,
            class_to_proto_index=class_to_proto_index,
        )
        tau_by_row = np.full(projected_rows.shape[0], float(model.tau_global), dtype=np.float64)
        return reject_indices, best_scores, reject_indices, tau_by_row, False

    if model.config.enable_ghost_threshold:
        diag_precision_stack = np.stack(
            [
                (
                    model.proto_bank[proto_id].diag_precision
                    if model.proto_bank[proto_id].diag_precision is not None
                    else np.ones(projected_rows.shape[1], dtype=np.float64)
                )
                for proto_id in model.proto_ids
            ],
            axis=0,
        )
        reject_indices, best_scores = _ghost_chi2_score_batch(
            projected_rows,
            prototype_matrix,
            diag_precision_stack,
            block_size=block_size,
        )
        tau_by_row = np.full(
            projected_rows.shape[0],
            1.0 - float(model.config.ghost_alpha),
            dtype=np.float64,
        )
        return reject_indices, best_scores, reject_indices, tau_by_row, False

    if (
        model.config.enable_reciprocal_points
        and model.reciprocal_matrix is not None
        and model.reciprocal_radii is not None
    ):
        reject_indices, best_scores = _reciprocal_score_batch(
            projected_rows,
            prototype_matrix,
            radii_vec,
            model.reciprocal_matrix,
            model.reciprocal_radii,
            model.config.reciprocal_lambda,
            block_size=block_size,
        )
        return (
            reject_indices,
            best_scores,
            reject_indices,
            tau_by_proto[reject_indices],
            True,
        )

    if model.config.enable_hyperspherical_scoring:
        reject_indices, best_scores = _hyperspherical_score_batch(
            projected_rows,
            prototype_matrix,
            block_size=block_size,
        )
        return (
            reject_indices,
            best_scores,
            reject_indices,
            tau_by_proto[reject_indices],
            True,
        )

    if model.config.enable_evm_calibration and model.evm_weibull_params:
        reject_indices, best_scores = _evm_score_batch(
            projected_rows,
            prototype_matrix,
            class_by_proto=class_by_proto,
            evm_weibull_params=model.evm_weibull_params,
            radii_vec=radii_vec,
            sigma_inv_stack=sigma_inv_stack,
            use_mahalanobis=model.config.use_mahalanobis,
            block_size=block_size,
        )
        tau_by_row = np.full(
            projected_rows.shape[0],
            1.0 - float(model.config.evm_inclusion_threshold),
            dtype=np.float64,
        )
        return reject_indices, best_scores, reject_indices, tau_by_row, False

    if model.config.scoring_mode == "relative_mahalanobis":
        if model.shared_sigma_inv is None or model.global_mean_proj is None:
            raise ValueError(
                "Relative Mahalanobis requires shared_sigma_inv and global_mean_proj."
            )
        reject_indices, best_scores = _relative_mahalanobis_score_batch(
            projected_rows,
            prototype_matrix,
            model.shared_sigma_inv,
            model.global_mean_proj,
            block_size=block_size,
        )
        return (
            reject_indices,
            best_scores,
            reject_indices,
            tau_by_proto[reject_indices],
            True,
        )

    if model.config.scoring_mode == "energy":
        # Build per-prototype temperatures if class-conditional mode is enabled.
        ppt: NDArray[np.float64] | None = None
        if model.config.enable_class_cond_temperature and hasattr(model, "support_store"):
            _cls_sup: dict[str, float] = {}
            for bundle in model.support_store.values():
                c = bundle.class_label
                _cls_sup[c] = _cls_sup.get(c, 0.0) + float(np.sum(bundle.w_bundle))
            if _cls_sup:
                _med = float(np.median(list(_cls_sup.values())))
                _gamma = model.config.temperature_support_gamma
                ppt = np.array(
                    [
                        model.config.energy_temperature
                        * (max(_cls_sup.get(class_by_proto[i], 1.0), 1.0) / max(_med, 1.0))
                        ** _gamma
                        for i in range(len(class_by_proto))
                    ],
                    dtype=np.float64,
                )

        reject_indices, best_scores = _energy_score_batch(
            projected_rows,
            prototype_matrix,
            radii_vec=radii_vec,
            sigma_inv_stack=sigma_inv_stack,
            use_mahalanobis=model.config.use_mahalanobis,
            temperature=model.config.energy_temperature,
            block_size=block_size,
            per_proto_temperature=ppt,
        )
    elif model.config.use_mahalanobis:
        if sigma_inv_stack is None:
            raise ValueError("Mahalanobis inference requires per-prototype inverse covariances.")
        reject_indices, best_scores = _nearest_mahalanobis_prototypes_batch(
            projected_rows,
            prototype_matrix,
            sigma_inv_stack,
            block_size=block_size,
        )
    else:
        reject_indices, best_scores = _nearest_normalized_prototypes_batch(
            projected_rows,
            prototype_matrix,
            radii_vec,
            block_size=block_size,
        )
    # --- Classification refinement: separate from rejection to avoid breaking
    #     threshold calibration.  Three mechanisms can compose:
    #     1. Support-weighted penalty: penalize high-support classes globally
    #     2. NCDR: for ambiguous samples, prefer the lower-support class
    #     3. Both can be enabled simultaneously
    class_indices_out = reject_indices.copy()

    # Gather class support once for all refinement mechanisms.
    class_support: dict[str, float] = {}
    if (
        model.config.score_support_penalty
        or model.config.enable_ncdr_classification
    ) and hasattr(model, "support_store"):
        for bundle in model.support_store.values():
            cls = bundle.class_label
            class_support[cls] = class_support.get(cls, 0.0) + float(
                np.sum(bundle.w_bundle)
            )

    use_penalty = model.config.score_support_penalty and bool(class_support)
    use_ncdr = model.config.enable_ncdr_classification and bool(class_support)

    if use_penalty or use_ncdr:
        safe_radii = np.maximum(radii_vec, 1e-12)
        median_support = float(np.median(list(class_support.values()))) if class_support else 1.0

        # Build per-prototype penalty vector for support-weighted scoring.
        penalty = np.ones(len(class_by_proto), dtype=np.float64)
        if use_penalty:
            penalty_fn = np.log if model.config.score_support_penalty_fn == "log" else np.sqrt
            raw_penalty = np.array(
                [
                    penalty_fn(max(class_support.get(class_by_proto[i], 1.0), 1.0))
                    / penalty_fn(max(median_support, 2.0))
                    for i in range(len(class_by_proto))
                ],
                dtype=np.float64,
            )
            alpha_p = model.config.score_support_penalty_alpha
            penalty = 1.0 + alpha_p * (raw_penalty - 1.0)

        # Build per-class prototype masks for NCDR.
        unique_classes = list(dict.fromkeys(class_by_proto))
        class_to_proto_idx: dict[str, NDArray[np.int64]] = {}
        if use_ncdr:
            for ci, cls in enumerate(unique_classes):
                class_to_proto_idx[cls] = np.array(
                    [i for i, c in enumerate(class_by_proto) if c == cls],
                    dtype=np.int64,
                )

        for start in range(0, projected_rows.shape[0], block_size):
            stop = min(start + block_size, projected_rows.shape[0])
            block = projected_rows[start:stop]
            dists = np.linalg.norm(
                prototype_matrix[None, :, :] - block[:, None, :], axis=2,
            )
            normed = dists / safe_radii[None, :]
            penalized = normed * penalty[None, :]

            if use_ncdr:
                # Per-class minimum penalized distance.
                n_classes = len(unique_classes)
                class_min = np.full((stop - start, n_classes), np.inf)
                class_best_proto = np.zeros((stop - start, n_classes), dtype=np.int64)
                for ci, cls in enumerate(unique_classes):
                    pmask = class_to_proto_idx[cls]
                    cls_dists = penalized[:, pmask]
                    best_local = cls_dists.argmin(axis=1)
                    class_min[:, ci] = cls_dists[np.arange(stop - start), best_local]
                    class_best_proto[:, ci] = pmask[best_local]

                sorted_ci = np.argsort(class_min, axis=1)
                ncdr_thr = model.config.ncdr_threshold
                for ri in range(stop - start):
                    top1 = int(sorted_ci[ri, 0])
                    top2 = int(sorted_ci[ri, 1]) if n_classes > 1 else top1
                    d1 = class_min[ri, top1]
                    d2 = class_min[ri, top2]
                    ratio = d1 / max(d2, 1e-12)
                    if ratio > ncdr_thr:
                        # Ambiguous: prefer lower-support class.
                        cls1 = unique_classes[top1]
                        cls2 = unique_classes[top2]
                        if class_support.get(cls1, 0) > class_support.get(cls2, 0):
                            class_indices_out[start + ri] = int(class_best_proto[ri, top2])
                        else:
                            class_indices_out[start + ri] = int(class_best_proto[ri, top1])
                    else:
                        class_indices_out[start + ri] = int(class_best_proto[ri, top1])
            else:
                class_indices_out[start:stop] = penalized.argmin(axis=1)

    return reject_indices, best_scores, class_indices_out, tau_by_proto[reject_indices], True


def predict_one(model: BCMRNFSTModel, x_raw: FrameLike) -> PredictionResult:
    """Predict a single sample under the active phase-2 scoring method."""

    results = predict_many(
        model,
        x_raw,
        retain_unknown_features=True,
        block_size=1,
    )
    if len(results) != 1:
        raise ValueError("Inference expects exactly one input row.")
    return results[0]


def _full_space_nearest_batch(
    phi_rows: NDArray[np.float64],
    full_prototype_matrix: NDArray[np.float64],
    full_radii_vec: NDArray[np.float64],
    *,
    block_size: int,
) -> NDArray[np.int64]:
    """Return nearest prototype indices in full explicit-map space (for classification only)."""

    safe_radii = np.maximum(full_radii_vec, 1e-12)
    best_indices = np.empty(phi_rows.shape[0], dtype=np.int64)
    for start in range(0, phi_rows.shape[0], block_size):
        stop = min(start + block_size, phi_rows.shape[0])
        block = phi_rows[start:stop]
        distances = np.linalg.norm(
            full_prototype_matrix[None, :, :] - block[:, None, :], axis=2,
        )
        normalized_scores = distances / safe_radii[None, :]
        block_minima = normalized_scores.min(axis=1, keepdims=True)
        tied_minima = np.isclose(normalized_scores, block_minima, rtol=0.0, atol=1e-12)
        best_indices[start:stop] = np.argmax(tied_minima, axis=1)
    return best_indices


def predict_many(
    model: BCMRNFSTModel,
    x_raw: FrameLike,
    *,
    retain_unknown_features: bool = False,
    block_size: int = 1024,
) -> list[PredictionResult]:
    """Predict many samples under the deployed projection with batched scoring."""

    W_proj, _, _, _ = _require_inference_state(model)
    X_eval, _ = transform_rows(
        model.schema,
        model.preprocessor,
        x_raw,
        label_column=model.config.label_column,
    )
    if X_eval.shape[0] == 0:
        return []

    phi_eval = np.asarray(model.phi.transform(X_eval), dtype=np.float64)
    projected_rows = np.asarray(phi_eval @ W_proj, dtype=np.float64)
    tau_by_proto, proto_ids, class_by_proto = _prototype_prediction_tables(model)
    reject_indices, best_scores, class_indices, tau_by_row, allow_dual = _score_projected_rows(
        model,
        projected_rows,
        block_size=block_size,
    )

    # --- Dual-space scoring: blend null-space rejection with full-space ---
    use_dual = (
        allow_dual
        and model.config.enable_dual_space_scoring
        and model.full_prototype_matrix is not None
        and model.full_radii_vec is not None
    )
    if use_dual:
        class_indices = _full_space_nearest_batch(
            phi_eval,
            model.full_prototype_matrix,
            model.full_radii_vec,
            block_size=block_size,
        )
        # Blend null-space rejection score with full-space distance for
        # improved rejection calibration (dual-space scoring from openworld
        # discovery project).  alpha controls the null-space weight.
        alpha = model.config.dual_space_alpha
        if alpha < 1.0:
            safe_full_radii = np.maximum(model.full_radii_vec, 1e-12)
            n_protos = model.full_prototype_matrix.shape[0]
            full_scores = np.empty(phi_eval.shape[0], dtype=np.float64)
            # Block-wise computation to avoid OOM on large datasets.
            for start in range(0, phi_eval.shape[0], block_size):
                stop = min(start + block_size, phi_eval.shape[0])
                block = phi_eval[start:stop]
                dists = np.empty((stop - start, n_protos), dtype=np.float64)
                for p in range(n_protos):
                    diffs = block - model.full_prototype_matrix[p]
                    dists[:, p] = np.sqrt(np.sum(diffs * diffs, axis=1))
                normed = dists / safe_full_radii[None, :]
                full_scores[start:stop] = normed.min(axis=1)
            # Normalize both score streams to [0, 1] range before blending.
            null_max = best_scores.max() if best_scores.max() > 0 else 1.0
            full_max = full_scores.max() if full_scores.max() > 0 else 1.0
            best_scores = alpha * (best_scores / null_max) + (1 - alpha) * (full_scores / full_max)
            best_scores = best_scores * null_max  # rescale back to null-space tau scale

    # --- PCT-RD: per-class two-tailed reference discriminant (v2 mechanism).
    #     Replaces the global gate when enabled. Provides per-class g_q score
    #     (Theorem A2 classification margin) plus per-class Q-quantile threshold
    #     (Theorem A4). Mixed with NCM via closed-form alpha (Theorem A3).
    pct_force_reject = np.zeros(len(best_scores), dtype=bool)
    if (
        getattr(model.config, "enable_pct_rd", False)
        and getattr(model, "pct_rd_state", None) is not None
        and bool(getattr(model.pct_rd_state, "cores", {}))
    ):
        from bcmrnfst.projection.pct_rd import score_pct_rd
        pct_state = model.pct_rd_state
        # Class-support amnesty for PCT-RD: low-support (K=5) classes admit
        # via NCM only — PCT-RD's per-class detector is unreliable on K=5.
        amn_min = int(getattr(model.config, "dvmad_gate_min_class_support", 0))
        cls_sup_count: dict[str, int] = {}
        for bundle in model.support_store.values():
            cls_sup_count[bundle.class_label] = cls_sup_count.get(bundle.class_label, 0) + int(bundle.size)
        # NCM distance to each class' nearest prototype, normalised by radius.
        # Reuse existing prototype_matrix + radii_vec (in projected space).
        n_classes = len(pct_state.cores)
        ordered_classes = sorted(pct_state.cores.keys())
        # Build NCM per-class distance matrix from existing per-prototype dists.
        # `best_scores` is per-row min over all protos; we need per-class min.
        proto_class_idx = np.asarray(
            [ordered_classes.index(c) if c in ordered_classes else -1
             for c in class_by_proto],
            dtype=np.int64,
        )
        # Per-class NCM: for each row, min normalised distance to protos of class q.
        n_rows = projected_rows.shape[0]
        ncm_per_class = np.full((n_rows, n_classes), np.inf, dtype=np.float64)
        # Pull projection-space prototype matrix + radii from model state.
        proto_mat = np.asarray(model.prototype_matrix, dtype=np.float64)
        radii_local = np.maximum(np.asarray(model.radii_vec, dtype=np.float64), 1e-9)
        for p_idx in range(proto_mat.shape[0]):
            cls_idx = int(proto_class_idx[p_idx])
            if cls_idx < 0:
                continue
            diff = projected_rows - proto_mat[p_idx]
            d = np.linalg.norm(diff, axis=1) / radii_local[p_idx]
            ncm_per_class[:, cls_idx] = np.minimum(ncm_per_class[:, cls_idx], d)
        ncm_per_class[~np.isfinite(ncm_per_class)] = ncm_per_class[np.isfinite(ncm_per_class)].max() if np.isfinite(ncm_per_class).any() else 1.0

        pct_scores, _ = score_pct_rd(pct_state, phi_eval)
        a_pct = float(pct_state.alpha_pct)
        a_ncm = float(pct_state.alpha_ncm)
        joint = a_ncm * ncm_per_class + a_pct * pct_scores
        joint[~np.isfinite(joint)] = np.inf
        joint_class_idx = np.argmin(joint, axis=1)

        # Existing class_indices is the NCDR-refined per-row prototype index
        # (with support-penalty + NCDR ratio refinement applied). Map it to
        # a class string to know when amnesty must preserve it.
        existing_class_per_row = np.asarray(
            [class_by_proto[int(p)] for p in class_indices],
            dtype=object,
        )
        existing_class_low = np.asarray(
            [cls_sup_count.get(c, 0) < amn_min for c in existing_class_per_row],
            dtype=bool,
        ) if amn_min > 0 else np.zeros(len(class_indices), dtype=bool)

        # Map joint argmin back to a prototype id of that class.
        first_proto_for_class: dict[str, int] = {}
        for p_idx, cls in enumerate(class_by_proto):
            if cls in ordered_classes and cls not in first_proto_for_class:
                first_proto_for_class[cls] = p_idx
        joint_proto_indices = np.asarray(
            [first_proto_for_class.get(
                ordered_classes[c] if 0 <= c < len(ordered_classes) else None,
                int(class_indices[i]),
            ) for i, c in enumerate(joint_class_idx)],
            dtype=np.int64,
        )

        # Apply override only where the existing classifier did NOT choose a
        # low-support class. Preserves NCDR/penalty refinement on K=5 admits.
        new_class_indices = np.where(existing_class_low, class_indices, joint_proto_indices)
        class_indices = new_class_indices.astype(np.int64)
        new_class_idx = joint_class_idx

        # Per-class rejection threshold (Theorem A4). Only fire reject when
        # chosen class has enough support (amnesty for K=5 admits). Use the
        # FINAL chosen class (post-amnesty) to look up g and tau.
        valid_idx = np.clip(new_class_idx, 0, pct_scores.shape[1] - 1)
        g_at_pred = pct_scores[np.arange(n_rows), valid_idx]
        eligible = np.asarray(
            [cls_sup_count.get(ordered_classes[c], 0) >= amn_min for c in valid_idx],
            dtype=bool,
        ) if amn_min > 0 else np.ones(n_rows, dtype=bool)
        tau_at_pred = np.asarray(
            [pct_state.tau_per_class.get(ordered_classes[c],
                                          pct_state.tau_global if pct_state.tau_global is not None else np.inf)
             for c in valid_idx],
            dtype=np.float64,
        )
        pct_force_reject = (g_at_pred > tau_at_pred) & eligible

        # Update best_scores so the downstream tau-compare still triggers
        # (we leave existing tau_by_row alone — pct_force_reject does the work).

    # --- EWM: eigenvalue-weighted Mahalanobis from the full T spectrum.
    #     d²_λ(x, μ_q) = Σ_j ω_j (W_jᵀ(x − μ_q))²,  ω_j = (1−λ_j)/(λ_j + δ_floor)
    #     the binary corner case is ω_j = 1[λ_j < ε ∨ λ_j > 1−ε].
    if (
        getattr(model.config, "enable_ewm", False)
        and getattr(model, "ewm_state", None) is not None
        and bool(getattr(model.ewm_state, "centroid_phi_per_class", {}))
        and float(getattr(model.config, "ewm_alpha_mix", 1.0)) > 1e-9
    ):
        from bcmrnfst.projection.dvmad_ewm import ewm_score_per_class
        es = model.ewm_state
        alpha_mix = float(getattr(model.config, "ewm_alpha_mix", 1.0))
        ewm_scores, ewm_classes = ewm_score_per_class(es, phi_eval)
        if ewm_classes:
            # Build per-row NCM-per-class to blend.
            n_rows = projected_rows.shape[0]
            n_cls = len(ewm_classes)
            ncm_per_class = np.full((n_rows, n_cls), np.inf, dtype=np.float64)
            radii_local = np.maximum(np.asarray(model.radii_vec, dtype=np.float64), 1e-9)
            proto_mat = np.asarray(model.prototype_matrix, dtype=np.float64)
            cls_to_idx = {c: i for i, c in enumerate(ewm_classes)}
            for p_idx, cls in enumerate(class_by_proto):
                if cls not in cls_to_idx:
                    continue
                ci = cls_to_idx[cls]
                d = np.linalg.norm(projected_rows - proto_mat[p_idx], axis=1) / radii_local[p_idx]
                ncm_per_class[:, ci] = np.minimum(ncm_per_class[:, ci], d)
            finite = np.isfinite(ncm_per_class)
            if finite.any():
                ncm_per_class[~finite] = ncm_per_class[finite].max()

            # Per-class normalisation so the two scores are comparable.
            ncm_med = float(np.median(ncm_per_class)) if n_cls else 1.0
            ewm_med = float(np.median(ewm_scores)) if n_cls else 1.0
            ncm_med = ncm_med if ncm_med > 1e-9 else 1.0
            ewm_med = ewm_med if ewm_med > 1e-9 else 1.0

            blended = (1.0 - alpha_mix) * (ncm_per_class / ncm_med) + alpha_mix * (ewm_scores / ewm_med)
            new_class_idx = np.argmin(blended, axis=1)

            # Map class string → representative prototype index.
            first_proto_for_class: dict[str, int] = {}
            for p_idx, cls in enumerate(class_by_proto):
                if cls in cls_to_idx and cls not in first_proto_for_class:
                    first_proto_for_class[cls] = p_idx
            new_class_indices = class_indices.copy()
            for i, c in enumerate(new_class_idx):
                cls = ewm_classes[int(c)]
                pid = first_proto_for_class.get(cls, int(class_indices[i]))
                new_class_indices[i] = pid
            class_indices = new_class_indices

    # --- CARAG: per-class anchor blend in NCDR ambiguity band (v2 mechanism).
    #     Re-routes admit→class-X to admit→class-Y inside the existing
    #     uncertainty zone using the per-class anchor distance δ_q(x).
    if (
        getattr(model.config, "enable_carag", False)
        and bool(getattr(model.config, "carag_use_anchor_blend", True))
        and getattr(model, "carag_state", None) is not None
        and bool(getattr(model.carag_state, "anchor_phi", {}))
        and float(getattr(model.carag_state, "beta", 0.0)) > 1e-6
        and getattr(model, "dvmad_core", None) is not None
    ):
        from bcmrnfst.projection.carag import per_class_anchor_signal
        cs = model.carag_state
        beta = float(cs.beta)
        anchor_phi = cs.anchor_phi
        ordered_classes = sorted(anchor_phi.keys())

        # Per-class NCM matrix.
        proto_class_idx = np.asarray(
            [ordered_classes.index(c) if c in ordered_classes else -1
             for c in class_by_proto], dtype=np.int64,
        )
        n_rows = projected_rows.shape[0]
        n_classes = len(ordered_classes)
        ncm_per_class = np.full((n_rows, n_classes), np.inf, dtype=np.float64)
        proto_mat = np.asarray(model.prototype_matrix, dtype=np.float64)
        radii_local = np.maximum(np.asarray(model.radii_vec, dtype=np.float64), 1e-9)
        for p_idx in range(proto_mat.shape[0]):
            cls_idx = int(proto_class_idx[p_idx])
            if cls_idx < 0:
                continue
            d = np.linalg.norm(projected_rows - proto_mat[p_idx], axis=1) / radii_local[p_idx]
            ncm_per_class[:, cls_idx] = np.minimum(ncm_per_class[:, cls_idx], d)
        finite_mask = np.isfinite(ncm_per_class)
        ncm_per_class[~finite_mask] = ncm_per_class[finite_mask].max() if finite_mask.any() else 1.0

        # Find top-2 classes per row + ratio.
        sorted_idx = np.argsort(ncm_per_class, axis=1)
        top1 = sorted_idx[:, 0]
        top2 = sorted_idx[:, 1] if n_classes > 1 else top1
        d1 = ncm_per_class[np.arange(n_rows), top1]
        d2 = ncm_per_class[np.arange(n_rows), top2]
        ratio = d1 / np.maximum(d2, 1e-12)
        lo = float(getattr(model.config, "carag_ambiguity_lo", 0.85))
        hi = float(getattr(model.config, "carag_ambiguity_hi", 1.0))
        in_band = (ratio >= lo) & (ratio <= hi)

        if in_band.any():
            # Anchor distances δ_q for in-band rows.
            in_band_idx = np.where(in_band)[0]
            phi_in = phi_eval[in_band_idx]
            delta = per_class_anchor_signal(model.dvmad_core, phi_in, anchor_phi)
            for ord_i, cls in enumerate(ordered_classes):
                if cls not in delta:
                    continue
                med = float(cs.delta_calib_median.get(cls, 0.0))
                std = max(float(cs.delta_calib_std.get(cls, 1.0)), 1e-9)
                z = (delta[cls] - med) / std
                # Anchor = class centroid in W. Small δ → close to class-q in
                # two-tailed space → favor class-q. Adding +β·z makes class-q's
                # score RISE for samples far from class-q (penalises wrong-class
                # candidates); class-q-typical samples get z≈0 and stay put.
                ncm_per_class[in_band_idx, ord_i] += beta * z

            # Re-pick argmin among top-1/top-2 only for in-band rows.
            new_top = np.where(
                ncm_per_class[in_band_idx, top1[in_band_idx]]
                <= ncm_per_class[in_band_idx, top2[in_band_idx]],
                top1[in_band_idx],
                top2[in_band_idx],
            )

            # Map chosen class index → representative prototype id, ONLY for
            # in-band rows. Non-in-band rows keep their existing NCDR-refined
            # class_indices unchanged.
            first_proto_for_class: dict[str, int] = {}
            for p_idx, cls in enumerate(class_by_proto):
                if cls in ordered_classes and cls not in first_proto_for_class:
                    first_proto_for_class[cls] = p_idx
            new_class_indices = class_indices.copy()
            for ii, row in enumerate(in_band_idx):
                cls = ordered_classes[int(new_top[ii])]
                proto_id = first_proto_for_class.get(cls, int(class_indices[row]))
                new_class_indices[row] = proto_id
            class_indices = new_class_indices

    # --- EWM gate-replacement (v3.5): compute EWM-min score for gate use.
    ewm_min_gate_score = None
    if (
        getattr(model.config, "enable_ewm", False)
        and (bool(getattr(model.config, "ewm_replace_gate", False))
             or bool(getattr(model.config, "ewm_and_combine_gate", False)))
        and getattr(model, "ewm_state", None) is not None
        and bool(getattr(model.ewm_state, "centroid_phi_per_class", {}))
    ):
        from bcmrnfst.projection.dvmad_ewm import ewm_score_per_class
        ewm_scores_for_gate, _ = ewm_score_per_class(model.ewm_state, phi_eval)
        if ewm_scores_for_gate.size:
            ewm_min_gate_score = ewm_scores_for_gate.min(axis=1)

    # --- NSD-Gate: re-expressed Sw-EWM gate via null-space + Fisher
    # weights (NO DVMADCore in the score path). Drop-in replacement for
    # ewm_min_gate_score; feeds the same gate machinery (tau / policy / amnesty).
    if (
        getattr(model.config, "enable_nsd_gate", False)
        and getattr(model, "nsd_gate_state", None) is not None
        and bool(getattr(getattr(model.nsd_gate_state, "ewm", None),
                          "centroid_phi_per_class", {}))
    ):
        from bcmrnfst.projection.dvmad_nsd_gate import nsd_gate_score
        nsd_scores = nsd_gate_score(model.nsd_gate_state, phi_eval)
        if nsd_scores.size:
            ewm_min_gate_score = np.asarray(nsd_scores, dtype=np.float64)

    # --- Optional novelty gate (disabled in GONS). Three policies:
    #   * 'or': reject if classifier OR gate says reject (boosts TUR, hurts CCR)
    #   * 'borderline': gate decides only when classifier score is within
    #                   [tau*(1-m), tau*(1+m)] — preserves CCR on confident admits
    #   * 'blend': new_score = (1-alpha)*score + alpha*norm(gate); single tau
    gate_force_reject = np.zeros(len(best_scores), dtype=bool)
    gate_scores: np.ndarray | None = None
    # NSD-Gate drives the gate via its OWN min-over-class score (no DVMADCore)
    # when enabled + a tau is calibrated; otherwise the EWM gate path is
    # unchanged (requires dvmad_core + dvmad_tau).
    _nsd_gate_active = (
        getattr(model.config, "enable_nsd_gate", False)
        and ewm_min_gate_score is not None
        and getattr(model, "dvmad_tau", None) is not None
    )
    if _nsd_gate_active or (
        getattr(model.config, "enable_dvmad_gate", False)
        and getattr(model, "dvmad_core", None) is not None
        and getattr(model, "dvmad_tau", None) is not None
    ):
        gate_input = phi_eval if model.config.dvmad_gate_use_phi else projected_rows
        if ewm_min_gate_score is not None:
            gate_scores = np.asarray(ewm_min_gate_score, dtype=np.float64)
        else:
            gate_scores = np.asarray(
                model.dvmad_core.predict(gate_input.astype(np.float64)),
                dtype=np.float64,
            )
        gate_tau = float(model.dvmad_tau)
        policy = str(getattr(model.config, "dvmad_gate_policy", "or"))
        # Class-support amnesty: skip gate when predicted class has < N support
        # points. Newly-admitted K-shot novels would otherwise be rejected.
        min_support = int(getattr(model.config, "dvmad_gate_min_class_support", 0))
        gate_eligible = np.ones(len(best_scores), dtype=bool)
        if min_support > 0:
            cls_sup: dict[str, int] = {}
            for bundle in model.support_store.values():
                cls_sup[bundle.class_label] = cls_sup.get(bundle.class_label, 0) + int(bundle.size)
            preds = [class_by_proto[int(class_indices[i])] for i in range(len(best_scores))]
            gate_eligible = np.asarray(
                [cls_sup.get(c, 0) >= min_support for c in preds],
                dtype=bool,
            )
        if policy == "or":
            gate_force_reject = (gate_scores > gate_tau) & gate_eligible
            # AND-combine with EWM: only reject when both v1 NN-distance AND
            # EWM-PCA score are above their respective Q-quantiles.
            if (
                bool(getattr(model.config, "ewm_and_combine_gate", False))
                and ewm_min_gate_score is not None
            ):
                ewm_tau = float(getattr(model, "_ewm_tau", 0.0) or 0.0)
                if ewm_tau <= 0:
                    ewm_tau = float(np.quantile(ewm_min_gate_score, 0.99))
                ewm_reject = ewm_min_gate_score > ewm_tau
                gate_force_reject = gate_force_reject & ewm_reject
        elif policy == "borderline":
            margin = float(getattr(model.config, "dvmad_gate_borderline_margin", 0.2))
            tau_arr = np.asarray(tau_by_row, dtype=np.float64)
            lo = tau_arr * (1.0 - margin)
            hi = tau_arr * (1.0 + margin)
            in_band = (best_scores >= lo) & (best_scores <= hi)
            gate_force_reject = in_band & (gate_scores > gate_tau) & gate_eligible
        elif policy == "blend":
            alpha = float(getattr(model.config, "dvmad_gate_blend_alpha", 0.5))
            tau_arr = np.asarray(tau_by_row, dtype=np.float64)
            tau_med = float(np.median(tau_arr)) if tau_arr.size else 1.0
            tau_med = tau_med if tau_med > 1e-9 else 1.0
            score_med = float(np.median(best_scores)) if best_scores.size else 1.0
            score_med = score_med if score_med > 1e-9 else 1.0
            gate_norm = (gate_scores / max(gate_tau, 1e-9)) * tau_med
            blended = (1.0 - alpha) * np.asarray(best_scores, dtype=np.float64) + alpha * gate_norm
            # Class-support amnesty: revert blend for ineligible samples.
            best_scores = np.where(gate_eligible, blended, np.asarray(best_scores, dtype=np.float64))

    results: list[PredictionResult] = []
    for row_index in range(len(best_scores)):
        reject_proto_idx = int(reject_indices[row_index])
        class_proto_idx = int(class_indices[row_index])
        proto_id = proto_ids[reject_proto_idx]
        tau_eff = float(tau_by_row[row_index])
        score = float(best_scores[row_index])
        if score > tau_eff or bool(gate_force_reject[row_index]) or bool(pct_force_reject[row_index]):
            phi_x = None
            if retain_unknown_features:
                phi_x = np.array(phi_eval[row_index], dtype=np.float64, copy=True)
            results.append(
                PredictionResult(
                    class_label="Unknown",
                    score=score,
                    proto_id=proto_id,
                    tau_effective=tau_eff,
                    projection_version=model.projection_version,
                    phi_x=phi_x,
                )
            )
            continue

        results.append(
            PredictionResult(
                class_label=class_by_proto[class_proto_idx],
                score=score,
                proto_id=proto_ids[class_proto_idx],
                tau_effective=tau_eff,
                projection_version=model.projection_version,
                phi_x=None,
            )
        )
    return results


def update_unknown_buffer(
    buffer: list[UnknownRecord],
    record: UnknownRecord,
    *,
    policy: UnknownBufferPolicy,
    B_u: int,
    unknown_seen_count: int,
    rng: Generator,
) -> list[UnknownRecord]:
    """Apply the configured bounded unknown-buffer update policy."""

    if B_u < 1:
        raise ValueError("B_u must be at least 1.")
    updated = list(buffer)
    if policy == "fifo":
        updated.append(record)
        return updated[-B_u:]

    if policy == "reservoir":
        if len(updated) < B_u:
            updated.append(record)
            return updated
        replace_index = int(rng.integers(0, unknown_seen_count))
        if replace_index < B_u:
            updated[replace_index] = record
        return updated

    raise ValueError(f"Unsupported unknown-buffer policy: {policy}")


def predict_and_update_unknown_buffer(
    model: BCMRNFSTModel,
    x_raw: FrameLike,
    *,
    t_now: float,
) -> PredictionResult:
    """Run prediction and retain only rejected unknowns in the bounded buffer."""

    result = predict_one(model, x_raw)
    if result.class_label != "Unknown" or result.phi_x is None:
        return result

    model.unknown_seen_count += 1
    record = UnknownRecord(
        x_raw=_normalize_single_sample(x_raw, label_column=model.config.label_column),
        phi_x=result.phi_x,
        score=result.score,
        t_now=float(t_now),
        projection_version=model.projection_version,
    )
    model.unknown_buffer = update_unknown_buffer(
        model.unknown_buffer,
        record,
        policy=model.config.unknown_buffer_policy,
        B_u=model.config.B_u,
        unknown_seen_count=model.unknown_seen_count,
        rng=model.rng,
    )
    update_sanity_metrics(model)
    return result


def _cap_rescored_unknowns(
    records: list[UnknownRecord],
    *,
    policy: UnknownBufferPolicy,
    B_u: int,
) -> list[UnknownRecord]:
    """Cap rescored survivors if the current budget shrank between refreshes."""

    if len(records) <= B_u:
        return records
    if policy == "fifo":
        return records[-B_u:]
    if policy == "reservoir":
        # Rescoring only removes entries; keep a deterministic prefix if the budget shrank.
        return records[:B_u]
    raise ValueError(f"Unsupported unknown-buffer policy: {policy}")


def rescore_unknown_buffer_after_refresh(model: BCMRNFSTModel) -> BCMRNFSTModel:
    """Re-score buffered unknowns after the deployed projection changes."""

    W_proj, _, _, _ = _require_inference_state(model)
    if not model.unknown_buffer:
        update_sanity_metrics(model)
        model.sanity_metrics["recovered_known_after_refresh"] = 0
        return model

    projected_rows = np.asarray(
        [record.phi_x @ W_proj for record in model.unknown_buffer],
        dtype=np.float64,
    )
    _, best_scores, _, tau_by_row, _ = _score_projected_rows(
        model,
        projected_rows,
        block_size=min(len(model.unknown_buffer), 1024),
    )
    rescored: list[UnknownRecord] = []
    recovered_count = 0

    for row_index, record in enumerate(model.unknown_buffer):
        score = float(best_scores[row_index])
        if score > float(tau_by_row[row_index]):
            rescored.append(
                UnknownRecord(
                    x_raw=record.x_raw,
                    phi_x=record.phi_x,
                    score=score,
                    t_now=record.t_now,
                    projection_version=model.projection_version,
                )
            )
        else:
            recovered_count += 1

    model.unknown_buffer = _cap_rescored_unknowns(
        rescored,
        policy=model.config.unknown_buffer_policy,
        B_u=model.config.B_u,
    )
    update_sanity_metrics(model)
    model.sanity_metrics["recovered_known_after_refresh"] = recovered_count
    return model
