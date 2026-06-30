"""Pending-item intake, provisional service, and consolidated refresh."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gons.calibration.thresholds import (
    CalibrationState,
    fit_evm_weibull_per_class,
    fit_per_class_quantile_thresholds,
    fit_per_prototype_thresholds,
    recalibrate_global_threshold,
    score_projected_row_for_calibration,
    trim_reservoir,
)
from gons.continual.inference import rescore_unknown_buffer_after_refresh
from gons.localization import localize_mutual_rnn_units
from gons.preprocess import transform_rows
from gons.projection import (
    build_threshold_metadata,
    compute_soft_null_projection,
    rebuild_stats_from_support_store,
    resolve_refresh_scales,
    warm_start_projection,
)
from gons.prototype import (
    Prototype,
    build_stable_prototypes_from_support_store,
    cap_prototype_bank,
    rebuild_dense_proto_arrays,
    refresh_prototype_tau_overrides,
)
from gons.state import (
    build_support_bundles_from_localization,
    decay_and_prune_bundle,
    derive_class_registry,
    enforce_active_state_budgets,
    update_sanity_metrics,
)
from gons.state.support import SupportBundle

if TYPE_CHECKING:
    from gons.state.model import GONSModel

PendingStatus: TypeAlias = Literal[
    "pending",
    "provisional",
    "staged",
    "consolidated",
    "blocked",
    "archived_pending",
]
PendingPathway: TypeAlias = Literal["singleton", "tiny", "cluster", "manual"]


@dataclass(frozen=True, slots=True)
class PendingPayload:
    """Class-specific explicit-map payload retained for provisional or refresh work."""

    phi_item: NDArray[np.float64]
    y_item: NDArray[np.object_]
    created_at: float
    projection_version_seen: int
    class_label: str
    pathway: PendingPathway
    source_batch_id: str


@dataclass(slots=True)
class PendingItemRecord:
    """Registry entry for one unresolved class-specific pending item."""

    status: PendingStatus
    class_label: str
    source_batch_id: str
    created_at: float
    projection_version_seen: int
    payload_location: Literal["hot", "archived"]
    pathway: PendingPathway
    reason: str
    retries: int = 0


@dataclass(slots=True)
class ArchiveManifest:
    """In-model archive manifest for deferred payloads and archived classes."""

    audit_log: list[dict[str, object]] = field(default_factory=list)
    archived_class_index: dict[str, dict[str, SupportBundle]] = field(default_factory=dict)
    archived_item_index: dict[str, PendingPayload] = field(default_factory=dict)


def _archive_manifest(model: GONSModel) -> ArchiveManifest:
    """Normalize the model archive manifest into a typed container."""

    if isinstance(model.archive_manifest, ArchiveManifest):
        return model.archive_manifest
    manifest = ArchiveManifest()
    model.archive_manifest = cast(Any, manifest)
    return manifest


def _item_id(batch_id: str, class_label: str) -> str:
    return f"{batch_id}:{class_label}"


def _drop_class_prototypes(model: GONSModel, class_label: str) -> None:
    model.proto_bank = {
        proto_id: prototype
        for proto_id, prototype in model.proto_bank.items()
        if prototype.class_label != class_label
    }


def _drop_provisional_prototypes(model: GONSModel, class_label: str | None = None) -> None:
    provisional_statuses = {"provisional", "singleton_provisional"}
    model.proto_bank = {
        proto_id: prototype
        for proto_id, prototype in model.proto_bank.items()
        if prototype.status not in provisional_statuses
        or (class_label is not None and prototype.class_label != class_label)
    }


def _support_weight_by_class(model: GONSModel, class_label: str) -> float:
    weights = [
        float(np.sum(bundle.w_bundle))
        for bundle in model.support_store.values()
        if bundle.class_label == class_label
    ]
    return float(sum(weights))


def _candidate_projected_centroid(
    bundle: SupportBundle,
    W_proj: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.asarray(np.mean(bundle.phi_bundle @ W_proj, axis=0), dtype=np.float64)


def _candidate_projected_radius(
    bundle: SupportBundle,
    W_proj: NDArray[np.float64],
) -> float:
    points = np.asarray(bundle.phi_bundle @ W_proj, dtype=np.float64)
    center = np.asarray(np.mean(points, axis=0), dtype=np.float64)
    if points.shape[0] == 0:
        return 0.0
    return float(np.max(np.linalg.norm(points - center, axis=1)))


def _affine_residual(U: NDArray[np.float64], V: NDArray[np.float64]) -> float:
    """Return the mean residual of U to the affine span of V."""

    if V.shape[0] == 0:
        return float("inf")
    v_mean = np.mean(V, axis=0)
    centered_v = V - v_mean
    if centered_v.shape[0] <= 1:
        residual = U - v_mean
        return float(np.mean(np.linalg.norm(residual, axis=1)))

    _, singular_values, vt = np.linalg.svd(centered_v, full_matrices=False)
    rank = int(np.sum(singular_values > 1e-10))
    if rank == 0:
        residual = U - v_mean
        return float(np.mean(np.linalg.norm(residual, axis=1)))

    basis = vt[:rank].T
    centered_u = U - v_mean
    projection = centered_u @ basis @ basis.T
    residual = centered_u - projection
    return float(np.mean(np.linalg.norm(residual, axis=1)))


def _candidate_model_from_supports(
    model: GONSModel,
    supports: dict[str, SupportBundle],
) -> GONSModel:
    temp_model = copy.copy(model)
    temp_model.support_store = supports
    return temp_model


def _provisionalize_candidate_bank(
    candidate_proto_bank: dict[str, Prototype],
    *,
    status: str,
    tau_multiplier: float,
    tau_global: float,
) -> dict[str, Prototype]:
    updated: dict[str, Prototype] = {}
    for proto_id, prototype in candidate_proto_bank.items():
        updated[proto_id] = Prototype(
            prototype_id=prototype.prototype_id,
            b=prototype.b,
            rho=prototype.rho,
            class_label=prototype.class_label,
            member_q_ids=prototype.member_q_ids,
            support_weight=prototype.support_weight,
            status=status,
            tau_override=tau_multiplier * tau_global,
            sigma_inv=prototype.sigma_inv,
            use_mahalanobis=prototype.use_mahalanobis,
            tau_local=prototype.tau_local,
            b_full=prototype.b_full,
            rho_full=prototype.rho_full,
            diag_precision=prototype.diag_precision,
        )
    return updated


def _attach_candidate_prototypes(
    model: GONSModel,
    class_label: str,
    candidate_proto_bank: dict[str, Prototype],
) -> GONSModel:
    _drop_provisional_prototypes(model, class_label)
    model.proto_bank.update(candidate_proto_bank)
    model = cap_prototype_bank(model)
    model = rebuild_dense_proto_arrays(model)
    update_sanity_metrics(model)
    return model


def _rebuild_calibration_from_support_store(model: GONSModel) -> GONSModel:
    """Rebuild calibration reservoirs from the current active support store."""

    if model.W_proj is None:
        raise ValueError("Projection is required before rebuilding calibration state.")

    calibration_state = CalibrationState()
    stable_statuses = {"stable", "fragile_stable"}
    stable_proto_bank = {
        proto_id: prototype
        for proto_id, prototype in model.proto_bank.items()
        if prototype.status in stable_statuses
    }
    if not stable_proto_bank:
        model.calibration_state = calibration_state
        return model

    for bundle in model.support_store.values():
        Z_q = np.asarray(bundle.phi_bundle @ model.W_proj, dtype=np.float64)
        for z_i in Z_q:
            same_class = [
                proto_id
                for proto_id, prototype in stable_proto_bank.items()
                if prototype.class_label == bundle.class_label
            ]
            other_class = [
                proto_id
                for proto_id, prototype in stable_proto_bank.items()
                if prototype.class_label != bundle.class_label
            ]
            if same_class:
                reservoir = calibration_state.pos_score_reservoir_by_class.setdefault(
                    bundle.class_label,
                    [],
                )
                reservoir.append(score_projected_row_for_calibration(z_i, model, proto_ids=same_class))
            if (
                model.config.enable_impostor_calibration
                and other_class
                and not model.config.enable_knn_scoring
            ):
                calibration_state.imp_score_reservoir.append(
                    score_projected_row_for_calibration(z_i, model, proto_ids=other_class)
                )

    if model.unknown_buffer:
        for record in model.unknown_buffer:
            z_u = np.asarray(record.phi_x @ model.W_proj, dtype=np.float64)
            calibration_state.unknown_score_reservoir.append(
                score_projected_row_for_calibration(z_u, model)
            )

    for class_label, scores in list(calibration_state.pos_score_reservoir_by_class.items()):
        calibration_state.pos_score_reservoir_by_class[class_label] = trim_reservoir(
            scores,
            model.config.calibration_reservoir_size,
        )
    calibration_state.imp_score_reservoir = trim_reservoir(
        calibration_state.imp_score_reservoir,
        model.config.calibration_reservoir_size,
    )
    calibration_state.unknown_score_reservoir = trim_reservoir(
        calibration_state.unknown_score_reservoir,
        model.config.calibration_reservoir_size,
    )

    model.calibration_state = calibration_state
    return model


def cache_labeled_batch_by_class(
    model: GONSModel,
    labeled_batch_df: pd.DataFrame,
    *,
    batch_id: str,
    pathway: PendingPathway,
    t_now: float,
) -> GONSModel:
    """Split a labeled batch by class and cache each class-specific item."""

    X_new, _ = transform_rows(
        model.schema,
        model.preprocessor,
        labeled_batch_df,
        label_column=model.config.label_column,
    )
    Phi_new = model.phi.transform(X_new)
    y_new = labeled_batch_df[model.config.label_column].astype(str).to_numpy(dtype=object)

    item_ids: list[str] = []
    for class_label_obj in np.unique(y_new):
        class_label = str(class_label_obj)
        item_id = _item_id(batch_id, class_label)
        if item_id in model.pending_item_registry:
            continue
        indices = np.flatnonzero(y_new == class_label_obj).astype(np.int64)
        payload = PendingPayload(
            phi_item=np.asarray(Phi_new[indices], dtype=np.float64),
            y_item=np.asarray(y_new[indices], dtype=object),
            created_at=float(t_now),
            projection_version_seen=model.projection_version,
            class_label=class_label,
            pathway=pathway,
            source_batch_id=batch_id,
        )
        model.pending_hot_cache[item_id] = payload
        model.pending_item_registry[item_id] = PendingItemRecord(
            status="pending",
            class_label=class_label,
            source_batch_id=batch_id,
            created_at=float(t_now),
            projection_version_seen=model.projection_version,
            payload_location="hot",
            pathway=pathway,
            reason="new_labeled_item",
        )
        item_ids.append(item_id)

    model.pending_batch_manifest[batch_id] = tuple(item_ids)
    return enforce_pending_budget(model)


def enforce_pending_budget(model: GONSModel) -> GONSModel:
    """Keep only a bounded number of pending payloads hot in memory."""

    hot_item_ids = list(model.pending_hot_cache)
    if len(hot_item_ids) <= model.config.B_pending_max:
        update_sanity_metrics(model)
        return model

    manifest = _archive_manifest(model)
    model.pending_batch_manifest["refresh_requested"] = True
    while len(model.pending_hot_cache) > model.config.B_pending_max:
        candidates = [
            (
                0 if model.pending_item_registry[item_id].status == "blocked" else 1,
                model.pending_item_registry[item_id].created_at,
                item_id,
            )
            for item_id in model.pending_hot_cache
        ]
        _, _, item_id = min(candidates)
        payload = model.pending_hot_cache.pop(item_id)
        manifest.archived_item_index[item_id] = payload
        record = model.pending_item_registry[item_id]
        record.payload_location = "archived"
        record.status = "archived_pending"
        record.reason = "archived due to hot pending budget"

    update_sanity_metrics(model)
    return model


def load_item_payload(model: GONSModel, item_id: str) -> PendingPayload:
    """Load one pending payload from the hot cache or archive manifest."""

    record = model.pending_item_registry[item_id]
    if record.payload_location == "hot":
        return cast(PendingPayload, model.pending_hot_cache[item_id])
    manifest = _archive_manifest(model)
    return manifest.archived_item_index[item_id]


def choose_archive_class(
    model: GONSModel,
    *,
    protected_classes: set[str] | None = None,
) -> str | None:
    """Choose an active class to archive under the current class budget."""

    protected = protected_classes or set()
    unresolved_classes = {
        record.class_label
        for record in model.pending_item_registry.values()
        if record.status != "consolidated"
    }
    candidates = [
        class_label
        for class_label in model.class_registry
        if class_label not in protected and class_label not in unresolved_classes
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda class_label: (_support_weight_by_class(model, class_label), class_label),
    )


def archive_class_state(model: GONSModel, class_label: str) -> GONSModel:
    """Archive one active class out of the deployed state."""

    if class_label not in model.class_registry:
        return model
    manifest = _archive_manifest(model)
    q_ids = [
        unit_id
        for unit_id, bundle in model.support_store.items()
        if bundle.class_label == class_label
    ]
    manifest.archived_class_index[class_label] = {
        unit_id: model.support_store[unit_id]
        for unit_id in q_ids
    }
    for unit_id in q_ids:
        model.support_store.pop(unit_id, None)
        model.q_to_class.pop(unit_id, None)
    _drop_class_prototypes(model, class_label)
    model.calibration_state.pos_score_reservoir_by_class.pop(class_label, None)
    model.class_registry = derive_class_registry(model.support_store)
    model = rebuild_dense_proto_arrays(model)
    return model


def load_archived_class_state(model: GONSModel, class_label: str) -> dict[str, SupportBundle]:
    """Return archived support bundles for one class, if present."""

    manifest = _archive_manifest(model)
    return copy.deepcopy(manifest.archived_class_index.get(class_label, {}))


def margin_gate_passes(
    candidate_proto_bank: dict[str, Prototype],
    deployed_proto_bank: dict[str, Prototype],
    *,
    tau_sep: float,
    eta_margin: float,
) -> bool:
    """Reject candidates that land too close to deployed cross-class prototypes."""

    for candidate in candidate_proto_bank.values():
        for deployed in deployed_proto_bank.values():
            if deployed.class_label == candidate.class_label:
                continue
            distance = float(np.linalg.norm(candidate.b - deployed.b))
            if distance <= tau_sep * (candidate.rho + deployed.rho) + eta_margin:
                return False
    return True


def local_affine_conflict_screen(
    candidate_supports: dict[str, SupportBundle],
    model: GONSModel,
    *,
    shortlist_k: int,
    eps_aff_local: float,
) -> bool:
    """Heuristic local reject screen using explicit-space affine residuals."""

    if model.W_proj is None:
        raise ValueError("Projection is required for local affine screening.")
    active_supports = list(model.support_store.values())
    if not active_supports:
        return True

    active_centroids = [
        _candidate_projected_centroid(bundle, model.W_proj)
        for bundle in active_supports
    ]
    active_radii = [
        _candidate_projected_radius(bundle, model.W_proj)
        for bundle in active_supports
    ]
    for candidate in candidate_supports.values():
        candidate_centroid = _candidate_projected_centroid(candidate, model.W_proj)
        candidate_radius = _candidate_projected_radius(candidate, model.W_proj)
        distances = [
            (float(np.linalg.norm(candidate_centroid - active_centroids[index])), index)
            for index in range(len(active_supports))
            if active_supports[index].class_label != candidate.class_label
        ]
        for _, index in sorted(distances)[:shortlist_k]:
            active = active_supports[index]
            centroid_distance = float(np.linalg.norm(candidate_centroid - active_centroids[index]))
            if centroid_distance > candidate_radius + active_radii[index] + eps_aff_local:
                continue
            residual = _affine_residual(candidate.phi_bundle, active.phi_bundle)
            if residual <= eps_aff_local:
                return False
    return True


def provisional_admit_small_item(
    model: GONSModel,
    item_id: str,
    *,
    t_now: float,
) -> GONSModel:
    """Attempt provisional service for a singleton or tiny pending item."""

    payload = load_item_payload(model, item_id)
    n_b = int(payload.phi_item.shape[0])
    if n_b > model.config.k_tiny_max:
        raise ValueError("Small-item provisional route received an oversized payload.")

    temp_supports = {
        f"item:{item_id}:tiny": SupportBundle(
            unit_id=f"item:{item_id}:tiny",
            phi_bundle=payload.phi_item,
            w_bundle=np.ones(n_b, dtype=np.float64),
            t_bundle=np.full(n_b, float(t_now), dtype=np.float64),
            class_label=payload.class_label,
            status="temp_tiny",
        )
    }
    scales = resolve_refresh_scales(model)
    temp_model = _candidate_model_from_supports(model, temp_supports)
    candidate_proto_bank = build_stable_prototypes_from_support_store(temp_model, scales)
    status = "singleton_provisional" if n_b == 1 else "provisional"
    tau_mult = model.config.singleton_tau_mult if n_b == 1 else model.config.tiny_tau_mult
    candidate_proto_bank = _provisionalize_candidate_bank(
        candidate_proto_bank,
        status=status,
        tau_multiplier=tau_mult,
        tau_global=model.tau_global,
    )

    sep_ok = margin_gate_passes(
        candidate_proto_bank,
        model.proto_bank,
        tau_sep=max(model.config.c_sep_singleton * model.tau_global, model.tau_global),
        eta_margin=scales.eta_margin,
    )
    aff_ok = (
        local_affine_conflict_screen(
            temp_supports,
            model,
            shortlist_k=model.config.affine_shortlist_k,
            eps_aff_local=scales.eps_aff_local,
        )
        if model.config.enable_local_affine_screen
        else True
    )
    record = model.pending_item_registry[item_id]
    if sep_ok and aff_ok:
        model = _attach_candidate_prototypes(model, payload.class_label, candidate_proto_bank)
        record.status = "provisional"
        record.reason = "small-item provisional route"
    else:
        _drop_provisional_prototypes(model, payload.class_label)
        model = rebuild_dense_proto_arrays(model)
        record.status = "pending"
        record.reason = "small-item failed local screen"
    update_sanity_metrics(model)
    return model


def provisional_admit_cluster_item(
    model: GONSModel,
    item_id: str,
    *,
    t_now: float,
) -> GONSModel:
    """Attempt provisional service for a clustered pending item."""

    payload = load_item_payload(model, item_id)
    localization = localize_mutual_rnn_units(
        payload.phi_item,
        payload.y_item,
        r=model.config.r,
        min_microclass_size=model.config.min_microclass_size,
        namespace=f"item:{item_id}",
        metric=model.config.localization_metric,
    )
    candidate_supports = build_support_bundles_from_localization(
        payload.phi_item,
        localization,
        t_now=float(t_now),
        max_bundle_size=model.config.R_q_max,
        namespace=f"item:{item_id}",
        mode="inference",
        trim_strategy=model.config.support_trim_strategy,
        residual_handling=model.config.residual_handling,
        rng=model.rng,
    )
    scales = resolve_refresh_scales(model)
    temp_model = _candidate_model_from_supports(model, candidate_supports)
    candidate_proto_bank = build_stable_prototypes_from_support_store(temp_model, scales)
    candidate_proto_bank = _provisionalize_candidate_bank(
        candidate_proto_bank,
        status="provisional",
        tau_multiplier=model.config.prov_tau_mult,
        tau_global=model.tau_global,
    )

    sep_ok = margin_gate_passes(
        candidate_proto_bank,
        model.proto_bank,
        tau_sep=scales.tau_sep,
        eta_margin=scales.eta_margin,
    )
    aff_ok = (
        local_affine_conflict_screen(
            candidate_supports,
            model,
            shortlist_k=model.config.affine_shortlist_k,
            eps_aff_local=scales.eps_aff_local,
        )
        if model.config.enable_local_affine_screen
        else True
    )
    record = model.pending_item_registry[item_id]
    if sep_ok and aff_ok:
        model = _attach_candidate_prototypes(model, payload.class_label, candidate_proto_bank)
        record.status = "provisional"
        record.reason = "cluster item passed local screens"
    else:
        _drop_provisional_prototypes(model, payload.class_label)
        model = rebuild_dense_proto_arrays(model)
        record.status = "pending"
        record.reason = "cluster item failed local screen"
    update_sanity_metrics(model)
    return model


def get_retained_class_support_for_refresh(
    model: GONSModel,
    class_label: str,
    *,
    t_now: float,
) -> dict[str, SupportBundle]:
    """Load active or archived support bundles and apply decay/pruning."""

    if class_label in model.class_registry:
        old_supports = {
            unit_id: bundle
            for unit_id, bundle in model.support_store.items()
            if bundle.class_label == class_label
        }
    elif not model.config.enable_archived_class_rehydration:
        old_supports = {}
    else:
        old_supports = load_archived_class_state(model, class_label)

    decayed: dict[str, SupportBundle] = {}
    for unit_id, bundle in old_supports.items():
        decayed_bundle = decay_and_prune_bundle(
            bundle,
            t_now=float(t_now),
            lambda_decay=model.config.lambda_decay,
            w_drop=model.config.w_drop,
            max_bundle_size=model.config.R_q_max,
            trim_strategy=model.config.support_trim_strategy,
            rng=model.rng,
        )
        if decayed_bundle is not None:
            decayed[unit_id] = decayed_bundle
    return decayed


def build_class_refresh_candidate(
    model: GONSModel,
    class_label: str,
    refresh_item_ids: tuple[str, ...],
    *,
    t_now: float,
) -> dict[str, SupportBundle]:
    """Build the per-class candidate support state used in consolidated refresh."""

    old_supports = get_retained_class_support_for_refresh(model, class_label, t_now=t_now)
    phi_blocks = [bundle.phi_bundle for bundle in old_supports.values()]
    for item_id in refresh_item_ids:
        payload = load_item_payload(model, item_id)
        if payload.class_label == class_label:
            phi_blocks.append(payload.phi_item)

    if not phi_blocks:
        return {}
    phi_joint = np.vstack(phi_blocks)
    y_joint = np.asarray([class_label] * phi_joint.shape[0], dtype=object)

    # PA-iNN: augment under-represented classes during refresh so that novel
    # classes admitted with very few support samples get scatter stabilization.
    phi_for_loc = phi_joint
    y_for_loc = y_joint
    if model.config.enable_pa_inn and phi_joint.shape[0] < model.config.pa_inn_min_samples:
        from gons.prototype.augmentation import augment_support_for_scatter

        phi_for_loc, y_for_loc = augment_support_for_scatter(
            phi_joint,
            y_joint,
            min_samples_per_class=model.config.pa_inn_min_samples,
            n_anchors=model.config.pa_inn_n_anchors,
            regularization=model.config.pa_inn_regularization,
            seed=model.config.seed,
        )

    localization = localize_mutual_rnn_units(
        phi_for_loc,
        y_for_loc,
        r=model.config.r,
        min_microclass_size=model.config.min_microclass_size,
        namespace=f"refresh:{class_label}",
        metric=model.config.localization_metric,
    )
    # If PA-iNN augmented the features, truncate the localization result to
    # only cover original (real) rows so the support store stays clean.
    n_real = phi_joint.shape[0]
    if localization.unit_ids.shape[0] > n_real:
        from gons.localization.mutual_rnn import LocalizationResult

        localization = LocalizationResult(
            unit_ids=localization.unit_ids[:n_real],
            unit_to_class=localization.unit_to_class,
            residual_components=localization.residual_components,
            effective_r_by_class=localization.effective_r_by_class,
        )
    return build_support_bundles_from_localization(
        phi_joint,
        localization,
        t_now=float(t_now),
        max_bundle_size=model.config.R_q_max,
        namespace=f"refresh:{class_label}",
        mode="refresh",
        trim_strategy=model.config.support_trim_strategy,
        residual_handling=model.config.residual_handling,
        rng=model.rng,
    )


def _shallow_model_copy(model: GONSModel) -> GONSModel:
    """Memory-efficient model copy for sandbox construction.

    Only deep-copies mutable containers that will be modified during
    sandbox construction.  Immutable objects (SupportBundle, config) and
    heavy arrays (prototype_matrix, knn_tree, phi_bundles) are shared
    with the original until overwritten by the rebuild step.
    """

    sandbox = copy.copy(model)
    # Mutable dicts that sandbox construction pops from / inserts into:
    sandbox.support_store = dict(model.support_store)
    sandbox.q_to_class = dict(model.q_to_class)
    sandbox.class_registry = dict(model.class_registry) if model.class_registry else {}
    sandbox.proto_bank = dict(model.proto_bank)
    sandbox.pending_item_registry = dict(model.pending_item_registry)
    sandbox.pending_hot_cache = dict(model.pending_hot_cache)
    sandbox.pending_batch_manifest = dict(model.pending_batch_manifest)
    sandbox.per_class_thresholds = dict(model.per_class_thresholds)
    sandbox.sanity_metrics = dict(model.sanity_metrics)
    # archive_class_state may mutate calibration_state reservoirs:
    sandbox.calibration_state = CalibrationState(
        pos_score_reservoir_by_class={
            k: list(v) for k, v in model.calibration_state.pos_score_reservoir_by_class.items()
        },
        imp_score_reservoir=list(model.calibration_state.imp_score_reservoir),
        unknown_score_reservoir=list(model.calibration_state.unknown_score_reservoir),
        mode=model.calibration_state.mode,
    )
    # archive_manifest may be mutated by archive_class_state:
    if isinstance(model.archive_manifest, ArchiveManifest):
        sandbox.archive_manifest = copy.deepcopy(model.archive_manifest)
    else:
        sandbox.archive_manifest = {}
    return sandbox


def build_refresh_sandbox(
    model: GONSModel,
    candidate_class_state: dict[str, dict[str, SupportBundle]],
    accepted_classes: set[str],
    *,
    t_now: float,
    protected_classes: set[str],
) -> GONSModel:
    """Build the deployed sandbox geometry for one candidate accepted set."""

    sandbox = _shallow_model_copy(model)
    _drop_provisional_prototypes(sandbox)
    sandbox.pending_item_registry = {}
    sandbox.pending_hot_cache = {}
    sandbox.pending_batch_manifest = {}

    for class_label in accepted_classes:
        remove_ids = [
            unit_id
            for unit_id, bundle in sandbox.support_store.items()
            if bundle.class_label == class_label
        ]
        for unit_id in remove_ids:
            sandbox.support_store.pop(unit_id, None)
            sandbox.q_to_class.pop(unit_id, None)
        for unit_id, bundle in candidate_class_state.get(class_label, {}).items():
            sandbox.support_store[unit_id] = bundle
            sandbox.q_to_class[unit_id] = bundle.class_label

    sandbox.class_registry = derive_class_registry(sandbox.support_store)
    while len(sandbox.class_registry) > sandbox.config.C_active_max:
        archive_candidate = choose_archive_class(
            sandbox,
            protected_classes=protected_classes | accepted_classes,
        )
        if archive_candidate is None:
            raise ValueError("No archivable class available to satisfy C_active_max.")
        sandbox = archive_class_state(sandbox, archive_candidate)

    sandbox.class_registry = derive_class_registry(sandbox.support_store)
    sandbox = enforce_active_state_budgets(sandbox)
    sandbox.stats = rebuild_stats_from_support_store(sandbox)
    projection_result = compute_soft_null_projection(sandbox)
    if sandbox.config.alpha_warmstart > 0.0:
        warmed_projection = warm_start_projection(
            model.W_proj,
            projection_result.W_proj,
            sandbox.config.alpha_warmstart,
        )
        projection_result = replace(
            projection_result,
            W_proj=warmed_projection,
            mode=f"{projection_result.mode}+warm_start",
        )
    sandbox.W_proj = projection_result.W_proj
    sandbox.projection_meta = projection_result
    scales = resolve_refresh_scales(
        sandbox,
        optional_small_spectrum=projection_result.small_eigvals,
        optional_range_spectrum=projection_result.range_eigvals,
    )
    sandbox.proto_bank = build_stable_prototypes_from_support_store(sandbox, scales)
    sandbox = rebuild_dense_proto_arrays(sandbox)
    update_sanity_metrics(sandbox)
    return sandbox


def global_affine_conflict_review_in_sandbox(
    candidate_class_state: dict[str, dict[str, SupportBundle]],
    accepted_classes: set[str],
    sandbox: GONSModel,
    *,
    eps_aff_global: float,
) -> set[str]:
    """Return accepted classes that still conflict in the sandbox geometry."""

    blocked: set[str] = set()
    if sandbox.W_proj is None:
        raise ValueError("Sandbox projection is required for global conflict review.")
    accepted_supports = {
        class_label: list(candidate_class_state.get(class_label, {}).values())
        for class_label in accepted_classes
    }
    active_by_class: dict[str, list[SupportBundle]] = {}
    for bundle in sandbox.support_store.values():
        active_by_class.setdefault(bundle.class_label, []).append(bundle)

    for class_label in accepted_classes:
        for candidate_bundle in accepted_supports.get(class_label, []):
            for other_class, bundles in active_by_class.items():
                if other_class == class_label:
                    continue
                for other_bundle in bundles:
                    candidate_centroid = _candidate_projected_centroid(
                        candidate_bundle,
                        sandbox.W_proj,
                    )
                    candidate_radius = _candidate_projected_radius(
                        candidate_bundle,
                        sandbox.W_proj,
                    )
                    other_centroid = _candidate_projected_centroid(other_bundle, sandbox.W_proj)
                    other_radius = _candidate_projected_radius(other_bundle, sandbox.W_proj)
                    centroid_distance = float(np.linalg.norm(candidate_centroid - other_centroid))
                    if centroid_distance > candidate_radius + other_radius + eps_aff_global:
                        continue
                    residual = _affine_residual(
                        candidate_bundle.phi_bundle,
                        other_bundle.phi_bundle,
                    )
                    if residual <= eps_aff_global:
                        blocked.add(class_label)
                        if other_class in accepted_classes:
                            blocked.add(other_class)
                        break
                if class_label in blocked:
                    break
            if class_label in blocked:
                break
    return blocked


def _cleanup_refresh_item(model: GONSModel, item_id: str) -> None:
    model.pending_hot_cache.pop(item_id, None)
    manifest = _archive_manifest(model)
    manifest.archived_item_index.pop(item_id, None)


def _exemplar_accuracy(
    model: GONSModel,
    exemplars: dict[str, NDArray[np.float64]],
) -> float:
    """Return label accuracy for explicit-space exemplars under the current geometry."""

    if (
        not exemplars
        or model.W_proj is None
        or model.prototype_matrix is None
        or not model.proto_ids
    ):
        return 0.0

    prototype_labels = np.asarray(
        [model.proto_bank[proto_id].class_label for proto_id in model.proto_ids],
        dtype=object,
    )
    safe_radii = (
        np.maximum(model.radii_vec, 1e-12)
        if model.radii_vec is not None
        else np.ones(model.prototype_matrix.shape[0], dtype=np.float64)
    )

    correct = 0
    total = 0
    for class_label, phi_rows in exemplars.items():
        if phi_rows.shape[0] == 0:
            continue
        projected = np.asarray(phi_rows @ model.W_proj, dtype=np.float64)
        if model.config.use_mahalanobis and model.sigma_inv_stack is not None:
            diff = model.prototype_matrix[None, :, :] - projected[:, None, :]
            quadratic = np.einsum("bnd,ndk,bnk->bn", diff, model.sigma_inv_stack, diff)
            scores = np.sqrt(np.maximum(quadratic, 0.0))
        else:
            distances = np.linalg.norm(
                model.prototype_matrix[None, :, :] - projected[:, None, :],
                axis=2,
            )
            scores = distances / safe_radii[None, :]
        best_indices = np.argmin(scores, axis=1)
        predicted = prototype_labels[best_indices]
        correct += int(np.sum(predicted == class_label))
        total += int(projected.shape[0])
    return float(correct) / max(total, 1)


def _restore_previous_projection_geometry(
    model: GONSModel,
    sandbox: GONSModel,
) -> GONSModel:
    """Keep the committed support store but rebuild geometry with the previous projection."""

    if model.W_proj is None:
        return sandbox
    sandbox.W_proj = np.asarray(model.W_proj, dtype=np.float64)
    sandbox.projection_meta = model.projection_meta
    scales = resolve_refresh_scales(
        sandbox,
        optional_small_spectrum=(
            model.projection_meta.small_eigvals if model.projection_meta is not None else None
        ),
        optional_range_spectrum=(
            model.projection_meta.range_eigvals if model.projection_meta is not None else None
        ),
    )
    sandbox.proto_bank = build_stable_prototypes_from_support_store(sandbox, scales)
    sandbox = rebuild_dense_proto_arrays(sandbox)
    update_sanity_metrics(sandbox)
    return sandbox


def _apply_refresh_validation(
    model: GONSModel,
    sandbox: GONSModel,
) -> GONSModel:
    """Reject refreshed geometry when it underperforms the retained exemplar store."""

    exemplars = model.validation_exemplars
    if not model.config.enable_refresh_validation or not exemplars:
        return sandbox

    pre_acc = _exemplar_accuracy(model, exemplars)
    post_acc = _exemplar_accuracy(sandbox, exemplars)
    allowed_drop = float(model.config.refresh_validation_min_acc)
    if post_acc >= pre_acc - allowed_drop:
        return sandbox
    return _restore_previous_projection_geometry(model, sandbox)


def consolidated_refresh(
    model: GONSModel,
    refresh_item_ids: list[str] | tuple[str, ...],
    *,
    t_now: float,
    force_accept_classes: set[str] | None = None,
) -> GONSModel:
    """Run the budget-aware fixed-point consolidated refresh.

    Parameters
    ----------
    force_accept_classes:
        Optional set of class labels that bypass the affine conflict review
        and are always accepted.  Used by the FSCIL runner to guarantee that
        novel-class support is committed into the model state.
    """

    item_ids = tuple(refresh_item_ids)
    if not item_ids:
        return model

    _force_accept = force_accept_classes or set()

    touched_classes = {
        model.pending_item_registry[item_id].class_label
        for item_id in item_ids
    }
    candidate_class_state = {
        class_label: build_class_refresh_candidate(
            model,
            class_label,
            item_ids,
            t_now=float(t_now),
        )
        for class_label in touched_classes
    }
    protected_classes = {
        record.class_label
        for pending_id, record in model.pending_item_registry.items()
        if pending_id not in item_ids and record.status != "consolidated"
    }

    accepted_classes = set(touched_classes)
    if model.config.refresh_mode == "single_pass":
        sandbox = build_refresh_sandbox(
            model,
            candidate_class_state,
            accepted_classes,
            t_now=float(t_now),
            protected_classes=protected_classes,
        )
        scales_sandbox = resolve_refresh_scales(sandbox)
        blocked_classes = global_affine_conflict_review_in_sandbox(
            candidate_class_state,
            accepted_classes,
            sandbox,
            eps_aff_global=scales_sandbox.eps_aff_global,
        )
        accepted_classes -= (blocked_classes - _force_accept)
    else:
        previous_accept: set[str] | None = None
        while accepted_classes != previous_accept:
            previous_accept = set(accepted_classes)
            sandbox = build_refresh_sandbox(
                model,
                candidate_class_state,
                accepted_classes,
                t_now=float(t_now),
                protected_classes=protected_classes,
            )
            scales_sandbox = resolve_refresh_scales(sandbox)
            blocked_classes = global_affine_conflict_review_in_sandbox(
                candidate_class_state,
                accepted_classes,
                sandbox,
                eps_aff_global=scales_sandbox.eps_aff_global,
            )
            accepted_classes -= (blocked_classes - _force_accept)
            if not accepted_classes:
                del sandbox
                break

    final_sandbox = build_refresh_sandbox(
        model,
        candidate_class_state,
        accepted_classes,
        t_now=float(t_now),
        protected_classes=protected_classes,
    )
    final_sandbox = _apply_refresh_validation(model, final_sandbox)
    model.support_store = final_sandbox.support_store
    model.q_to_class = final_sandbox.q_to_class
    model.class_registry = final_sandbox.class_registry
    model.stats = final_sandbox.stats
    model.W_proj = final_sandbox.W_proj
    model.projection_meta = final_sandbox.projection_meta
    model.projection_version += 1
    model.proto_bank = final_sandbox.proto_bank
    model.archive_manifest = final_sandbox.archive_manifest
    model = rebuild_dense_proto_arrays(model)
    model = _rebuild_calibration_from_support_store(model)
    model.tau_global = recalibrate_global_threshold(model)
    if model.config.enable_per_class_quantile_threshold:
        model.per_class_thresholds = fit_per_class_quantile_thresholds(model)
    else:
        model.per_class_thresholds = {}
    model = refresh_prototype_tau_overrides(model)
    if model.config.enable_evm_calibration:
        model.evm_weibull_params = fit_evm_weibull_per_class(model)
    if model.config.enable_per_proto_threshold:
        for proto_id, tau_local in fit_per_prototype_thresholds(model).items():
            if proto_id in model.proto_bank:
                model.proto_bank[proto_id] = replace(model.proto_bank[proto_id], tau_local=tau_local)
    current_scales = resolve_refresh_scales(
        model,
        optional_small_spectrum=(
            model.projection_meta.small_eigvals if model.projection_meta else None
        ),
        optional_range_spectrum=(
            model.projection_meta.range_eigvals if model.projection_meta else None
        ),
    )
    model.threshold_meta = build_threshold_metadata(
        current_scales,
        tau_global=model.tau_global,
        calibration_mode=model.calibration_state.mode or "insufficient",
    )
    if model.config.enable_unknown_buffer_rescoring:
        model = rescore_unknown_buffer_after_refresh(model)

    blocked_items = [
        item_id
        for item_id in item_ids
        if model.pending_item_registry[item_id].class_label not in accepted_classes
    ]
    for item_id in blocked_items:
        record = model.pending_item_registry[item_id]
        if record.pathway in {"singleton", "tiny"}:
            model = provisional_admit_small_item(model, item_id, t_now=float(t_now))
        else:
            model = provisional_admit_cluster_item(model, item_id, t_now=float(t_now))

    manifest = _archive_manifest(model)
    for item_id in item_ids:
        record = model.pending_item_registry[item_id]
        class_label = record.class_label
        if class_label in accepted_classes:
            record.status = "consolidated"
            record.reason = "joint committed refresh"
            _cleanup_refresh_item(model, item_id)
        else:
            if record.status != "provisional":
                record.status = "blocked"
                record.reason = "blocked in budget-aware refresh"
            record.retries += 1
        manifest.audit_log.append(
            {
                "item_id": item_id,
                "class_label": class_label,
                "status": record.status,
                "reason": record.reason,
                "projection_version": model.projection_version,
                "t_now": float(t_now),
            }
        )
    update_sanity_metrics(model)
    return model
