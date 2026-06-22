"""Model state containers and offline fit scaffolding."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from bcmrnfst.calibration.thresholds import CalibrationState
from bcmrnfst.config.model import BCMRNFSTConfig
from bcmrnfst.maps.explicit import (
    ExplicitMap,
    ExplicitMapState,
    save_explicit_map_state,
)
from bcmrnfst.projection.core import MomentStats, ProjectionResult, ThresholdMetadata
from bcmrnfst.prototype.bank import (
    Prototype,
    serialize_projection_meta,
    serialize_threshold_meta,
)
from bcmrnfst.state.support import SupportBundle

if TYPE_CHECKING:
    from bcmrnfst.continual.inference import UnknownRecord


ACTIVE_UNIT_COUNT_LIMIT_ERROR_PREFIX = "Active unit count exceeds Q_stats_max:"


def is_active_unit_count_limit_error(value: object) -> bool:
    """Return whether one exception or message matches the active-unit budget guard."""

    return ACTIVE_UNIT_COUNT_LIMIT_ERROR_PREFIX in str(value)


@dataclass(slots=True)
class BCMRNFSTModel:
    """Offline GONS model state."""

    schema: Any
    preprocessor: Any
    phi: ExplicitMap
    phi_state: ExplicitMapState
    support_store: dict[str, SupportBundle]
    q_to_class: dict[str, str]
    class_registry: dict[str, tuple[str, ...]]
    proto_bank: dict[str, Prototype]
    calibration_state: CalibrationState
    pending_item_registry: dict[str, Any]
    pending_hot_cache: dict[str, Any]
    pending_batch_manifest: dict[str, Any]
    archive_manifest: dict[str, Any]
    next_canonical_q_id: int
    projection_version: int
    unknown_buffer: list[UnknownRecord]
    unknown_seen_count: int
    rng: Generator
    config: BCMRNFSTConfig
    tau_global: float
    W_proj: NDArray[np.float64] | None = None
    projection_meta: ProjectionResult | None = None
    threshold_meta: ThresholdMetadata | None = None
    prototype_matrix: NDArray[np.float64] | None = None
    radii_vec: NDArray[np.float64] | None = None
    sigma_inv_stack: NDArray[np.float64] | None = None
    proto_ids: tuple[str, ...] = ()
    full_prototype_matrix: NDArray[np.float64] | None = None
    full_radii_vec: NDArray[np.float64] | None = None
    evm_weibull_params: dict[str, tuple[float, float, float]] | None = None
    per_class_thresholds: dict[str, float] = field(default_factory=dict)
    shared_sigma_inv: NDArray[np.float64] | None = None
    global_mean_proj: NDArray[np.float64] | None = None
    knn_tree: Any = None
    knn_labels: NDArray[np.object_] | None = None
    reciprocal_matrix: NDArray[np.float64] | None = None
    reciprocal_radii: NDArray[np.float64] | None = None
    validation_exemplars: dict[str, NDArray[np.float64]] | None = None
    stats: MomentStats | None = None
    sanity_metrics: dict[str, float | int | str] = field(default_factory=dict)
    dvmad_core: Any = None
    dvmad_tau: float | None = None
    dvmad_tau_scale: float | None = None
    pct_rd_state: Any = None
    carag_state: Any = None
    ewm_state: Any = None
    _ewm_tau: float | None = None
    nsd_gate_state: Any = None  # NSD-Gate re-expressed Sw-EWM gate state.

    @property
    def active_class_count(self) -> int:
        """Number of active class labels in the bounded state."""

        return len(self.class_registry)

    @property
    def active_unit_count(self) -> int:
        """Number of active localized support bundles."""

        return len(self.support_store)


def derive_class_registry(support_store: dict[str, SupportBundle]) -> dict[str, tuple[str, ...]]:
    """Build a class-to-unit registry from the support store."""

    grouped: dict[str, list[str]] = {}
    for unit_id, bundle in support_store.items():
        grouped.setdefault(bundle.class_label, []).append(unit_id)
    return {class_label: tuple(sorted(unit_ids)) for class_label, unit_ids in grouped.items()}


def initialize_model_shell(
    *,
    schema: Any,
    preprocessor: Any,
    phi: ExplicitMap,
    phi_state: ExplicitMapState,
    support_store: dict[str, SupportBundle],
    config: BCMRNFSTConfig,
) -> BCMRNFSTModel:
    """Create the initial offline model shell before projection and calibration."""

    q_to_class = {unit_id: bundle.class_label for unit_id, bundle in support_store.items()}
    class_registry = derive_class_registry(support_store)
    return BCMRNFSTModel(
        schema=schema,
        preprocessor=preprocessor,
        phi=phi,
        phi_state=phi_state,
        support_store=support_store,
        q_to_class=q_to_class,
        class_registry=class_registry,
        proto_bank={},
        calibration_state=CalibrationState(),
        pending_item_registry={},
        pending_hot_cache={},
        pending_batch_manifest={},
        archive_manifest={},
        next_canonical_q_id=len(support_store),
        projection_version=0,
        unknown_buffer=[],
        unknown_seen_count=0,
        rng=default_rng(config.seed),
        config=config,
        tau_global=config.initial_tau_global,
    )


def enforce_active_state_budgets(model: BCMRNFSTModel) -> BCMRNFSTModel:
    """Validate the bounded active-state constraints used by the offline fit."""

    if model.active_class_count > model.config.C_active_max:
        raise ValueError(
            f"Active class count exceeds C_active_max: "
            f"{model.active_class_count} > {model.config.C_active_max}"
        )
    if model.active_unit_count > model.config.Q_stats_max:
        raise ValueError(
            f"{ACTIVE_UNIT_COUNT_LIMIT_ERROR_PREFIX} "
            f"{model.active_unit_count} > {model.config.Q_stats_max}"
        )
    for unit_id, bundle in model.support_store.items():
        if bundle.size > model.config.R_q_max:
            raise ValueError(
                f"Support bundle '{unit_id}' exceeds R_q_max: "
                f"{bundle.size} > {model.config.R_q_max}"
            )
    return model


def update_sanity_metrics(model: BCMRNFSTModel) -> BCMRNFSTModel:
    """Refresh lightweight sanity metrics for the current offline state."""

    projection_dim = int(model.W_proj.shape[1]) if model.W_proj is not None else 0
    stats_n_total = float(model.stats.n_total) if model.stats is not None else 0.0
    model.sanity_metrics = {
        "active_class_count": model.active_class_count,
        "active_unit_count": model.active_unit_count,
        "calibration_mode": model.calibration_state.mode or "insufficient",
        "hot_pending_count": len(model.pending_hot_cache),
        "n_total": stats_n_total,
        "pending_item_count": len(model.pending_item_registry),
        "projection_dim": projection_dim,
        "projection_mode": (
            model.projection_meta.mode
            if model.projection_meta is not None
            else "unset"
        ),
        "projection_version": model.projection_version,
        "provisional_prototype_count": len(
            [
                prototype
                for prototype in model.proto_bank.values()
                if prototype.status in {"provisional", "singleton_provisional"}
            ]
        ),
        "prototype_count": len(model.proto_bank),
        "tau_global": float(model.tau_global),
        "archived_pending_count": len(
            [
                record
                for record in model.pending_item_registry.values()
                if getattr(record, "payload_location", None) == "archived"
            ]
        ),
        "unknown_buffer_size": len(model.unknown_buffer),
        "unknown_seen_count": model.unknown_seen_count,
    }
    return model


def save_model_artifacts(model: BCMRNFSTModel, output_dir: Path) -> None:
    """Persist the fitted map state plus projection and threshold metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    save_explicit_map_state(model.phi, output_dir / "map_state.joblib")
    projection_payload = (
        {}
        if model.projection_meta is None
        else serialize_projection_meta(model.projection_meta)
    )
    threshold_payload = (
        {}
        if model.threshold_meta is None
        else serialize_threshold_meta(model.threshold_meta)
    )
    (output_dir / "projection_meta.json").write_text(
        json.dumps(projection_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "threshold_meta.json").write_text(
        json.dumps(threshold_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
