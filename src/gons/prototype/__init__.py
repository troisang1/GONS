"""Prototype bank construction and scoring."""

from gons.prototype.bank import (
    Prototype,
    build_stable_prototypes_from_support_store,
    calibrate_prototype_center,
    cap_prototype_bank,
    nearest_mahalanobis_prototype,
    nearest_normalized_prototype,
    nearest_prototype,
    prototype_scores,
    rebuild_dense_proto_arrays,
    reduce_prototype_scores,
    refresh_prototype_tau_overrides,
    serialize_projection_meta,
    serialize_threshold_meta,
)

__all__ = [
    "Prototype",
    "build_stable_prototypes_from_support_store",
    "calibrate_prototype_center",
    "cap_prototype_bank",
    "nearest_mahalanobis_prototype",
    "nearest_normalized_prototype",
    "nearest_prototype",
    "prototype_scores",
    "reduce_prototype_scores",
    "rebuild_dense_proto_arrays",
    "refresh_prototype_tau_overrides",
    "serialize_projection_meta",
    "serialize_threshold_meta",
]
