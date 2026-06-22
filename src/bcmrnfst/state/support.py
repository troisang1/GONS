"""Support-bundle representations and trimming hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from bcmrnfst.localization.mutual_rnn import FeatureArray, LocalizationResult

SupportStatus: TypeAlias = Literal["temp_regular", "temp_fragile", "temp_tiny"]
TrimStrategy: TypeAlias = Literal["latest", "head", "uniform", "leverage"]
ResidualHandling: TypeAlias = Literal["preserve", "drop", "force_merge"]


@dataclass(frozen=True, slots=True)
class SupportBundle:
    """Bounded explicit-space support bundle for a localized unit."""

    unit_id: str
    phi_bundle: FeatureArray
    w_bundle: NDArray[np.float64]
    t_bundle: NDArray[np.float64]
    class_label: str
    status: SupportStatus

    @property
    def size(self) -> int:
        """Number of retained support rows."""

        return int(self.phi_bundle.shape[0])


def trim_support_bundle(
    phi_bundle: FeatureArray,
    w_bundle: NDArray[np.float64],
    t_bundle: NDArray[np.float64],
    *,
    max_size: int,
    strategy: TrimStrategy = "latest",
    rng: Generator | None = None,
) -> tuple[FeatureArray, NDArray[np.float64], NDArray[np.float64]]:
    """Trim a support bundle when it exceeds the configured budget."""

    current_size = int(phi_bundle.shape[0])
    if current_size <= max_size:
        return phi_bundle, w_bundle, t_bundle
    if strategy == "latest":
        keep_indices = np.arange(current_size - max_size, current_size, dtype=np.int64)
    elif strategy == "head":
        keep_indices = np.arange(max_size, dtype=np.int64)
    elif strategy == "uniform":
        keep_indices = np.linspace(0, current_size - 1, num=max_size, dtype=np.int64)
    elif strategy == "leverage":
        sampling_rng = rng or np.random.default_rng(0)
        U, singular_values, _ = np.linalg.svd(np.asarray(phi_bundle, dtype=np.float64), full_matrices=False)
        rank = int(np.sum(singular_values > 1e-10))
        if rank == 0:
            keep_indices = np.sort(
                sampling_rng.choice(current_size, size=max_size, replace=False).astype(np.int64)
            )
        else:
            leverage_scores = np.sum(U[:, :rank] ** 2, axis=1)
            probabilities = leverage_scores / np.maximum(np.sum(leverage_scores), 1e-12)
            keep_indices = np.sort(
                sampling_rng.choice(
                    current_size,
                    size=max_size,
                    replace=False,
                    p=probabilities,
                ).astype(np.int64)
            )
    else:
        raise ValueError(f"Unsupported trim strategy: {strategy}")
    return phi_bundle[keep_indices], w_bundle[keep_indices], t_bundle[keep_indices]


def decay_weight(delta_t: NDArray[np.float64], lambda_decay: float) -> NDArray[np.float64]:
    """Apply exponential support decay to elapsed times."""

    if lambda_decay <= 0.0:
        return np.ones_like(delta_t, dtype=np.float64)
    return np.asarray(np.exp(-lambda_decay * delta_t), dtype=np.float64)


def decay_and_prune_bundle(
    bundle: SupportBundle,
    *,
    t_now: float,
    lambda_decay: float,
    w_drop: float,
    max_bundle_size: int,
    trim_strategy: TrimStrategy = "latest",
    rng: Generator | None = None,
) -> SupportBundle | None:
    """Decay one support bundle and drop entries that fall below the weight floor."""

    delta_t = np.maximum(float(t_now) - bundle.t_bundle, 0.0)
    decayed_weights = bundle.w_bundle * decay_weight(delta_t, lambda_decay)
    keep_mask = (
        decayed_weights >= w_drop
        if w_drop > 0.0
        else np.ones_like(decayed_weights, dtype=bool)
    )
    if not np.any(keep_mask):
        return None

    phi_bundle = bundle.phi_bundle[keep_mask]
    w_bundle = decayed_weights[keep_mask]
    t_bundle = bundle.t_bundle[keep_mask]
    phi_bundle, w_bundle, t_bundle = trim_support_bundle(
        phi_bundle,
        w_bundle,
        t_bundle,
        max_size=max_bundle_size,
        strategy=trim_strategy,
        rng=rng,
    )
    return SupportBundle(
        unit_id=bundle.unit_id,
        phi_bundle=phi_bundle,
        w_bundle=w_bundle,
        t_bundle=t_bundle,
        class_label=bundle.class_label,
        status=bundle.status,
    )


def build_support_bundles_from_localization(
    features: FeatureArray,
    localization: LocalizationResult,
    *,
    t_now: float,
    max_bundle_size: int,
    namespace: str,
    mode: Literal["init", "refresh", "inference"] = "init",
    trim_strategy: TrimStrategy = "latest",
    residual_handling: ResidualHandling = "preserve",
    rng: Generator | None = None,
) -> dict[str, SupportBundle]:
    """Build regular and fragile support bundles from localization output."""

    features = np.asarray(features, dtype=np.float64)
    supports: dict[str, SupportBundle] = {}
    ordered_regular_units: list[str] = []
    for unit_id in localization.unit_ids.tolist():
        if isinstance(unit_id, str) and unit_id not in ordered_regular_units:
            ordered_regular_units.append(unit_id)

    for unit_id in ordered_regular_units:
        indices = np.flatnonzero(localization.unit_ids == unit_id).astype(np.int64)
        phi_bundle = features[indices]
        w_bundle = np.ones(phi_bundle.shape[0], dtype=np.float64)
        t_bundle = np.full(phi_bundle.shape[0], t_now, dtype=np.float64)
        phi_bundle, w_bundle, t_bundle = trim_support_bundle(
            phi_bundle,
            w_bundle,
            t_bundle,
            max_size=max_bundle_size,
            strategy=trim_strategy,
            rng=rng,
        )
        supports[unit_id] = SupportBundle(
            unit_id=unit_id,
            phi_bundle=phi_bundle,
            w_bundle=w_bundle,
            t_bundle=t_bundle,
            class_label=localization.unit_to_class[unit_id],
            status="temp_regular",
        )

    for fragile_counter, residual in enumerate(localization.residual_components):
        if residual_handling == "drop":
            continue
        fragile_unit_id = f"{namespace}:fragile:{fragile_counter}"
        phi_bundle = features[np.asarray(residual.indices, dtype=np.int64)]
        w_bundle = np.ones(phi_bundle.shape[0], dtype=np.float64)
        t_bundle = np.full(phi_bundle.shape[0], t_now, dtype=np.float64)
        if residual_handling == "force_merge":
            same_class_units = [
                unit_id
                for unit_id, bundle in supports.items()
                if bundle.class_label == residual.class_label
            ]
            if same_class_units:
                merge_target = min(
                    same_class_units,
                    key=lambda unit_id: float(
                        np.linalg.norm(
                            np.mean(supports[unit_id].phi_bundle, axis=0)
                            - np.mean(phi_bundle, axis=0)
                        )
                    ),
                )
                merged_phi = np.vstack([supports[merge_target].phi_bundle, phi_bundle])
                merged_w = np.concatenate([supports[merge_target].w_bundle, w_bundle])
                merged_t = np.concatenate([supports[merge_target].t_bundle, t_bundle])
                merged_phi, merged_w, merged_t = trim_support_bundle(
                    merged_phi,
                    merged_w,
                    merged_t,
                    max_size=max_bundle_size,
                    strategy=trim_strategy,
                    rng=rng,
                )
                supports[merge_target] = SupportBundle(
                    unit_id=supports[merge_target].unit_id,
                    phi_bundle=merged_phi,
                    w_bundle=merged_w,
                    t_bundle=merged_t,
                    class_label=supports[merge_target].class_label,
                    status=supports[merge_target].status,
                )
                continue
        status: SupportStatus = "temp_fragile" if mode in {"init", "refresh"} else "temp_tiny"
        phi_bundle, w_bundle, t_bundle = trim_support_bundle(
            phi_bundle,
            w_bundle,
            t_bundle,
            max_size=max_bundle_size,
            strategy=trim_strategy,
            rng=rng,
        )
        supports[fragile_unit_id] = SupportBundle(
            unit_id=fragile_unit_id,
            phi_bundle=phi_bundle,
            w_bundle=w_bundle,
            t_bundle=t_bundle,
            class_label=residual.class_label,
            status=status,
        )

    return supports
