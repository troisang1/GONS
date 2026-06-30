"""Offline GONS fit orchestration."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from gons.calibration.thresholds import (
    fit_evm_weibull_per_class,
    fit_per_class_quantile_thresholds,
    fit_per_prototype_thresholds,
    recalibrate_global_threshold,
    update_calibration_state,
)
from gons.config.model import GONSConfig
from gons.localization import localize_mutual_rnn_units
from gons.maps import (
    PreWhitenedExplicitMap,
    WhitenedExplicitMap,
    compute_within_class_whitening,
    fit_explicit_map,
)
from gons.preprocess import fit_schema_and_preprocessor, transform_rows
from gons.projection.core import (
    build_threshold_metadata,
    compute_soft_null_projection,
    rebuild_stats_from_support_store,
    resolve_refresh_scales,
)
from gons.prototype.bank import (
    build_stable_prototypes_from_support_store,
    cap_prototype_bank,
    rebuild_dense_proto_arrays,
    refresh_prototype_tau_overrides,
)
from gons.state import build_support_bundles_from_localization
from gons.state.model import (
    GONSModel,
    enforce_active_state_budgets,
    initialize_model_shell,
    update_sanity_metrics,
)


def _sample_validation_exemplars(
    Phi_train: np.ndarray,
    y_train: np.ndarray,
    *,
    budget: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Sample a class-balanced explicit-space exemplar store for refresh validation."""

    if budget < 1 or len(y_train) == 0:
        return {}
    unique_labels = np.unique(y_train)
    if unique_labels.size == 0:
        return {}

    rng = np.random.default_rng(seed)
    per_class = max(1, budget // len(unique_labels))
    exemplars: dict[str, np.ndarray] = {}
    for class_label in unique_labels:
        indices = np.flatnonzero(y_train == class_label).astype(np.int64)
        if indices.size == 0:
            continue
        sample_size = min(per_class, indices.size)
        chosen = rng.choice(indices, size=sample_size, replace=False).astype(np.int64)
        exemplars[str(class_label)] = np.asarray(Phi_train[chosen], dtype=np.float64)
    return exemplars


def fit_gons(
    df_train: pd.DataFrame,
    df_cal: pd.DataFrame,
    config: GONSConfig,
) -> GONSModel:
    """Run the initial offline GONS fit path end to end."""

    bundle = fit_schema_and_preprocessor(
        df_train,
        label_column=config.label_column,
        config=config.preprocessing,
    )
    X_train, _ = transform_rows(
        bundle.schema,
        bundle.preprocessor,
        df_train,
        label_column=config.label_column,
    )
    y_train = df_train[config.label_column].astype(str).to_numpy(dtype=object)

    map_fit_input = X_train
    pre_map_whitening = None
    if config.enable_pre_map_whitening:
        pre_map_whitening = compute_within_class_whitening(X_train, y_train)
        map_fit_input = np.asarray(X_train @ pre_map_whitening.T, dtype=np.float64)

    phi = fit_explicit_map(
        map_fit_input,
        config.explicit_map_config(),
        labels=y_train,
    )
    if pre_map_whitening is not None:
        phi = PreWhitenedExplicitMap(base_map=phi, whitening_matrix=pre_map_whitening)

    Phi_train = np.asarray(phi.transform(X_train), dtype=np.float64)
    if config.enable_contrastive_rff:
        phi = WhitenedExplicitMap.from_fitted_base(phi, Phi_train, y_train)
        Phi_train = np.asarray(phi.transform(X_train), dtype=np.float64)
    phi_state = phi.to_state()

    # PA-iNN: augment under-represented classes with synthetic anchors for
    # scatter computation.  The augmented features are used only for
    # localization and support bundle construction; the original (real)
    # samples are what get stored in the support store.
    Phi_for_localization = Phi_train
    y_for_localization = y_train
    if config.enable_pa_inn:
        from gons.prototype.augmentation import augment_support_for_scatter

        Phi_for_localization, y_for_localization = augment_support_for_scatter(
            Phi_train,
            y_train,
            min_samples_per_class=config.pa_inn_min_samples,
            n_anchors=config.pa_inn_n_anchors,
            regularization=config.pa_inn_regularization,
            seed=config.seed,
        )

    localization = localize_mutual_rnn_units(
        Phi_for_localization,
        y_for_localization,
        r=config.r,
        min_microclass_size=config.min_microclass_size,
        namespace="init",
        metric=config.localization_metric,
    )
    # If PA-iNN augmented the features, truncate localization to real rows.
    if localization.unit_ids.shape[0] > Phi_train.shape[0]:
        from gons.localization.mutual_rnn import LocalizationResult

        localization = LocalizationResult(
            unit_ids=localization.unit_ids[: Phi_train.shape[0]],
            unit_to_class=localization.unit_to_class,
            residual_components=localization.residual_components,
            effective_r_by_class=localization.effective_r_by_class,
        )
    init_supports = build_support_bundles_from_localization(
        Phi_train,
        localization,
        t_now=0.0,
        max_bundle_size=config.R_q_max,
        namespace="init",
        mode="init",
        trim_strategy=config.support_trim_strategy,
        residual_handling=config.residual_handling,
        rng=np.random.default_rng(config.seed),
    )

    model = initialize_model_shell(
        schema=bundle.schema,
        preprocessor=bundle.preprocessor,
        phi=phi,
        phi_state=phi_state,
        support_store=init_supports,
        config=config,
    )
    if config.enable_refresh_validation:
        model.validation_exemplars = _sample_validation_exemplars(
            Phi_train,
            y_train,
            budget=int(config.refresh_validation_budget),
            seed=int(config.seed),
        )
    model = enforce_active_state_budgets(model)
    model.stats = rebuild_stats_from_support_store(model)

    projection_result = compute_soft_null_projection(model)
    model.W_proj = projection_result.W_proj
    model.projection_meta = projection_result

    scales = resolve_refresh_scales(
        model,
        optional_small_spectrum=projection_result.small_eigvals,
        optional_range_spectrum=projection_result.range_eigvals,
    )
    model.proto_bank = build_stable_prototypes_from_support_store(model, scales)
    model = cap_prototype_bank(model)
    model = rebuild_dense_proto_arrays(model)

    model = update_calibration_state(model, df_cal)
    model.tau_global = recalibrate_global_threshold(model)
    if model.config.enable_per_class_quantile_threshold:
        model.per_class_thresholds = fit_per_class_quantile_thresholds(model)
    model = refresh_prototype_tau_overrides(model)
    if model.config.enable_evm_calibration:
        model.evm_weibull_params = fit_evm_weibull_per_class(model)
    if model.config.enable_per_proto_threshold:
        for proto_id, tau_local in fit_per_prototype_thresholds(model).items():
            if proto_id in model.proto_bank:
                model.proto_bank[proto_id] = replace(model.proto_bank[proto_id], tau_local=tau_local)
    model.threshold_meta = build_threshold_metadata(
        scales,
        tau_global=model.tau_global,
        calibration_mode=model.calibration_state.mode or "insufficient",
    )
    model = update_sanity_metrics(model)
    return model
