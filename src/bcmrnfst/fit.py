"""Offline GONS fit orchestration."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from bcmrnfst.calibration.thresholds import (
    fit_evm_weibull_per_class,
    fit_per_class_quantile_thresholds,
    fit_per_prototype_thresholds,
    recalibrate_global_threshold,
    update_calibration_state,
)
from bcmrnfst.config.model import BCMRNFSTConfig
from bcmrnfst.localization import localize_mutual_rnn_units
from bcmrnfst.maps import (
    PreWhitenedExplicitMap,
    WhitenedExplicitMap,
    compute_within_class_whitening,
    fit_explicit_map,
)
from bcmrnfst.preprocess import fit_schema_and_preprocessor, transform_rows
from bcmrnfst.projection.core import (
    build_threshold_metadata,
    compute_soft_null_projection,
    rebuild_stats_from_support_store,
    resolve_refresh_scales,
)
from bcmrnfst.projection.dvmad_gate import calibrate_dvmad_tau, fit_dvmad_gate
from bcmrnfst.prototype.bank import (
    build_stable_prototypes_from_support_store,
    cap_prototype_bank,
    rebuild_dense_proto_arrays,
    refresh_prototype_tau_overrides,
)
from bcmrnfst.state import build_support_bundles_from_localization
from bcmrnfst.state.model import (
    BCMRNFSTModel,
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


def fit_bc_mrnfst(
    df_train: pd.DataFrame,
    df_cal: pd.DataFrame,
    config: BCMRNFSTConfig,
) -> BCMRNFSTModel:
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
        from bcmrnfst.prototype.augmentation import augment_support_for_scatter

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
        from bcmrnfst.localization.mutual_rnn import LocalizationResult

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
    if getattr(model.config, "enable_dvmad_gate", False) or getattr(model.config, "enable_carag", False):
        if getattr(model.config, "enable_carag", False):
            from bcmrnfst.projection.carag import calibrate_carag_beta, fit_carag_gate
            gate_core, gate_tau, carag_state = fit_carag_gate(model)
            model.dvmad_core = gate_core
            model.dvmad_tau = gate_tau
            model.carag_state = carag_state
        else:
            gate_core, gate_tau = fit_dvmad_gate(model)
            model.dvmad_core = gate_core
            model.dvmad_tau = gate_tau

        if gate_core is not None and df_cal is not None:
            X_cal, _ = transform_rows(
                bundle.schema,
                bundle.preprocessor,
                df_cal,
                label_column=config.label_column,
            )
            phi_cal = np.asarray(model.phi.transform(X_cal), dtype=np.float64)
            calibration_features = (
                phi_cal if model.config.dvmad_gate_use_phi else phi_cal @ model.W_proj
            )
            calibrate_dvmad_tau(model, calibration_features)
            if getattr(model.config, "enable_carag", False) and model.carag_state is not None:
                pass
                from bcmrnfst.projection.carag import calibrate_carag_beta as _cb
                # NCM-per-class on calibration for closed-form β.
                cal_labels = df_cal[config.label_column].astype(str).to_numpy(dtype=object)
                W_proj = model.W_proj
                if W_proj is not None and model.prototype_matrix is not None:
                    radii = np.maximum(np.asarray(model.radii_vec, dtype=np.float64), 1e-9)
                    proto_mat = np.asarray(model.prototype_matrix, dtype=np.float64)
                    Z_cal = phi_cal @ W_proj
                    classes_sorted = sorted(model.carag_state.anchor_phi.keys())
                    if classes_sorted:
                        n_cls = len(classes_sorted)
                        cls_to_idx = {c: i for i, c in enumerate(classes_sorted)}
                        ncm = np.full((Z_cal.shape[0], n_cls), np.inf, dtype=np.float64)
                        from bcmrnfst.prototype.bank import serialize_projection_meta as _spm  # noqa
                        # Build proto class list from proto_bank
                        class_by_proto = [model.proto_bank[pid].class_label for pid in model.proto_ids]
                        for p_idx in range(proto_mat.shape[0]):
                            cls = class_by_proto[p_idx]
                            if cls not in cls_to_idx:
                                continue
                            d = np.linalg.norm(Z_cal - proto_mat[p_idx], axis=1) / radii[p_idx]
                            ncm[:, cls_to_idx[cls]] = np.minimum(ncm[:, cls_to_idx[cls]], d)
                        ncm[~np.isfinite(ncm)] = 1.0
                        model.carag_state = _cb(
                            model, model.carag_state, phi_cal, cal_labels,
                            ncm_distance_per_class=ncm, class_order=classes_sorted,
                        )
    if getattr(model.config, "enable_ewm", False):
        from bcmrnfst.projection.dvmad_ewm import fit_ewm, ewm_score_per_class
        model.ewm_state = fit_ewm(model)
        # Compute EWM-tau from cal features whenever EWM is enabled (used for
        # both gate-replace and AND-combine paths). Theorem D3 / DKW analogue.
        if df_cal is not None and model.ewm_state.centroid_phi_per_class:
            X_cal_e, _ = transform_rows(
                bundle.schema, bundle.preprocessor, df_cal,
                label_column=config.label_column,
            )
            phi_cal_e = np.asarray(model.phi.transform(X_cal_e), dtype=np.float64)
            ewm_scores_cal, _ = ewm_score_per_class(model.ewm_state, phi_cal_e)
            if ewm_scores_cal.size:
                ewm_min_cal = ewm_scores_cal.min(axis=1)
                q = float(getattr(model.config, "dvmad_quantile", 0.99))
                tau_e = float(np.quantile(ewm_min_cal, q))
                model._ewm_tau = tau_e
                if bool(getattr(model.config, "ewm_replace_gate", False)):
                    model.dvmad_tau = tau_e
    if getattr(model.config, "enable_nsd_gate", False):
        # NSD-Gate: re-expressed Sw-EWM gate via null-space + Fisher
        # weights. Calibrate tau through the SAME path as the EWM-replace
        # gate (min-over-class score, quantile dvmad_quantile -> model.dvmad_tau),
        # so the gate is a protocol-parity drop-in replacement for the deployed
        # ewm_min_gate_score. Does NOT construct a DVMADCore (core.py untouched).
        from bcmrnfst.projection.dvmad_nsd_gate import fit_nsd_gate, nsd_gate_score
        model.nsd_gate_state = fit_nsd_gate(model)
        if df_cal is not None and model.nsd_gate_state.ewm.centroid_phi_per_class:
            X_cal_n, _ = transform_rows(
                bundle.schema, bundle.preprocessor, df_cal,
                label_column=config.label_column,
            )
            phi_cal_n = np.asarray(model.phi.transform(X_cal_n), dtype=np.float64)
            nsd_min_cal = nsd_gate_score(model.nsd_gate_state, phi_cal_n)
            if nsd_min_cal.size:
                q = float(getattr(model.config, "dvmad_quantile", 0.99))
                tau_n = float(np.quantile(nsd_min_cal, q))
                model.nsd_gate_state.tau = tau_n
                model.dvmad_tau = tau_n
    if getattr(model.config, "enable_pct_rd", False):
        from bcmrnfst.projection.pct_rd import calibrate_pct_rd, fit_pct_rd
        pct_state = fit_pct_rd(model)
        if pct_state.cores and df_cal is not None:
            X_cal, _ = transform_rows(
                bundle.schema,
                bundle.preprocessor,
                df_cal,
                label_column=config.label_column,
            )
            phi_cal = np.asarray(model.phi.transform(X_cal), dtype=np.float64)
            cal_labels = df_cal[config.label_column].astype(str).to_numpy(dtype=object)
            pct_state = calibrate_pct_rd(model, pct_state, phi_cal, cal_labels)
        model.pct_rd_state = pct_state
    model = update_sanity_metrics(model)
    return model
