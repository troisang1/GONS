"""GONS config builders for the reproduction runners.

Single source of truth for the GONS FIXED config used in the main result and as
the FULL base of the ablation. Builds the canonical GONS config with the global
FIXED grid cell baked in.

GONS FIXED grid cell (argmax of the per-dataset-mean OS-HM sweep):
    threshold_quantile = 0.85
    enable_contrastive_rff = False
    score_support_penalty_alpha = 0.0
    enable_ncdr_classification = True
    scoring_mode = "min_distance"   (deployed)
    map_n_components = 512, d_proj_max = 32
    map_gamma = resolve_map_gamma(dataset, seed)  -- the kNN-20 bandwidth rule,
                identical for every dataset, evaluated on that dataset's own
                base-session training split
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent

# Per-dataset paths + base_class_ratio ONLY. No per-dataset model knobs live here.
# Datasets whose processed splits are not bundled are still listed; the runner
# skips a dataset whose train.csv is missing (with a clear message).
DATASETS: dict[str, dict] = {
    "toniot": {
        "train": "data/processed/toniot/train.csv",
        "cal": "data/processed/toniot/calibration.csv",
        "test": "data/processed/toniot/test.csv",
        "ratio": 0.5,
    },
    "nbaiot": {
        "train": "data/processed/nbaiot_5k/train.csv",
        "cal": "data/processed/nbaiot_5k/calibration.csv",
        "test": "data/processed/nbaiot_5k/test.csv",
        "ratio": 0.5,
    },
    "cicids2018": {
        "train": "data/processed/cicids2018_5k/train.csv",
        "cal": "data/processed/cicids2018_5k/calibration.csv",
        "test": "data/processed/cicids2018_5k/test.csv",
        "ratio": 0.5,
    },
    "5g_nidd": {
        "train": "data/processed/5g_nidd_5k/train.csv",
        "cal": "data/processed/5g_nidd_5k/calibration.csv",
        "test": "data/processed/5g_nidd_5k/test.csv",
        "ratio": 0.45,
    },
    "nsl_kdd": {
        "train": "data/processed/nsl_kdd_ta/train.csv",
        "cal": "data/processed/nsl_kdd_ta/calibration.csv",
        "test": "data/processed/nsl_kdd_ta/test.csv",
        "ratio": 0.5,
    },
    # Added 2026-08-03: the reported benchmark is seven datasets.
    "edge_iiotset": {
        "train": "data/processed/edge_iiotset_5k_ports/train.csv",
        "cal": "data/processed/edge_iiotset_5k_ports/calibration.csv",
        "test": "data/processed/edge_iiotset_5k_ports/test.csv",
        "ratio": 0.5,
    },
    "ciciot2023_cat": {
        "train": "data/processed/ciciot2023_category_10k/train.csv",
        "cal": "data/processed/ciciot2023_category_10k/calibration.csv",
        "test": "data/processed/ciciot2023_category_10k/test.csv",
        "ratio": 0.5,
    },
}

# The GONS FIXED grid cell (one global config, applied to every dataset).
# global_gi=4 from the canonical per-dataset-mean OS-HM sweep.
GONS_FIXED = {
    "threshold_quantile": 0.85,
    "enable_contrastive_rff": False,
    "score_support_penalty_alpha": 0.0,
    "enable_ncdr_classification": True,
    "scoring_mode": "min_distance",
}

# Per-dataset TUNED grid cells (argmax of the per-dataset s42 sweep). These are
# the FULL base of the ablation (the ablation uses the per-ds TUNED config, not
# the global FIXED one).
GONS_TUNED_PER_DS: dict[str, dict] = {
    "toniot": {"threshold_quantile": 0.85, "enable_contrastive_rff": True,
               "score_support_penalty_alpha": 0.20, "enable_ncdr_classification": True},
    "nbaiot": {"threshold_quantile": 0.85, "enable_contrastive_rff": False,
               "score_support_penalty_alpha": 0.0, "enable_ncdr_classification": False},
    "cicids2018": {"threshold_quantile": 0.90, "enable_contrastive_rff": False,
                   "score_support_penalty_alpha": 0.0, "enable_ncdr_classification": True},
    "5g_nidd": {"threshold_quantile": 0.85, "enable_contrastive_rff": False,
                "score_support_penalty_alpha": 0.20, "enable_ncdr_classification": True},
    "nsl_kdd": {"threshold_quantile": 0.85, "enable_contrastive_rff": False,
                "score_support_penalty_alpha": 0.20, "enable_ncdr_classification": True},
}

DATA_CONFIG_PATH = "configs/data/dataset.yaml"


def dataset_available(ds_name: str) -> bool:
    ds = DATASETS[ds_name]
    return (REPO / ds["train"]).exists() and (REPO / ds["cal"]).exists() and (REPO / ds["test"]).exists()


# Stored in place of a resolved bandwidth when the dataset's splits are not
# bundled. maps/explicit.py raises on it rather than substituting a value.
MAP_GAMMA_UNRESOLVED = -2.0

_GAMMA_CACHE: dict[tuple[str, int], float] = {}


def resolve_map_gamma(ds_name: str, seed: int) -> float:
    """Evaluate the kNN-20 bandwidth rule on this (dataset, seed)'s base split.

    The bandwidth is a fit-time constant derived from the base-session training
    matrix, and it is the same constant for every session of the run. It is resolved
    once here rather than inside the map, so that it is always taken from the base
    split specifically.

    The base split must match the pipeline's exactly: `int(len(labels) * ratio)`
    truncates (it is not `round`), the label list is lowercased and sorted before
    sampling, and the preprocessor is fit on the base-class training frame only.

    Reference value: N-BaIoT seed 42 -> 0.0173399338.
    """
    key = (ds_name, seed)
    if key in _GAMMA_CACHE:
        return _GAMMA_CACHE[key]

    import random as _random

    import numpy as np
    import pandas as pd

    from gons.config.data import PreprocessingConfig
    from gons.maps.explicit import _compute_knn_gamma
    from gons.preprocess import fit_schema_and_preprocessor, transform_rows

    ds = DATASETS[ds_name]
    train = pd.read_csv(REPO / ds["train"])
    labels = sorted(train["label"].astype(str).str.lower().unique())
    n_base = min(max(1, int(len(labels) * float(ds["ratio"]))), len(labels) - 1)
    base = sorted(_random.Random(seed).sample(labels, k=n_base))
    tr_base = train[train["label"].astype(str).str.lower().isin(base)].reset_index(drop=True)

    bundle = fit_schema_and_preprocessor(
        tr_base, label_column="label",
        config=PreprocessingConfig(numeric_scaler="quantile"),
    )
    X, _ = transform_rows(bundle.schema, bundle.preprocessor, tr_base, label_column="label")
    gamma = _compute_knn_gamma(np.asarray(X, dtype=np.float64))
    _GAMMA_CACHE[key] = gamma
    return gamma


def make_gons_cfg(ds_name: str, seed: int, tag: str, overrides: dict | None = None) -> dict:
    """Build a GONS FSCIL config for one (dataset, seed).

    `overrides` (used by the ablation) mutates the model dict AFTER the FIXED
    config is applied — e.g. {"scoring_mode": "energy"} or {"map_n_components": 128}.
    """

    ds = DATASETS[ds_name]
    fixed = GONS_FIXED
    model = {
        "label_column": "label",
        "seed": seed,
        "map_kind": "rff",
        "map_kernel": "rbf",
        # One rule, applied identically to every dataset, resolved against that
        # dataset's own base-session training split. Resolving reads the training
        # CSV, so when the splits are not bundled the UNRESOLVED sentinel is stored
        # instead; it raises on use, and a run is impossible without the data anyway.
        "map_gamma": (
            resolve_map_gamma(ds_name, seed)
            if dataset_available(ds_name)
            else MAP_GAMMA_UNRESOLVED
        ),
        "map_n_components": 512,
        "d_proj_max": 32,
        "residual_handling": "drop",
        "energy_temperature": 1.0,
        "enable_contrastive_rff": fixed["enable_contrastive_rff"],
        "score_support_penalty_alpha": fixed["score_support_penalty_alpha"],
        "score_support_penalty": fixed["score_support_penalty_alpha"] > 0,
        "score_support_penalty_fn": "log",
        "enable_ncdr_classification": fixed["enable_ncdr_classification"],
        "ncdr_threshold": 0.85,
        "threshold_quantile": fixed["threshold_quantile"],
        "scoring_mode": fixed["scoring_mode"],
        "refresh_mode": "single_pass",
        "enable_local_affine_screen": False,
        "support_trim_strategy": "uniform",
        "preprocessing": {"numeric_scaler": "quantile"},
    }
    cfg = {
        "task": "fscil",
        "experiment_name": tag,
        "method": "gons",
        "dataset_config_path": DATA_CONFIG_PATH,
        "train_path": ds["train"],
        "calibration_path": ds["cal"],
        "test_path": ds["test"],
        "output_root": "artifacts/runs",
        "base_class_ratio": ds["ratio"],
        "base_class_seed": seed,
        "novel_class_order": "sorted",
        "novel_support_fraction": 1.0,
        "novel_support_max_k": 5,
        "refresh_after_session": True,
        "model": model,
    }
    if overrides:
        # refresh_after_session is a top-level key (used by the no_refresh arm).
        if "refresh_after_session" in overrides:
            cfg["refresh_after_session"] = overrides.pop("refresh_after_session")
        cfg["model"].update(overrides)
    return cfg


def make_gons_tuned_cfg(ds_name: str, seed: int, tag: str, overrides: dict | None = None) -> dict:
    """GONS per-dataset TUNED config — the FULL base of the ablation."""

    cfg = make_gons_cfg(ds_name, seed, tag=tag)
    tuned = GONS_TUNED_PER_DS[ds_name]
    m = cfg["model"]
    m["threshold_quantile"] = tuned["threshold_quantile"]
    m["enable_contrastive_rff"] = tuned["enable_contrastive_rff"]
    m["score_support_penalty_alpha"] = tuned["score_support_penalty_alpha"]
    m["score_support_penalty"] = tuned["score_support_penalty_alpha"] > 0
    m["enable_ncdr_classification"] = tuned["enable_ncdr_classification"]
    m["scoring_mode"] = "min_distance"
    if overrides:
        if "refresh_after_session" in overrides:
            cfg["refresh_after_session"] = overrides.pop("refresh_after_session")
        m.update(overrides)
    return cfg
