"""GONS (gate-OFF) config builders for the reproduction runners.

Single source of truth for the GONS FIXED config used in the main result and as
the FULL base of the ablation. Builds the canonical GONS config with the gate
HARD-OFF (gate-OFF invariant) and the global FIXED grid cell baked in.

GONS FIXED grid cell (argmax of the per-dataset-mean OS-HM sweep, global_gi=4):
    threshold_quantile = 0.85
    enable_contrastive_rff = False
    score_support_penalty_alpha = 0.0
    enable_ncdr_classification = True
    scoring_mode = "min_distance"   (deployed)
    map_n_components = 512, d_proj_max = 32
Per-dataset map_gamma is kept from the canonical per-dataset "ours" family.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent

# Per-dataset paths + base_class_ratio + per-ds map_gamma (canonical "ours" family).
# Datasets whose processed splits are not bundled are still listed; the runner
# skips a dataset whose train.csv is missing (with a clear message).
DATASETS: dict[str, dict] = {
    "toniot": {
        "train": "data/processed/toniot/train.csv",
        "cal": "data/processed/toniot/calibration.csv",
        "test": "data/processed/toniot/test.csv",
        "ratio": 0.5,
        "map_gamma": 1.0,
    },
    "nbaiot": {
        "train": "data/processed/nbaiot_5k/train.csv",
        "cal": "data/processed/nbaiot_5k/calibration.csv",
        "test": "data/processed/nbaiot_5k/test.csv",
        "ratio": 0.5,
        "map_gamma": -1.0,
    },
    "cicids2018": {
        "train": "data/processed/cicids2018_5k/train.csv",
        "cal": "data/processed/cicids2018_5k/calibration.csv",
        "test": "data/processed/cicids2018_5k/test.csv",
        "ratio": 0.5,
        "map_gamma": 1.0,
    },
    "5g_nidd": {
        "train": "data/processed/5g_nidd_5k/train.csv",
        "cal": "data/processed/5g_nidd_5k/calibration.csv",
        "test": "data/processed/5g_nidd_5k/test.csv",
        "ratio": 0.45,
        "map_gamma": -1.0,
    },
    "nsl_kdd": {
        "train": "data/processed/nsl_kdd_ta/train.csv",
        "cal": "data/processed/nsl_kdd_ta/calibration.csv",
        "test": "data/processed/nsl_kdd_ta/test.csv",
        "ratio": 0.5,
        "map_gamma": -1.0,
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

DATA_CONFIG_PATH = "configs/data/ciciot2023.yaml"


def dataset_available(ds_name: str) -> bool:
    ds = DATASETS[ds_name]
    return (REPO / ds["train"]).exists() and (REPO / ds["cal"]).exists() and (REPO / ds["test"]).exists()


def make_gons_cfg(ds_name: str, seed: int, tag: str, overrides: dict | None = None) -> dict:
    """Build a GONS (gate-OFF) track_a_fscil config for one (dataset, seed).

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
        "map_gamma": ds["map_gamma"],
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
        # ---- HARD gate-OFF: the GONS invariant ----
        "enable_dvmad_gate": False,
        "enable_ewm": False,
        "ewm_replace_gate": False,
    }
    cfg = {
        "task": "track_a_fscil",
        "experiment_name": tag,
        "method": "bc_mrnfst",
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
    """GONS per-dataset TUNED config (gate-OFF) — the FULL base of the ablation."""

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


def gate_active(model: dict) -> bool:
    return bool(model.get("enable_dvmad_gate")) and bool(model.get("enable_ewm"))
