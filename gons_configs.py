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

# Per-dataset TUNED grid cells -- the per-dataset argmax of the exhaustive
# selection grid (32 cells x 6 bandwidth rules), each scored by ITS OWN 10-seed
# mean. This is the "every method tuned per dataset" arm the paper compares the
# ONE fixed config against; it is NOT the base of the ablation (that is FIXED --
# see run_gons_ablation.py).
#
# The bandwidth rule is part of what the tuned arm selects, so each entry carries
# its own `gamma_rule`; the FIXED config holds the rule at kNN-20 for every
# dataset. Reproducing `results/headline_fixed_vs_tuned_7ds.csv` (Tuned column)
# needs both the grid cell and the rule.
#
# Per-dataset 10-seed OS-HM these reach (results/headline_fixed_vs_tuned_7ds.csv):
#   toniot .8125  nbaiot .7901  cicids2018 .7192  5g_nidd .8564
#   nsl_kdd .6510  edge_iiotset .6647  ciciot2023_cat .6243   -> mean .7312
# against FIXED .7082, i.e. the cost of shipping one global config is +0.0230.
GONS_TUNED_PER_DS: dict[str, dict] = {
    "toniot": {"threshold_quantile": 0.85, "enable_contrastive_rff": False,
               "score_support_penalty_alpha": 0.20, "enable_ncdr_classification": False,
               "gamma_rule": "knn10"},
    "nbaiot": {"threshold_quantile": 0.85, "enable_contrastive_rff": False,
               "score_support_penalty_alpha": 0.20, "enable_ncdr_classification": False,
               "gamma_rule": "knn50"},
    "cicids2018": {"threshold_quantile": 0.95, "enable_contrastive_rff": False,
                   "score_support_penalty_alpha": 0.0, "enable_ncdr_classification": True,
                   "gamma_rule": "knn5"},
    "5g_nidd": {"threshold_quantile": 0.99, "enable_contrastive_rff": False,
                "score_support_penalty_alpha": 0.20, "enable_ncdr_classification": False,
                "gamma_rule": "knn20"},
    "nsl_kdd": {"threshold_quantile": 0.85, "enable_contrastive_rff": False,
                "score_support_penalty_alpha": 0.0, "enable_ncdr_classification": True,
                "gamma_rule": "median"},
    "edge_iiotset": {"threshold_quantile": 0.90, "enable_contrastive_rff": True,
                     "score_support_penalty_alpha": 0.20, "enable_ncdr_classification": False,
                     "gamma_rule": "knn20"},
    "ciciot2023_cat": {"threshold_quantile": 0.85, "enable_contrastive_rff": False,
                       "score_support_penalty_alpha": 0.0, "enable_ncdr_classification": False,
                       "gamma_rule": "knn5"},
}

# The bandwidth rule the FIXED config applies to every dataset.
FIXED_GAMMA_RULE = "knn20"

# ============================================================================ #
# MODEL_SEED IS A CONSTANT, AND IT IS NOT THE PROTOCOL SEED.                    #
#                                                                              #
# Two different seeds are in play and conflating them does not reproduce the    #
# published numbers:                                                           #
#                                                                              #
#   base_class_seed  = the PROTOCOL seed, 42..51. Picks the base/novel class    #
#                      split and the few-shot support rows. This is what        #
#                      "10 seeds" means in the paper.                          #
#   model.seed       = the MODEL rng (the RFF draw). Held at 42 on every        #
#                      published cell.                                         #
#                                                                              #
# Every published run was produced by `app.py run-experiment`, whose `--seed`   #
# option defaults to 42 and is passed to the runner unconditionally, so the     #
# runner overwrote model.seed with 42 on every cell regardless of the protocol  #
# seed. Letting model.seed follow the protocol seed instead redraws the RFF per #
# seed: seed 42 still matches exactly (42 == 42) while seeds 43..51 drift, in   #
# both directions, by up to ~0.03 OS-HM per cell.                              #
# ============================================================================ #
MODEL_SEED = 42

DATA_CONFIG_PATH = "configs/data/dataset.yaml"


def dataset_available(ds_name: str) -> bool:
    ds = DATASETS[ds_name]
    return (REPO / ds["train"]).exists() and (REPO / ds["cal"]).exists() and (REPO / ds["test"]).exists()


# Stored in place of a resolved bandwidth when the dataset's splits are not
# bundled. maps/explicit.py raises on it rather than substituting a value.
MAP_GAMMA_UNRESOLVED = -2.0

_GAMMA_CACHE: dict[tuple[str, int, str], float] = {}


def resolve_map_gamma(ds_name: str, seed: int, rule: str = "knn20") -> float:
    """Evaluate a bandwidth rule on this (dataset, seed)'s base split.

    The bandwidth is a fit-time constant derived from the base-session training
    matrix, and it is the same constant for every session of the run. It is resolved
    once here rather than inside the map, so that it is always taken from the base
    split specifically.

    `rule` is `knn<k>` (gamma = 1 / mean squared distance to the k-th nearest
    neighbour) or `median` (gamma = 1 / median pairwise squared distance). The
    FIXED config uses kNN-20 on every dataset; the per-dataset TUNED arm selects
    the rule along with the grid cell (see GONS_TUNED_PER_DS). Both branches call
    the library's own primitives in `gons.maps.explicit`, so a rule resolved here
    is bit-identical to the one the map would compute internally.

    The base split must match the pipeline's exactly: `int(len(labels) * ratio)`
    truncates (it is not `round`), the label list is lowercased and sorted before
    sampling, and the preprocessor is fit on the base-class training frame only.

    Reference value: N-BaIoT seed 42, kNN-20 -> 0.0173399338.
    """
    key = (ds_name, seed, rule)
    if key in _GAMMA_CACHE:
        return _GAMMA_CACHE[key]

    import random as _random

    import numpy as np
    import pandas as pd

    from gons.config.data import PreprocessingConfig
    from gons.maps.explicit import _compute_knn_gamma, _compute_median_heuristic_gamma
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
    Xf = np.asarray(X, dtype=np.float64)
    if rule == "median":
        gamma = _compute_median_heuristic_gamma(Xf)
    elif rule.startswith("knn") and rule[3:].isdigit():
        gamma = _compute_knn_gamma(Xf, k=int(rule[3:]))
    else:
        raise ValueError(f"unknown bandwidth rule {rule!r}; expected 'median' or 'knn<k>'")
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
        # NOT the protocol seed -- see MODEL_SEED above. The protocol seed enters
        # through base_class_seed below.
        "seed": MODEL_SEED,
        "map_kind": "rff",
        "map_kernel": "rbf",
        # One rule, applied identically to every dataset, resolved against that
        # dataset's own base-session training split. Resolving reads the training
        # CSV, so when the splits are not bundled the UNRESOLVED sentinel is stored
        # instead; it raises on use, and a run is impossible without the data anyway.
        "map_gamma": (
            resolve_map_gamma(ds_name, seed, rule=FIXED_GAMMA_RULE)
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
    """GONS per-dataset TUNED config — the per-dataset-tuned comparison arm.

    Applies that dataset's own argmax grid cell AND its own bandwidth rule. The
    rule matters: five of the seven tuned winners sit on a rule other than the
    FIXED kNN-20, so holding the bandwidth at kNN-20 here would not reproduce
    `results/headline_fixed_vs_tuned_7ds.csv`.

    This is NOT the ablation base — the shipped ablation is on the FIXED config
    (see run_gons_ablation.py).
    """

    cfg = make_gons_cfg(ds_name, seed, tag=tag)
    tuned = GONS_TUNED_PER_DS[ds_name]
    m = cfg["model"]
    m["threshold_quantile"] = tuned["threshold_quantile"]
    m["enable_contrastive_rff"] = tuned["enable_contrastive_rff"]
    m["score_support_penalty_alpha"] = tuned["score_support_penalty_alpha"]
    m["score_support_penalty"] = tuned["score_support_penalty_alpha"] > 0
    m["enable_ncdr_classification"] = tuned["enable_ncdr_classification"]
    m["scoring_mode"] = "min_distance"
    m["map_gamma"] = (
        resolve_map_gamma(ds_name, seed, rule=tuned["gamma_rule"])
        if dataset_available(ds_name)
        else MAP_GAMMA_UNRESOLVED
    )
    if overrides:
        if "refresh_after_session" in overrides:
            cfg["refresh_after_session"] = overrides.pop("refresh_after_session")
        m.update(overrides)
    return cfg
