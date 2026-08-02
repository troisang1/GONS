"""GONS release unit tests: config invariants, scorer properties, parity.

Run: `pytest -q` from the repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # gons_configs / gons_run live at repo root
sys.path.insert(0, str(REPO / "src"))

from gons.evaluation.aggregate_openset import open_set_session_metrics  # noqa: E402
from gons_configs import (  # noqa: E402
    GONS_TUNED_PER_DS,
    dataset_available,
    make_gons_cfg,
    make_gons_tuned_cfg,
)


# --------------------------------------------------------------------------- #
# Config invariants.                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ds", list(GONS_TUNED_PER_DS))
def test_tuned_cfg_scoring_mode(ds):
    cfg = make_gons_tuned_cfg(ds, 42, tag="t")
    assert cfg["model"]["scoring_mode"] == "min_distance"


def test_fixed_cfg_is_the_global_cell():
    m = make_gons_cfg("toniot", 42, tag="t")["model"]
    assert m["threshold_quantile"] == 0.85
    assert m["enable_contrastive_rff"] is False
    assert m["score_support_penalty_alpha"] == 0.0
    assert m["enable_ncdr_classification"] is True
    assert m["map_n_components"] == 512


# --------------------------------------------------------------------------- #
# Official scorer properties (single source of truth).                        #
# --------------------------------------------------------------------------- #
def _frame(true_labels, pred_labels):
    return pd.DataFrame({"true_label": true_labels, "predicted_label": pred_labels})


def test_scorer_perfect_known_and_reject():
    # all knowns classified right, all unknowns rejected -> OS-HM = 1.
    df = _frame(["a", "b", "z", "z"], ["a", "b", "unknown", "unknown"])
    m = open_set_session_metrics(df, base_classes={"a", "b"}, admitted_novel=set())
    assert m["ccr"] == 1.0
    assert m["tur"] == 1.0
    assert m["os_hm"] == 1.0


def test_scorer_all_absorbed_unknowns():
    # unknowns never rejected -> TUR = 0 -> OS-HM = 0.
    df = _frame(["a", "z", "z"], ["a", "a", "a"])
    m = open_set_session_metrics(df, base_classes={"a"}, admitted_novel=set())
    assert m["tur"] == 0.0
    assert m["os_hm"] == 0.0


def test_scorer_admitted_novel_counts_as_known():
    df = _frame(["a", "c"], ["a", "c"])
    m = open_set_session_metrics(df, base_classes={"a"}, admitted_novel={"c"})
    assert m["n_true_known"] == 2
    assert m["ccr"] == 1.0


def test_scorer_harmonic_mean_formula():
    # CCR=0.5 (1/2 knowns right), TUR=1.0 -> HM = 2*.5*1/(1.5) = 0.6667
    df = _frame(["a", "a", "z"], ["a", "unknown", "unknown"])
    m = open_set_session_metrics(df, base_classes={"a"}, admitted_novel=set())
    assert m["ccr"] == 0.5
    assert m["tur"] == 1.0
    assert m["os_hm"] == pytest.approx(0.6667, abs=1e-3)


# --------------------------------------------------------------------------- #
# Parity (skipped unless the bundled dataset is present). Anchors against the  #
# canonical reference numbers for N-BaIoT fixed seed 42.                          #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not dataset_available("nbaiot"), reason="nbaiot not bundled")
def test_parity_nbaiot_fixed_s42():
    from gons_run import run_gons_cfg

    cfg = make_gons_cfg("nbaiot", 42, tag="parity_nbaiot_s42")
    r = run_gons_cfg(cfg, tag="parity_nbaiot_s42")
    assert r["status"] == "ok"
    # Reference cell: os_hm=0.6565 ccr=0.6197 tur=0.7191.
    assert r["os_hm"] == pytest.approx(0.6565, abs=2e-3)
    assert r["ccr"] == pytest.approx(0.6197, abs=2e-3)
    assert r["tur"] == pytest.approx(0.7191, abs=2e-3)
