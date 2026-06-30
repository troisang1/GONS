"""Official OS-HM scorer for the GONS release (single source of truth).

`open_set_session_metrics` is the *one* scoring entrypoint. The main-result
runner, the ablation runner, and the smoke test all score through this exact
function — there is no second metric implementation in this repo. The reproduced
numbers come from here and nowhere else.

Open-set protocol (per session):
  - A test row whose `true_label` is NOT in (base_classes ∪ admitted_novel) is a
    "true Unknown" and should be predicted as 'unknown' for full credit.
  - CCR  = correct-classification rate over true-known rows.
  - TUR  = true-unknown-rejection rate over true-unknown rows.
  - OS-HM = harmonic mean of CCR and TUR (the headline metric).

Aggregation (per run dir): mean OS-HM over the open-set sessions only
(sessions with at least one true-unknown row; the final all-admitted session is
excluded). Multi-seed aggregation (mean / std over seeds) is done by the runner.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["os_hm", "ccr", "tur", "f1_unknown", "macro_f1_with_unknown"]


def open_set_session_metrics(
    df: pd.DataFrame, base_classes: set[str], admitted_novel: set[str]
) -> dict[str, float]:
    """Compute per-session open-set metrics from a session predictions frame."""

    known_classes = base_classes | admitted_novel
    true_labels = df["true_label"].astype(str).str.lower()
    pred_labels = df["predicted_label"].astype(str).str.lower()

    true_known = true_labels.isin(known_classes)
    pred_unknown = pred_labels == "unknown"
    pred_correct_class = pred_labels == true_labels

    n = len(df)
    n_true_known = int(true_known.sum())
    n_true_unknown = int((~true_known).sum())

    # CCR: among true knowns, fraction predicted to the correct known class
    ccr = float(pred_correct_class[true_known].mean()) if n_true_known else 0.0
    # TUR: among true unknowns, fraction correctly rejected
    tur = float(pred_unknown[~true_known].mean()) if n_true_unknown else 0.0
    # Open-set HM (headline metric)
    os_hm = (2 * ccr * tur / (ccr + tur)) if (ccr + tur) > 0 else 0.0

    # F1-Unknown (treating 'unknown' as a class label)
    tp_unk = int((pred_unknown & ~true_known).sum())
    fp_unk = int((pred_unknown & true_known).sum())  # false reject of a known
    fn_unk = int((~pred_unknown & ~true_known).sum())  # absorbed an unknown
    prec_unk = tp_unk / (tp_unk + fp_unk) if (tp_unk + fp_unk) else 0.0
    rec_unk = tp_unk / (tp_unk + fn_unk) if (tp_unk + fn_unk) else 0.0
    f1_unk = 2 * prec_unk * rec_unk / (prec_unk + rec_unk) if (prec_unk + rec_unk) else 0.0

    # Macro F1 over (known classes + Unknown)
    labels = sorted(known_classes) + ["unknown"]
    f1_per_class = []
    y_true_collapsed = true_labels.where(true_known, other="unknown")
    for c in labels:
        tp = int(((pred_labels == c) & (y_true_collapsed == c)).sum())
        fp = int(((pred_labels == c) & (y_true_collapsed != c)).sum())
        fn = int(((pred_labels != c) & (y_true_collapsed == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1_per_class.append(f1)
    macro_f1 = sum(f1_per_class) / len(f1_per_class)

    return {
        "n_samples": n,
        "n_true_known": n_true_known,
        "n_true_unknown": n_true_unknown,
        "ccr": round(ccr, 4),
        "tur": round(tur, 4),
        "os_hm": round(os_hm, 4),
        "f1_unknown": round(f1_unk, 4),
        "macro_f1_with_unknown": round(macro_f1, 4),
    }


def full_metrics_from_run(run_dir: Path) -> dict[str, float] | None:
    """Aggregate the canonical OS-HM (+ companions) over a run dir's sessions.

    Reads `summary_overview.json` for base/novel classes, then averages each
    metric over the open-set sessions (those with >0 true-unknown rows).
    Returns None if the run produced no scorable open-set sessions.
    """

    sj = run_dir / "summary_overview.json"
    if not sj.exists():
        return None
    s = json.loads(sj.read_text())
    base = set(map(str.lower, s.get("base_classes") or []))
    novel = list(map(str.lower, s.get("novel_classes") or []))
    sessions_dir = run_dir / "sessions"
    if not sessions_dir.exists():
        return None
    agg: dict[str, list[float]] = {k: [] for k in METRICS}
    for sd in sorted(sessions_dir.iterdir()):
        m = re.match(r"session_(\d+)_", sd.name)
        if not m:
            continue
        sid = int(m.group(1))
        pred = sd / "predictions.csv"
        if not pred.exists():
            continue
        met = open_set_session_metrics(pd.read_csv(pred), base, set(novel[:sid]))
        if met["n_true_unknown"] > 0:
            for k in METRICS:
                agg[k].append(met[k])
    if not agg["os_hm"]:
        return None
    return {k: round(float(np.mean(v)), 4) for k, v in agg.items()}


__all__ = ["METRICS", "full_metrics_from_run", "open_set_session_metrics"]
