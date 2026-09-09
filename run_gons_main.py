#!/usr/bin/env python
"""GONS main result — 7 datasets x 10 seeds, OS-HM.

Runs GONS on every available dataset across seeds 42..51, scores each run through
the official OS-HM scorer, and prints a per-dataset mean +/- std table. Resumable:
rows already in the output jsonl are skipped.

Two arms:
    --arm fixed  (default)  the ONE global config, identical on every dataset.
                            This is the headline: results/oshm_matrix_7ds.csv.
    --arm tuned             each dataset's own argmax config AND bandwidth rule.
                            The per-dataset-tuned comparison arm; together with
                            `fixed` it reproduces both columns of
                            results/headline_fixed_vs_tuned_7ds.csv.

Usage:
    python run_gons_main.py
    python run_gons_main.py --arm tuned
    python run_gons_main.py --datasets nbaiot nsl_kdd --seeds 42 43
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

from gons_configs import DATASETS, dataset_available, make_gons_cfg, make_gons_tuned_cfg
from gons_run import run_gons_cfg

REPO = Path(__file__).resolve().parent
OUT = REPO / "artifacts" / "reports" / "gons_main.jsonl"
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
METRIC_COLS = ["os_hm", "ccr", "tur", "f1_unknown", "macro_f1_with_unknown"]


def load_done(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("status") == "ok":
                    done[r["tag"]] = r
            except Exception:
                pass
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--arm", choices=["fixed", "tuned"], default="fixed",
                    help="fixed = the one global config (headline); "
                         "tuned = each dataset's own argmax config + bandwidth rule.")
    a = ap.parse_args()
    make_cfg = make_gons_cfg if a.arm == "fixed" else make_gons_tuned_cfg

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(OUT)

    available = [d for d in a.datasets if dataset_available(d)]
    skipped = [d for d in a.datasets if not dataset_available(d)]
    if skipped:
        print(f"[skip] datasets not found under data/processed/: {skipped}")
        print("       (see data/README.md to prepare the five unbundled datasets)")
    print(f"GONS main result [{a.arm}]: datasets={available} seeds={a.seeds}\n")

    rows: list[dict] = []
    for ds in available:
        for seed in a.seeds:
            tag = f"gons_{ds}_{a.arm}_s{seed}"
            if tag in done:
                print(f"  {tag}  SKIP (cached os_hm={done[tag].get('os_hm')})")
                rows.append(done[tag])
                continue
            cfg = make_cfg(ds, seed, tag=tag)
            r = run_gons_cfg(cfg, tag=tag)
            r.update({"dataset": ds, "seed": seed, "method": f"gons_{a.arm}"})
            with open(OUT, "a") as f:
                f.write(json.dumps(r) + "\n")
            rows.append(r)
            print(f"  {tag}  os_hm={r.get('os_hm')} t={r.get('elapsed')}s [{r['status']}]")

    _summary(rows, available, a.arm)
    print(f"\nWrote {OUT}")
    return 0


def _summary(rows: list[dict], datasets: list[str], arm: str = "fixed") -> None:
    print(f"\n{'='*78}")
    print(f"GONS MAIN RESULT [{arm}] — OS-HM (mean +/- std over seeds)")
    print(f"{'='*78}")
    print(f"{'Dataset':<12} {'n':>3}  " + "  ".join(f"{c:>16}" for c in METRIC_COLS))
    print("-" * 78)
    ds_means: list[float] = []
    for ds in datasets:
        vals = {c: [r[c] for r in rows if r.get("dataset") == ds and r.get(c) is not None]
                for c in METRIC_COLS}
        if not vals["os_hm"]:
            continue
        n = len(vals["os_hm"])
        cells = []
        for c in METRIC_COLS:
            v = vals[c]
            m = mean(v)
            s = pstdev(v) if len(v) > 1 else 0.0
            cells.append(f"{m:.3f}+/-{s:.3f}")
        ds_means.append(mean(vals["os_hm"]))
        print(f"{ds:<12} {n:>3}  " + "  ".join(f"{c:>16}" for c in cells))
    print("-" * 78)
    if ds_means:
        print(f"{'MEAN OS-HM':<12}      {mean(ds_means):.4f}  (across {len(ds_means)} datasets)")


if __name__ == "__main__":
    raise SystemExit(main())
