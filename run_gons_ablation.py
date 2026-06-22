#!/usr/bin/env python
"""GONS 4-component ablation (GONS gate-OFF, per-dataset TUNED config).

Ablates the GONS FULL (per-dataset tuned, min-distance scoring, refresh ON,
capacity 512) config one component at a time across seeds 42..51, scoring each
run through the official OS-HM scorer. Prints the signed Delta OS-HM of each arm
vs FULL (positive = the removed component helps). Resumable.

Arms:
    full         — the deployed GONS config (baseline).
    no_refresh   — disable the between-session consolidated refresh.
    no_quantile  — disable quantile (NCDR) classification.
    scoring      — min-distance -> energy scoring.
    cap128       — RFF capacity 512 -> 128.

Usage:
    python run_gons_ablation.py
    python run_gons_ablation.py --datasets nbaiot --seeds 42 43
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

from gons_configs import DATASETS, dataset_available, make_gons_tuned_cfg
from gons_run import run_gons_cfg

REPO = Path(__file__).resolve().parent
OUT = REPO / "artifacts" / "reports" / "gons_ablation.jsonl"
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

ARMS = ["full", "no_refresh", "no_quantile", "scoring", "cap128"]
ARM_OVERRIDES: dict[str, dict] = {
    "full": {},
    "no_refresh": {"refresh_after_session": False, "refresh_mode": "single_pass"},
    "no_quantile": {"enable_ncdr_classification": False},
    "scoring": {"scoring_mode": "energy"},
    "cap128": {"map_n_components": 128},
}


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
    a = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(OUT)

    available = [d for d in a.datasets if dataset_available(d)]
    skipped = [d for d in a.datasets if not dataset_available(d)]
    if skipped:
        print(f"[skip] datasets not found under data/processed/: {skipped}")
    print(f"GONS ablation: datasets={available} arms={ARMS} seeds={a.seeds} (gate-OFF)\n")

    rows: list[dict] = []
    for ds in available:
        for arm in ARMS:
            for seed in a.seeds:
                tag = f"abl_{ds}_{arm}_s{seed}"
                if tag in done:
                    print(f"  {tag}  SKIP (cached os_hm={done[tag].get('os_hm')})")
                    rows.append(done[tag])
                    continue
                cfg = make_gons_tuned_cfg(ds, seed, tag=tag, overrides=dict(ARM_OVERRIDES[arm]))
                r = run_gons_cfg(cfg, tag=tag)
                r.update({"dataset": ds, "arm": arm, "seed": seed, "method": "gons_ablation"})
                with open(OUT, "a") as f:
                    f.write(json.dumps(r) + "\n")
                rows.append(r)
                print(f"  {tag}  os_hm={r.get('os_hm')} t={r.get('elapsed')}s [{r['status']}]")

    _summary(rows, available)
    print(f"\nWrote {OUT}")
    return 0


def _arm_mean(rows: list[dict], ds: str, arm: str) -> float | None:
    vals = [r["os_hm"] for r in rows
            if r.get("dataset") == ds and r.get("arm") == arm and r.get("os_hm") is not None]
    return mean(vals) if vals else None


def _summary(rows: list[dict], datasets: list[str]) -> None:
    print(f"\n{'='*70}")
    print("GONS ABLATION — Delta OS-HM vs FULL (positive = component helps), gate-OFF")
    print(f"{'='*70}")
    # per-dataset table
    print(f"{'Dataset':<12} {'FULL':>8}  " + "  ".join(f"{a:>12}" for a in ARMS[1:]))
    print("-" * 70)
    overall: dict[str, list[float]] = {a: [] for a in ARMS}
    for ds in datasets:
        full = _arm_mean(rows, ds, "full")
        if full is None:
            continue
        overall["full"].append(full)
        cells = []
        for arm in ARMS[1:]:
            v = _arm_mean(rows, ds, arm)
            if v is None:
                cells.append("n/a")
                continue
            overall[arm].append(v)
            cells.append(f"+{full - v:.4f}")
        print(f"{ds:<12} {full:>8.4f}  " + "  ".join(f"{c:>12}" for c in cells))
    print("-" * 70)
    if overall["full"]:
        full_m = mean(overall["full"])
        print(f"{'OVERALL':<12} {full_m:>8.4f}  " +
              "  ".join(f"{('+%.4f' % (full_m - mean(overall[a]))) if overall[a] else 'n/a':>12}"
                       for a in ARMS[1:]))
        print("\n(component contribution = FULL - arm, averaged over datasets)")


if __name__ == "__main__":
    raise SystemExit(main())
