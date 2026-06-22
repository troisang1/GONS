#!/usr/bin/env python
"""GONS smoke test — fast single (dataset, seed) end-to-end fit + OS-HM assert.

Runs GONS (gate-OFF) on N-BaIoT seed 42 through the full FSCIL
protocol and asserts a valid OS-HM in the expected neighborhood (~0.77).
If this passes, the import closure, data wiring, and official scorer all work.

Usage:
    python smoke_test.py
"""

from __future__ import annotations

import sys

from gons_configs import dataset_available, make_gons_cfg
from gons_run import run_gons_cfg

DATASET = "nbaiot"
SEED = 42
# generous band: the headline expectation is ~0.77; we just assert "valid + sane".
OS_HM_LOW, OS_HM_HIGH = 0.55, 0.95


def main() -> int:
    if not dataset_available(DATASET):
        print(f"SMOKE FAIL: bundled dataset '{DATASET}' not found under data/processed/.")
        return 2

    print(f"GONS smoke test: dataset={DATASET} seed={SEED} (gate-OFF)")
    cfg = make_gons_cfg(DATASET, SEED, tag=f"smoke_{DATASET}_s{SEED}")
    r = run_gons_cfg(cfg, tag=f"smoke_{DATASET}_s{SEED}")

    print(f"  status={r['status']} elapsed={r.get('elapsed')}s gate_active={r['gate_active']}")
    if r["status"] != "ok":
        print(f"SMOKE FAIL: run did not produce metrics: {r.get('err', r['status'])}")
        return 1

    os_hm = r["os_hm"]
    print(f"  OS-HM={os_hm}  CCR={r['ccr']}  TUR={r['tur']}  "
          f"F1-Unk={r['f1_unknown']}  MacroF1+U={r['macro_f1_with_unknown']}")

    if r["gate_active"]:
        print("SMOKE FAIL: gate_active=True — GONS must be gate-OFF.")
        return 1
    if not (OS_HM_LOW <= os_hm <= OS_HM_HIGH):
        print(f"SMOKE FAIL: OS-HM {os_hm} outside sane band [{OS_HM_LOW}, {OS_HM_HIGH}].")
        return 1

    print("SMOKE PASS: valid OS-HM, gate-OFF.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
