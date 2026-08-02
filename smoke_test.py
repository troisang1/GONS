#!/usr/bin/env python
"""GONS smoke test — fast single (dataset, seed) end-to-end fit + OS-HM assert.

Runs GONS on N-BaIoT seed 42 through the full FSCIL protocol and asserts the
reference OS-HM of 0.6565. If this passes, the import closure, data wiring,
bandwidth rule, eigensolver path and official scorer are all correct.

Usage:
    python smoke_test.py
"""

from __future__ import annotations

import sys

from gons_configs import dataset_available, make_gons_cfg
from gons_run import run_gons_cfg

DATASET = "nbaiot"
SEED = 42
# Tight band around the reference value: a wide "is it sane" range would accept a
# run whose bandwidth or eigensolver path is wrong.
OS_HM_REF = 0.6565
OS_HM_TOL = 0.002
OS_HM_LOW, OS_HM_HIGH = OS_HM_REF - OS_HM_TOL, OS_HM_REF + OS_HM_TOL


def main() -> int:
    if not dataset_available(DATASET):
        print(f"SMOKE FAIL: bundled dataset '{DATASET}' not found under data/processed/.")
        return 2

    print(f"GONS smoke test: dataset={DATASET} seed={SEED}")
    cfg = make_gons_cfg(DATASET, SEED, tag=f"smoke_{DATASET}_s{SEED}")
    r = run_gons_cfg(cfg, tag=f"smoke_{DATASET}_s{SEED}")

    print(f"  status={r['status']} elapsed={r.get('elapsed')}s")
    if r["status"] != "ok":
        print(f"SMOKE FAIL: run did not produce metrics: {r.get('err', r['status'])}")
        return 1

    os_hm = r["os_hm"]
    print(f"  OS-HM={os_hm}  CCR={r['ccr']}  TUR={r['tur']}  "
          f"F1-Unk={r['f1_unknown']}  MacroF1+U={r['macro_f1_with_unknown']}")

    if not (OS_HM_LOW <= os_hm <= OS_HM_HIGH):
        print(f"SMOKE FAIL: OS-HM {os_hm} outside sane band [{OS_HM_LOW}, {OS_HM_HIGH}].")
        return 1

    print("SMOKE PASS: valid OS-HM.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
