#!/usr/bin/env python
"""Regenerate the human-readable YAML config records under configs/.

`gons_configs.py` is the authoritative source; the YAML files exist so a reader
can see the exact configuration without running Python. This script rewrites them
from that source, so the two can never drift.

Writes:
    configs/fixed/gons_fixed.yaml       the ONE global FIXED config. Values that
                                        vary per dataset (paths, base_class_ratio)
                                        and the bandwidth resolved per (dataset,
                                        seed) are shown as placeholders, because
                                        the config itself is dataset-independent.
    configs/tuned/<dataset>.yaml        the per-dataset TUNED config, one per
                                        dataset in DATASETS, at seed 42.

Usage:
    python export_configs.py
    python export_configs.py --check     # exit 1 if any file is out of date
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from gons_configs import (
    DATASETS,
    FIXED_GAMMA_RULE,
    GONS_TUNED_PER_DS,
    MAP_GAMMA_UNRESOLVED,
    make_gons_cfg,
    make_gons_tuned_cfg,
)

REPO = Path(__file__).resolve().parent
EXPORT_SEED = 42

FIXED_HEADER = f"""\
# GENERATED record (do not hand-edit). Regenerate with: python export_configs.py
# Source: gons_configs.py -> make_gons_cfg.
#
# ONE fixed configuration, shared by every dataset. `map_gamma` is the uniform
# {FIXED_GAMMA_RULE} bandwidth RULE, evaluated on each dataset's own base-session
# training split, so the only things that vary across datasets are the data paths
# and the base-class ratio.
#
# Reference value for verifying a re-implementation of the rule:
#   N-BaIoT seed 42 -> map_gamma = 0.0173399338 -> OS-HM 0.6565
"""


def _tuned_header(ds: str) -> str:
    t = GONS_TUNED_PER_DS[ds]
    return f"""\
# GENERATED record (do not hand-edit). Regenerate with: python export_configs.py
# Source: gons_configs.py -> make_gons_tuned_cfg (per-dataset TUNED grid cell).
#
# The per-dataset-TUNED comparison arm for {ds}, at seed {EXPORT_SEED}. Reproduce with:
#   python run_gons_main.py --arm tuned --datasets {ds}
#
# This is NOT the ablation base -- the shipped ablation is on the FIXED config.
# `map_gamma` here is the '{t["gamma_rule"]}' bandwidth rule (the tuned arm selects the
# rule as well as the grid cell), resolved on this dataset+seed's base split.
"""


def fixed_record() -> dict:
    """The FIXED config with the dataset-varying fields shown as placeholders."""
    ds0 = next(iter(DATASETS))
    cfg = make_gons_cfg(ds0, EXPORT_SEED, tag=f"fixed_<dataset>_s{EXPORT_SEED}")
    cfg["train_path"] = "<DATASETS[dataset]['train']>"
    cfg["calibration_path"] = "<DATASETS[dataset]['cal']>"
    cfg["test_path"] = "<DATASETS[dataset]['test']>"
    cfg["base_class_ratio"] = "<DATASETS[dataset]['ratio']>"
    cfg["model"]["map_gamma"] = (
        f"<resolved per (dataset, seed) by gons_configs.resolve_map_gamma "
        f"-- {FIXED_GAMMA_RULE} rule>"
    )
    return cfg


def tuned_record(ds: str) -> dict:
    cfg = make_gons_tuned_cfg(ds, EXPORT_SEED, tag=f"tuned_{ds}_s{EXPORT_SEED}")
    if cfg["model"]["map_gamma"] == MAP_GAMMA_UNRESOLVED:
        # Splits not bundled: record the rule rather than the sentinel, which
        # would read as a real bandwidth.
        cfg["model"]["map_gamma"] = (
            f"<resolved per (dataset, seed) by gons_configs.resolve_map_gamma "
            f"-- {GONS_TUNED_PER_DS[ds]['gamma_rule']} rule>"
        )
    return cfg


def _dump(header: str, record: dict) -> str:
    return header + yaml.safe_dump(record, sort_keys=False, default_flow_style=False)


def targets() -> dict[Path, str]:
    out = {REPO / "configs" / "fixed" / "gons_fixed.yaml": _dump(FIXED_HEADER, fixed_record())}
    for ds in DATASETS:
        out[REPO / "configs" / "tuned" / f"{ds}.yaml"] = _dump(_tuned_header(ds), tuned_record(ds))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="Do not write; exit 1 if any record is out of date.")
    a = ap.parse_args()

    stale = []
    for path, text in targets().items():
        current = path.read_text() if path.exists() else None
        if current == text:
            continue
        stale.append(path)
        if not a.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    rel = [str(p.relative_to(REPO)) for p in stale]
    if a.check:
        if stale:
            print("OUT OF DATE (run `python export_configs.py`):")
            for r in rel:
                print(f"  {r}")
            return 1
        print("configs/ is up to date with gons_configs.py")
        return 0
    print(f"rewrote {len(stale)} record(s)" + (": " + ", ".join(rel) if rel else "; all were current"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
