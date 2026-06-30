"""Prepare a dataset into the processed train / calibration / test splits GONS reads.

Turns a single labeled tabular CSV into the exact split format under
`data/processed/<name>/` that `run_gons_main.py` / `run_gons_ablation.py` consume
(see `data/README.md`). This is the reproducible recipe behind the bundled
datasets, as a turnkey script.

Recipe (deterministic, seed-fixed):
  1. Normalize the label column to lowercase, spaces/hyphens -> "_".
  2. Per class (sorted): take min(cap, available) rows, sampled without
     replacement at the fixed seed, then shuffled at the fixed seed.
  3. Split each class 60 / 20 / 20 into train / calibration / test.
  4. Write train.csv, calibration.csv, test.csv (label column + numeric features).

Caps used for the paper's datasets: 10000/class for ToN-IoT; 5000/class for the
other IDS datasets (N-BaIoT, CICIDS2018, 5G-NIDD, NSL-KDD). Pass --cap to match.

Usage:
  python prepare_dataset.py --input raw/nbaiot.csv --name nbaiot_5k --cap 5000
  python prepare_dataset.py --input raw/toniot.csv --name toniot --cap 10000 --label-col label

The output dir (data/processed/<name>/) must match the paths in `gons_configs.py`
(the DATASETS table) for the runner to pick the dataset up automatically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SEED = 42
TRAIN_FRAC, CAL_FRAC = 0.6, 0.2  # remainder (0.2) is test


def normalize_label(s: object) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


def balanced_split(
    df: pd.DataFrame, label_col: str, cap: int, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-class balanced subsample + deterministic 60/20/20 train/cal/test split."""
    train_parts, cal_parts, test_parts = [], [], []
    for _, group in df.groupby(label_col, sort=True):
        n = min(cap, len(group))
        sampled = group.sample(n=n, random_state=seed, replace=False)
        shuffled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_tr = int(n * TRAIN_FRAC)
        n_ca = int(n * CAL_FRAC)
        train_parts.append(shuffled.iloc[:n_tr])
        cal_parts.append(shuffled.iloc[n_tr : n_tr + n_ca])
        test_parts.append(shuffled.iloc[n_tr + n_ca :])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(cal_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="Raw labeled CSV (one label column + numeric features).")
    ap.add_argument("--name", required=True, help="Dataset name -> writes data/processed/<name>/.")
    ap.add_argument("--cap", type=int, default=10000, help="Per-class row cap (10000 ToN-IoT, 5000 other IDS).")
    ap.add_argument("--label-col", default="label", help="Label column name in the input CSV.")
    ap.add_argument("--seed", type=int, default=SEED, help="Fixed seed for subsample + shuffle (default 42).")
    ap.add_argument("--out-root", type=Path, default=Path("data/processed"), help="Output root.")
    args = ap.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    if args.label_col not in df.columns:
        raise SystemExit(f"label column '{args.label_col}' not in {args.input} (columns: {list(df.columns)[:8]}...)")
    df = df.rename(columns={args.label_col: "label"})
    df["label"] = df["label"].map(normalize_label)

    train, cal, test = balanced_split(df, label_col="label", cap=args.cap, seed=args.seed)

    out_dir = args.out_root / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    cal.to_csv(out_dir / "calibration.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    n_classes = df["label"].nunique()
    print(
        f"wrote {out_dir}/ : train={len(train)} calibration={len(cal)} test={len(test)} "
        f"({n_classes} classes, cap={args.cap}, seed={args.seed}, split=60/20/20)"
    )


if __name__ == "__main__":
    main()
