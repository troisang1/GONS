"""CICIoT2023 dataset loading helpers.

CICIoT2023 contains 33 individual attack types in 7 categories, plus benign
traffic, across 105 IoT devices.  Each record has 47 flow-level features.

Reference:
    Neto et al., "CICIoT2023: A Real-Time Dataset and Benchmark for
    Large-Scale Attacks in IoT Environment", Sensors 2023.

The dataset ships as multiple CSV files (one per attack category).  This
loader merges them into a single frame, normalizes labels, and optionally
applies category-level or fine-grained labeling.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from bcmrnfst.data.ton_iot import (
    DatasetMetadata,
    LoadedTabularDataset,
    detect_label_column,
    extract_dataset_metadata,
)
from bcmrnfst.preprocess.schema import infer_feature_groups
from bcmrnfst.runtime import find_repo_root


# Category-level mapping: 33 fine-grained attacks -> 7 categories + Benign
ATTACK_CATEGORY_MAP: dict[str, str] = {
    # DDoS (normalized: hyphens become underscores)
    "ddos_ack_fragmentation": "ddos",
    "ddos_udp_flood": "ddos",
    "ddos_slowloris": "ddos",
    "ddos_icmp_flood": "ddos",
    "ddos_rstfinflood": "ddos",
    "ddos_pshack_flood": "ddos",
    "ddos_http_flood": "ddos",
    "ddos_udp_fragmentation": "ddos",
    "ddos_icmp_fragmentation": "ddos",
    "ddos_syn_flood": "ddos",
    "ddos_synonymousip_flood": "ddos",
    "ddos_tcp_flood": "ddos",
    # DoS
    "dos_udp_flood": "dos",
    "dos_syn_flood": "dos",
    "dos_tcp_flood": "dos",
    "dos_http_flood": "dos",
    # Mirai
    "mirai_greeth_flood": "mirai",
    "mirai_greip_flood": "mirai",
    "mirai_udpplain": "mirai",
    # Recon
    "recon_pingsweep": "recon",
    "recon_osscan": "recon",
    "recon_hostdiscovery": "recon",
    "recon_portscan": "recon",
    # Spoofing
    "dns_spoofing": "spoofing",
    "mitm_arpspoofing": "spoofing",
    # Web-based
    "dictionarybruteforce": "web",
    "browserhijacking": "web",
    "commandinjection": "web",
    "sqlinjection": "web",
    "uploading_attack": "web",
    "xss": "web",
    "backdoor_malware": "web",
    "vulnerabilityscan": "web",
    # Benign
    "benigntraffic": "benign",
}


def _normalize_ciciot_label(value: object) -> str:
    """Normalize raw CICIoT2023 label values."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(value).strip().lower()).strip("_") or "unknown"


def load_ciciot2023_csvs(
    data_dir: Path,
    *,
    label_granularity: Literal["category", "fine"] = "category",
    max_files: int | None = None,
) -> pd.DataFrame:
    """Load all CICIoT2023 CSV files from *data_dir* into a single frame.

    Parameters
    ----------
    data_dir:
        Directory containing the CICIoT2023 CSV files.
    label_granularity:
        ``"category"`` maps the 33 attacks to 8 categories (7 attack + benign).
        ``"fine"`` preserves the original 34 labels.
    max_files:
        If set, limit the number of CSV files loaded (for quick testing).
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    if max_files is not None:
        csv_files = csv_files[:max_files]

    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding="utf-8-sig", low_memory=False)
        # Normalize column names
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Detect label column (CICIoT2023 uses "label")
    label_candidates = ["label", "attack_type", "class", "type"]
    label_col = detect_label_column(combined, label_candidates)

    # Normalize labels
    combined["label"] = combined[label_col].map(_normalize_ciciot_label)

    # Apply category mapping if requested
    if label_granularity == "category":
        combined["label"] = combined["label"].map(
            lambda x: ATTACK_CATEGORY_MAP.get(x, x)
        )

    # Drop the original label column if different from "label"
    if label_col != "label" and label_col in combined.columns:
        combined = combined.drop(columns=[label_col])

    # Drop any identifier-like columns
    drop_cols = [c for c in combined.columns if c in ("src_ip", "dst_ip", "src_port", "dst_port", "flow_id", "timestamp")]
    if drop_cols:
        combined = combined.drop(columns=drop_cols, errors="ignore")

    return combined


def build_balanced_ciciot2023_subset(
    *,
    source_dir: Path,
    output_dir: Path,
    samples_per_class: int,
    seed: int = 42,
    label_granularity: Literal["category", "fine"] = "category",
    train_ratio: float = 0.6,
    calibration_ratio: float = 0.2,
    max_source_files: int | None = None,
) -> dict[str, Path]:
    """Build balanced train/calibration/test CSV splits from CICIoT2023.

    Returns a dict with keys ``train_path``, ``calibration_path``, ``test_path``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and merge
    combined = load_ciciot2023_csvs(
        source_dir,
        label_granularity=label_granularity,
        max_files=max_source_files,
    )

    # Remove non-numeric, non-label columns that might cause issues
    # Keep only numeric features + label
    non_numeric = []
    for col in combined.columns:
        if col == "label":
            continue
        if not pd.api.types.is_numeric_dtype(combined[col]):
            non_numeric.append(col)
    if non_numeric:
        combined = combined.drop(columns=non_numeric)

    # Replace inf/nan
    combined = combined.replace([float("inf"), float("-inf")], float("nan"))
    combined = combined.dropna()

    # Sample balanced subset
    label_counts = combined["label"].value_counts()
    rng = pd.np if hasattr(pd, "np") else __import__("numpy").random
    import numpy as np

    rng_gen = np.random.default_rng(seed)

    train_frames, cal_frames, test_frames = [], [], []
    for label in sorted(label_counts.index):
        class_df = combined[combined["label"] == label]
        available = len(class_df)
        n_select = min(samples_per_class, available)

        selected_idx = rng_gen.choice(class_df.index, size=n_select, replace=False)
        selected = class_df.loc[selected_idx]

        # Split
        test_ratio = 1.0 - train_ratio - calibration_ratio
        n_train = max(1, int(n_select * train_ratio))
        n_cal = max(1, int(n_select * calibration_ratio))
        n_test = n_select - n_train - n_cal

        shuffled = selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        train_frames.append(shuffled.iloc[:n_train])
        cal_frames.append(shuffled.iloc[n_train : n_train + n_cal])
        if n_test > 0:
            test_frames.append(shuffled.iloc[n_train + n_cal :])

    train_df = pd.concat(train_frames, ignore_index=True)
    cal_df = pd.concat(cal_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()

    train_path = output_dir / "train.csv"
    cal_path = output_dir / "calibration.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    cal_df.to_csv(cal_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"CICIoT2023 subset created:")
    print(f"  Train: {len(train_df)} rows -> {train_path}")
    print(f"  Calibration: {len(cal_df)} rows -> {cal_path}")
    print(f"  Test: {len(test_df)} rows -> {test_path}")
    print(f"  Classes: {sorted(train_df['label'].unique())}")

    return {
        "train_path": train_path,
        "calibration_path": cal_path,
        "test_path": test_path,
    }
