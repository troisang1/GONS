"""Configuration models for baseline runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

import yaml
from pydantic import BaseModel, Field, PositiveInt, field_validator

from bcmrnfst.config.data import ToNIoTDatasetConfig, load_ton_iot_config
from bcmrnfst.config.model import BCMRNFSTConfig
from bcmrnfst.runtime import find_repo_root

BaselineKind: TypeAlias = Literal[
    "inn_scc_nnfst",
    "batch_knfst",
    "incremental_kernel_nfst",
    "pca_nfst",
    "openmax",
    "ewc_ncm",
    "msp",
    "doc",
    "cnd_ids",
    "bic",
    "rfs",
    "topic",
    "nc_fscil",
]
KernelMetric: TypeAlias = Literal["linear", "rbf", "poly", "sigmoid", "cosine"]
UpdateMode: TypeAlias = Literal["full_retrain", "incremental"]
DEFAULT_KERNEL_EXACT_MAX_TRAIN_ROWS = 1_000_000_000


class BaselineExperimentConfig(BaseModel):
    """Typed baseline experiment configuration loaded from YAML."""

    experiment_name: str
    method: BaselineKind
    dataset_config_path: Path = Path("configs/data/ton_iot.yaml")
    train_path: Path
    calibration_path: Path
    test_path: Path | None = None
    output_root: Path = Path("artifacts/runs")

    kernel: KernelMetric = "rbf"
    kernel_gamma: float = 1.0
    kernel_degree: PositiveInt = 3
    kernel_coef0: float = 1.0
    kernel_exact_max_train_rows: PositiveInt = DEFAULT_KERNEL_EXACT_MAX_TRAIN_ROWS
    # Optional class-stratified landmark cap for incremental_kernel_nfst (IKNDA). When
    # set, EVERY exact-kernel fit (incl. datasets with rows <= the error-gate) uses a
    # bounded landmark support of this many rows, so each incremental KNFST fit is
    # tractable+fast (O(cap^2) Gram + O(cap^3) eigh) regardless of train size. Decoupled
    # from kernel_exact_max_train_rows (the SIGSEGV/OOM error-gate). None preserves prior
    # behavior (landmark cap == error-gate). Only consulted for incremental_kernel_nfst.
    kernel_landmark_cap: PositiveInt | None = None

    pca_components: PositiveInt | None = None

    inn_neighbor_k: PositiveInt = 5
    inn_center_separation_scale: float = 1.0

    enable_dbscan_discovery: bool = True
    dbscan_eps: float | None = None
    dbscan_min_samples: PositiveInt = 3

    openmax_hidden_dim: PositiveInt = 128
    openmax_hidden_layers: PositiveInt = 2
    openmax_epochs: PositiveInt = 50
    openmax_lr: float = 0.001
    openmax_batch_size: PositiveInt = 256
    openmax_tail_size: PositiveInt = 20
    openmax_alpha_rank: PositiveInt = 3
    openmax_threshold: float = 0.5

    ewc_hidden_dim: PositiveInt = 128
    ewc_hidden_layers: PositiveInt = 2
    ewc_epochs: PositiveInt = 100
    ewc_lr: float = 0.001
    ewc_batch_size: PositiveInt = 256
    ewc_lambda: float = 1000.0
    ewc_ncm_threshold_quantile: float = 0.95

    msp_hidden_dim: PositiveInt = 128
    msp_hidden_layers: PositiveInt = 2
    msp_epochs: PositiveInt = 100
    msp_lr: float = 0.001
    msp_batch_size: PositiveInt = 256
    msp_threshold: float = 0.5

    doc_hidden_dim: PositiveInt = 128
    doc_hidden_layers: PositiveInt = 2
    doc_epochs: PositiveInt = 100
    doc_lr: float = 0.001
    doc_batch_size: PositiveInt = 256
    doc_sigma_factor: float = 3.0

    cnd_hidden_dim: PositiveInt = 128
    cnd_hidden_layers: PositiveInt = 2
    cnd_feature_dim: PositiveInt = 64
    cnd_epochs: PositiveInt = 50
    cnd_batch_size: PositiveInt = 256
    cnd_pca_components: PositiveInt = 32
    cnd_ewc_lambda: float = 100.0
    cnd_incremental_epochs: PositiveInt = 10

    bic_hidden_dim: PositiveInt = 128
    bic_hidden_layers: PositiveInt = 2
    bic_epochs: PositiveInt = 100
    bic_batch_size: PositiveInt = 256
    bic_incremental_epochs: PositiveInt = 20

    rfs_hidden_dim: PositiveInt = 128
    rfs_hidden_layers: PositiveInt = 2
    rfs_epochs: PositiveInt = 100
    rfs_batch_size: PositiveInt = 256

    topic_hidden_dim: PositiveInt = 128
    topic_hidden_layers: PositiveInt = 2
    topic_feature_dim: PositiveInt = 64
    topic_epochs: PositiveInt = 100
    topic_batch_size: PositiveInt = 256
    topic_n_anchors: PositiveInt = 100
    topic_incremental_epochs: PositiveInt = 10
    topic_anchor_weight: float = 10.0

    nc_fscil_hidden_dim: PositiveInt = 128
    nc_fscil_hidden_layers: PositiveInt = 2
    nc_fscil_feature_dim: PositiveInt = 64
    nc_fscil_epochs: PositiveInt = 100
    nc_fscil_batch_size: PositiveInt = 256
    nc_fscil_q_max: PositiveInt = 64

    update_mode: UpdateMode = "full_retrain"
    incremental_finetune_epochs: PositiveInt = 10
    incremental_replay_budget: PositiveInt = 256
    incremental_freeze_backbone: bool = False
    incremental_lr_mult: float = 0.1

    model: BCMRNFSTConfig = Field(default_factory=BCMRNFSTConfig)

    @field_validator(
        "kernel_gamma",
        "kernel_coef0",
        "inn_center_separation_scale",
        "openmax_lr",
        "ewc_lr",
        "msp_lr",
        "doc_lr",
        "ewc_lambda",
        "doc_sigma_factor",
        "incremental_lr_mult",
    )
    @classmethod
    def validate_positive_float(cls, value: float) -> float:
        """Require strictly positive continuous hyperparameters."""

        if value <= 0.0:
            raise ValueError("Expected a strictly positive floating-point value.")
        return value

    @field_validator("dbscan_eps")
    @classmethod
    def validate_optional_positive_float(cls, value: float | None) -> float | None:
        """Require positive DBSCAN epsilon when explicitly provided."""

        if value is not None and value <= 0.0:
            raise ValueError("Expected dbscan_eps to be strictly positive when provided.")
        return value

    @field_validator("openmax_threshold", "msp_threshold")
    @classmethod
    def validate_probability_threshold(cls, value: float) -> float:
        """Require probability-style rejection thresholds to stay in [0, 1]."""

        if not 0.0 <= value <= 1.0:
            raise ValueError("Expected threshold to be in the interval [0, 1].")
        return value

    @field_validator("ewc_ncm_threshold_quantile")
    @classmethod
    def validate_quantile(cls, value: float) -> float:
        """Require the EWC+NCM rejection quantile to stay in the unit interval."""

        if not 0.0 < value <= 1.0:
            raise ValueError("Expected ewc_ncm_threshold_quantile to be in the interval (0, 1].")
        return value

    def resolve_path(self, path: Path, *, repo_root: Path | None = None) -> Path:
        """Resolve a config path relative to the repository root."""

        root = repo_root or find_repo_root()
        return path if path.is_absolute() else root / path

    def resolve_dataset_config(self, *, repo_root: Path | None = None) -> ToNIoTDatasetConfig:
        """Load the referenced dataset configuration."""

        return load_ton_iot_config(
            self.resolve_path(self.dataset_config_path, repo_root=repo_root),
            repo_root=repo_root,
        )


def load_baseline_experiment_config(
    path: Path,
    *,
    repo_root: Path | None = None,
) -> BaselineExperimentConfig:
    """Load a baseline experiment config from YAML."""

    resolved_path = path if path.is_absolute() else (repo_root or find_repo_root()) / path
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {resolved_path}")
    return BaselineExperimentConfig.model_validate(payload)


__all__ = [
    "DEFAULT_KERNEL_EXACT_MAX_TRAIN_ROWS",
    "BaselineExperimentConfig",
    "BaselineKind",
    "KernelMetric",
    "UpdateMode",
    "load_baseline_experiment_config",
]
