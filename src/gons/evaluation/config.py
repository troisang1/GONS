"""Configuration models for evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

import yaml
from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator

from gons.baselines.config import (
    DEFAULT_KERNEL_EXACT_MAX_TRAIN_ROWS,
    BaselineExperimentConfig,
    UpdateMode,
)
from gons.config.data import ToNIoTDatasetConfig, load_ton_iot_config
from gons.config.model import GONSConfig, RefreshMode, SupportTrimStrategy
from gons.data.ton_iot import normalize_label_value
from gons.runtime import find_repo_root

StaticOpenSetMethod: TypeAlias = Literal[
    "gons",
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
SessionContinualMethod: TypeAlias = StaticOpenSetMethod
PrequentialStreamMethod: TypeAlias = StaticOpenSetMethod
TinyAdmissionMethod: TypeAlias = StaticOpenSetMethod
FscilMethod: TypeAlias = StaticOpenSetMethod
CollapseTestsMethod: TypeAlias = Literal["gons"]
FidelityStudyMethod: TypeAlias = Literal["gons"]














class TinyAdmissionExperimentConfig(BaseModel):
    """Typed configuration for the FSCIL tiny-admission runner."""

    task: Literal["tiny_admission"] = "tiny_admission"
    experiment_name: str
    method: TinyAdmissionMethod
    dataset_config_path: Path = Path("configs/data/ton_iot.yaml")
    train_path: Path
    calibration_path: Path
    test_path: Path
    support_path: Path
    output_root: Path = Path("artifacts/runs")

    holdout_groups: list[list[str]] = Field(default_factory=list)
    include_leave_one_class_out: bool = True
    max_episodes: PositiveInt | None = None
    minimum_known_classes: PositiveInt = 2
    support_burst_sizes: list[PositiveInt] = Field(default_factory=lambda: [1])
    refresh_every_bursts: PositiveInt = 1
    run_final_refresh: bool = True
    enable_provisional_admission: bool = True
    ensemble_size: PositiveInt = 1
    ensemble_seed_offset: PositiveInt = 1000
    ensemble_aggregation: Literal["average", "max", "majority_vote"] = "average"
    post_promotion_enable_local_affine_screen: bool = False
    post_promotion_refresh_every_bursts: PositiveInt | None = None
    post_promotion_refresh_mode: RefreshMode | None = None
    post_promotion_support_trim_strategy: SupportTrimStrategy | None = None

    kernel: Literal["linear", "rbf", "poly", "sigmoid", "cosine"] = "rbf"
    kernel_gamma: float = 1.0
    kernel_degree: PositiveInt = 3
    kernel_coef0: float = 1.0
    kernel_exact_max_train_rows: PositiveInt = DEFAULT_KERNEL_EXACT_MAX_TRAIN_ROWS
    # Bounded class-stratified landmark support for incremental_kernel_nfst (IKNDA).
    # Without this field the runner's to_baseline_fit_config() dropped the YAML knob
    # silently, so each cumulative IKNDA session re-fit ran the FULL exact O(n^2) Gram +
    # O(n^3) eigh on the growing train (33k+ rows) -> ~1500s/session OOM/SIGSEGV/timeout.
    # When set, every exact-kernel fit subsamples to this many landmarks (fast + bounded).
    kernel_landmark_cap: PositiveInt | None = None

    pca_components: PositiveInt | None = None

    inn_neighbor_k: PositiveInt = 5
    inn_center_separation_scale: float = 1.0

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

    update_mode: UpdateMode = "full_retrain"
    incremental_finetune_epochs: PositiveInt = 10
    incremental_replay_budget: PositiveInt = 256
    incremental_freeze_backbone: bool = False
    incremental_lr_mult: float = 0.1

    model: GONSConfig = Field(default_factory=GONSConfig)

    @field_validator(
        "kernel_gamma",
        "kernel_coef0",
        "inn_center_separation_scale",
        "openmax_lr",
        "ewc_lr",
        "msp_lr",
        "doc_lr",
        "doc_sigma_factor",
        "incremental_lr_mult",
    )
    @classmethod
    def validate_positive_float(cls, value: float) -> float:
        """Require strictly positive continuous hyperparameters."""

        if value <= 0.0:
            raise ValueError("Expected a strictly positive floating-point value.")
        return value

    @field_validator("openmax_threshold", "msp_threshold")
    @classmethod
    def validate_unit_threshold(cls, value: float) -> float:
        """Require rejection thresholds to stay in the unit interval."""

        if not 0.0 <= value <= 1.0:
            raise ValueError("Expected threshold to be in the interval [0, 1].")
        return value

    @field_validator("holdout_groups", mode="before")
    @classmethod
    def normalize_admission_holdout_groups(cls, value: object) -> list[list[str]]:
        """Normalize explicit FSCIL holdout groups and require singleton groups."""

        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("Expected holdout_groups to be a list of label groups.")
        normalized_groups: list[list[str]] = []
        for group in value:
            if not isinstance(group, list):
                raise TypeError("Expected each holdout group to be a list of labels.")
            normalized_group: list[str] = []
            for label in group:
                normalized_label = normalize_label_value(label)
                if normalized_label not in normalized_group:
                    normalized_group.append(normalized_label)
            if len(normalized_group) != 1:
                raise ValueError("FSCIL holdout groups must contain exactly one novel label.")
            if normalized_group not in normalized_groups:
                normalized_groups.append(normalized_group)
        return normalized_groups

    @model_validator(mode="after")
    def validate_admission_policy(self) -> TinyAdmissionExperimentConfig:
        """Require a valid FSCIL episode and burst policy."""

        if not self.holdout_groups and not self.include_leave_one_class_out:
            raise ValueError(
                "Enable leave-one-class-out or provide explicit singleton holdout_groups."
            )
        if not self.support_burst_sizes:
            raise ValueError("FSCIL requires at least one support burst size.")
        if any(size > self.model.k_tiny_max for size in self.support_burst_sizes):
            raise ValueError(
                "Each FSCIL support burst must stay within model.k_tiny_max to remain a "
                "singleton/tiny admission episode."
            )
        if self.ensemble_size > 1 and self.method != "gons":
            raise ValueError("FSCIL ensembles are only supported for method=gons.")
        return self

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

    def to_baseline_fit_config(self) -> BaselineExperimentConfig:
        """Adapt the FSCIL config into the shared baseline fit config."""

        if self.method == "gons":
            raise ValueError("gons does not use the baseline config path.")
        return BaselineExperimentConfig(
            experiment_name=self.experiment_name,
            method=self.method,
            dataset_config_path=self.dataset_config_path,
            train_path=self.train_path,
            calibration_path=self.calibration_path,
            test_path=self.test_path,
            output_root=self.output_root,
            kernel=self.kernel,
            kernel_gamma=self.kernel_gamma,
            kernel_degree=self.kernel_degree,
            kernel_coef0=self.kernel_coef0,
            kernel_exact_max_train_rows=self.kernel_exact_max_train_rows,
            kernel_landmark_cap=self.kernel_landmark_cap,
            pca_components=self.pca_components,
            inn_neighbor_k=self.inn_neighbor_k,
            inn_center_separation_scale=self.inn_center_separation_scale,
            enable_dbscan_discovery=False,
            openmax_hidden_dim=self.openmax_hidden_dim,
            openmax_hidden_layers=self.openmax_hidden_layers,
            openmax_epochs=self.openmax_epochs,
            openmax_lr=self.openmax_lr,
            openmax_batch_size=self.openmax_batch_size,
            openmax_tail_size=self.openmax_tail_size,
            openmax_alpha_rank=self.openmax_alpha_rank,
            openmax_threshold=self.openmax_threshold,
            ewc_hidden_dim=self.ewc_hidden_dim,
            ewc_hidden_layers=self.ewc_hidden_layers,
            ewc_epochs=self.ewc_epochs,
            ewc_lr=self.ewc_lr,
            ewc_batch_size=self.ewc_batch_size,
            ewc_lambda=self.ewc_lambda,
            ewc_ncm_threshold_quantile=self.ewc_ncm_threshold_quantile,
            msp_hidden_dim=self.msp_hidden_dim,
            msp_hidden_layers=self.msp_hidden_layers,
            msp_epochs=self.msp_epochs,
            msp_lr=self.msp_lr,
            msp_batch_size=self.msp_batch_size,
            msp_threshold=self.msp_threshold,
            doc_hidden_dim=self.doc_hidden_dim,
            doc_hidden_layers=self.doc_hidden_layers,
            doc_epochs=self.doc_epochs,
            doc_lr=self.doc_lr,
            doc_batch_size=self.doc_batch_size,
            doc_sigma_factor=self.doc_sigma_factor,
            update_mode=self.update_mode,
            incremental_finetune_epochs=self.incremental_finetune_epochs,
            incremental_replay_budget=self.incremental_replay_budget,
            incremental_freeze_backbone=self.incremental_freeze_backbone,
            incremental_lr_mult=self.incremental_lr_mult,
            model=self.model,
        )


def load_tiny_admission_config(
    path: Path,
    *,
    repo_root: Path | None = None,
) -> TinyAdmissionExperimentConfig:
    """Load a FSCIL tiny-admission config from YAML."""

    resolved_path = path if path.is_absolute() else (repo_root or find_repo_root()) / path
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {resolved_path}")
    return TinyAdmissionExperimentConfig.model_validate(payload)


class FscilExperimentConfig(BaseModel):
    """Typed configuration for the FSCIL FSCIL runner."""

    task: Literal["fscil"] = "fscil"
    experiment_name: str
    method: FscilMethod
    dataset_config_path: Path = Path("configs/data/ton_iot.yaml")
    train_path: Path
    calibration_path: Path
    test_path: Path
    output_root: Path = Path("artifacts/runs")

    base_class_ratio: float = Field(default=0.5, gt=0.0, lt=1.0)
    base_class_seed: int = 42
    novel_support_fraction: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Fraction of novel-class train rows exposed during each FSCIL session.",
    )
    novel_support_max_k: int | None = Field(
        default=None,
        description="If set, cap the novel support at this many samples (absolute K-shot).",
    )
    novel_class_order: Literal["sorted", "random"] = "sorted"
    binary_collapse: bool = Field(
        default=False,
        description="If True, collapse all non-benign labels to 'attack' during training and inference. "
                    "Session planning still uses original multiclass labels.",
    )
    binary_benign_labels: list[str] = Field(
        default_factory=lambda: ["benign", "normal", "benigntraffic"],
        description="Labels considered benign for binary collapse.",
    )

    refresh_after_session: bool = True
    enable_provisional_admission: bool = True
    ensemble_size: PositiveInt = 1
    ensemble_seed_offset: PositiveInt = 1000
    ensemble_aggregation: Literal["average", "max", "majority_vote"] = "average"

    kernel: Literal["linear", "rbf", "poly", "sigmoid", "cosine"] = "rbf"
    kernel_gamma: float = 1.0
    kernel_degree: PositiveInt = 3
    kernel_coef0: float = 1.0
    kernel_exact_max_train_rows: PositiveInt = DEFAULT_KERNEL_EXACT_MAX_TRAIN_ROWS
    # Bounded class-stratified landmark support for incremental_kernel_nfst (IKNDA).
    # Without this field the FSCIL adapter dropped the YAML knob silently, so each
    # cumulative IKNDA session re-fit ran the FULL exact O(n^2) Gram + O(n^3) eigh on
    # the growing train (33k+ rows) -> the ~1500s/session OOM/SIGSEGV/timeout. When set,
    # every exact-kernel fit subsamples to this many landmarks (tractable + fast).
    kernel_landmark_cap: PositiveInt | None = None

    pca_components: PositiveInt | None = None

    inn_neighbor_k: PositiveInt = 5
    inn_center_separation_scale: float = 1.0

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

    update_mode: UpdateMode = "full_retrain"
    incremental_finetune_epochs: PositiveInt = 10
    incremental_replay_budget: PositiveInt = 256
    incremental_freeze_backbone: bool = False
    incremental_lr_mult: float = 0.1

    model: GONSConfig = Field(default_factory=GONSConfig)

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

    @field_validator("openmax_threshold", "msp_threshold")
    @classmethod
    def validate_unit_threshold(cls, value: float) -> float:
        """Require rejection thresholds to stay in the unit interval."""

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

    @model_validator(mode="after")
    def validate_fscil_policy(self) -> FscilExperimentConfig:
        """Require a valid FSCIL policy."""

        if self.ensemble_size > 1 and self.method != "gons":
            raise ValueError("FSCIL ensembles are only supported for method=gons.")
        return self

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

    def to_baseline_fit_config(self) -> BaselineExperimentConfig:
        """Adapt the FSCIL FSCIL config into the shared baseline fit config."""

        if self.method == "gons":
            raise ValueError("gons does not use the baseline config path.")
        return BaselineExperimentConfig(
            experiment_name=self.experiment_name,
            method=self.method,
            dataset_config_path=self.dataset_config_path,
            train_path=self.train_path,
            calibration_path=self.calibration_path,
            test_path=self.test_path,
            output_root=self.output_root,
            kernel=self.kernel,
            kernel_gamma=self.kernel_gamma,
            kernel_degree=self.kernel_degree,
            kernel_coef0=self.kernel_coef0,
            kernel_exact_max_train_rows=self.kernel_exact_max_train_rows,
            kernel_landmark_cap=self.kernel_landmark_cap,
            pca_components=self.pca_components,
            inn_neighbor_k=self.inn_neighbor_k,
            inn_center_separation_scale=self.inn_center_separation_scale,
            enable_dbscan_discovery=False,
            openmax_hidden_dim=self.openmax_hidden_dim,
            openmax_hidden_layers=self.openmax_hidden_layers,
            openmax_epochs=self.openmax_epochs,
            openmax_lr=self.openmax_lr,
            openmax_batch_size=self.openmax_batch_size,
            openmax_tail_size=self.openmax_tail_size,
            openmax_alpha_rank=self.openmax_alpha_rank,
            openmax_threshold=self.openmax_threshold,
            ewc_hidden_dim=self.ewc_hidden_dim,
            ewc_hidden_layers=self.ewc_hidden_layers,
            ewc_epochs=self.ewc_epochs,
            ewc_lr=self.ewc_lr,
            ewc_batch_size=self.ewc_batch_size,
            ewc_lambda=self.ewc_lambda,
            ewc_ncm_threshold_quantile=self.ewc_ncm_threshold_quantile,
            msp_hidden_dim=self.msp_hidden_dim,
            msp_hidden_layers=self.msp_hidden_layers,
            msp_epochs=self.msp_epochs,
            msp_lr=self.msp_lr,
            msp_batch_size=self.msp_batch_size,
            msp_threshold=self.msp_threshold,
            doc_hidden_dim=self.doc_hidden_dim,
            doc_hidden_layers=self.doc_hidden_layers,
            doc_epochs=self.doc_epochs,
            doc_lr=self.doc_lr,
            doc_batch_size=self.doc_batch_size,
            doc_sigma_factor=self.doc_sigma_factor,
            update_mode=self.update_mode,
            incremental_finetune_epochs=self.incremental_finetune_epochs,
            incremental_replay_budget=self.incremental_replay_budget,
            incremental_freeze_backbone=self.incremental_freeze_backbone,
            incremental_lr_mult=self.incremental_lr_mult,
            model=self.model,
        )


def load_fscil_config(
    path: Path,
    *,
    repo_root: Path | None = None,
) -> FscilExperimentConfig:
    """Load a FSCIL FSCIL config from YAML."""

    resolved_path = path if path.is_absolute() else (repo_root or find_repo_root()) / path
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {resolved_path}")
    return FscilExperimentConfig.model_validate(payload)










__all__ = [
    "FscilExperimentConfig",
    "FscilMethod",
    "TinyAdmissionExperimentConfig",
    "TinyAdmissionMethod",
    "load_fscil_config",
    "load_tiny_admission_config",
]
