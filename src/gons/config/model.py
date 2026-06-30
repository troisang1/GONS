"""Core model configuration for offline GONS fitting."""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, Field, PositiveInt, field_validator

from gons.config.data import PreprocessingConfig
from gons.maps.explicit import ExplicitMapConfig, LandmarkStrategy, MapKind

NystromLandmarkStrategy: TypeAlias = LandmarkStrategy
UnknownBufferPolicy: TypeAlias = Literal["fifo", "reservoir"]
ThresholdMode: TypeAlias = Literal["relative", "fixed_absolute"]
ScoringMode: TypeAlias = Literal["min_distance", "energy", "relative_mahalanobis"]
SmallEigSolver: TypeAlias = Literal["ladder", "dense_only"]
RefreshMode: TypeAlias = Literal["fixed_point", "single_pass"]
ResidualHandling: TypeAlias = Literal["preserve", "drop", "force_merge"]
SupportTrimStrategy: TypeAlias = Literal["latest", "uniform", "leverage"]
CalibrationMode: TypeAlias = Literal["grid_search", "evt", "loco", "per_proto_evt"]
ShrinkageMode: TypeAlias = Literal["adaptive", "diagonal"]
PRACTICAL_UNBOUNDED_LIMIT = 1_000_000_000


class GONSConfig(BaseModel):
    """Typed configuration for the offline GONS fit path."""

    label_column: str = "label"
    seed: int = 42

    map_kind: MapKind = "linear"
    map_n_components: PositiveInt = 64
    map_gamma: float = 1.0
    map_kernel: str = "rbf"
    map_n_landmarks: PositiveInt | None = None
    nystrom_landmark_strategy: LandmarkStrategy = "random"
    enable_pre_map_whitening: bool = False

    r: PositiveInt = 5
    min_microclass_size: PositiveInt = 2
    R_q_max: PositiveInt = PRACTICAL_UNBOUNDED_LIMIT
    C_active_max: PositiveInt = PRACTICAL_UNBOUNDED_LIMIT
    Q_stats_max: PositiveInt = PRACTICAL_UNBOUNDED_LIMIT
    Q_proto_max: PositiveInt = PRACTICAL_UNBOUNDED_LIMIT

    d_proj_max: PositiveInt = 8
    # Ablation E4: replace null-space projection with PCA or random projection
    # of the same dimension ("Why null-space, not PCA or random?"). Default
    # `null_space` preserves the standard GONS behavior.
    projection_mode: Literal[
        "null_space", "pca", "random"
    ] = "null_space"
    n_dense_max: PositiveInt = 256
    small_eig_solver: SmallEigSolver = "ladder"
    use_shift_invert_small: bool = False
    eig_tol: float = 1e-7
    eig_maxiter: PositiveInt = 5000
    c_reg: float = 0.0
    alpha_warmstart: float = 0.0

    threshold_mode: ThresholdMode = "relative"
    scoring_mode: ScoringMode = "min_distance"
    calibration_mode: CalibrationMode = "grid_search"
    c_range: float = 1e-6
    c_null: float = 1e-6
    c_aff_local: float = 1.0
    c_aff_global: float = 2.0
    c_margin: float = 1.0
    c_rho: float = 0.1
    c_sep: float = 1.1

    radius_quantile: float = 0.9
    fragile_tau_mult: float = 0.8
    initial_tau_global: float = 1.0

    alpha_false_known: float = 0.05
    beta_true_support: float = 0.9
    threshold_quantile: float = 0.95
    threshold_grid_size: PositiveInt = 64
    calibration_reservoir_size: PositiveInt = 512
    evt_tail_size: PositiveInt = 20
    evt_target_coverage: float = 0.95
    energy_temperature: float = 1.0
    enable_evm_calibration: bool = False
    evm_tail_size: PositiveInt = 20
    evm_inclusion_threshold: float = 0.5
    enable_per_proto_threshold: bool = False
    per_proto_evt_tail_size: PositiveInt = 15
    per_proto_evt_coverage: float = 0.95
    per_proto_tau_fallback_mult: float = 1.0
    enable_per_class_quantile_threshold: bool = False
    per_class_threshold_quantile: float = 0.95
    enable_ghost_threshold: bool = False
    ghost_alpha: float = 0.05
    enable_knn_scoring: bool = False
    knn_k: PositiveInt = 10
    enable_reciprocal_points: bool = False
    reciprocal_lambda: float = 0.5
    enable_hyperspherical_scoring: bool = False
    enable_impostor_calibration: bool = True
    unknown_buffer_policy: UnknownBufferPolicy = "fifo"
    enable_unknown_buffer_rescoring: bool = True
    use_mahalanobis: bool = False
    shrinkage_mode: ShrinkageMode = "adaptive"
    B_u: PositiveInt = 512
    B_pending_max: PositiveInt = 128
    k_tiny_max: PositiveInt = 4
    affine_shortlist_k: PositiveInt = 8
    enable_local_affine_screen: bool = True
    singleton_tau_mult: float = 0.5
    tiny_tau_mult: float = 0.7
    prov_tau_mult: float = 0.85
    refresh_mode: RefreshMode = "fixed_point"
    enable_refresh_validation: bool = False
    refresh_validation_budget: PositiveInt = 256
    refresh_validation_min_acc: float = 0.0
    residual_handling: ResidualHandling = "preserve"
    support_trim_strategy: SupportTrimStrategy = "latest"
    enable_archived_class_rehydration: bool = True
    enable_teen_calibration: bool = False
    teen_lambda_base: float = 0.3
    teen_temperature: float = 0.1
    teen_top_k: PositiveInt = 5
    c_squeeze: float = 0.0
    enable_contrastive_rff: bool = False
    enable_dual_space_scoring: bool = False
    dual_space_alpha: float = 0.5
    enable_pa_inn: bool = False
    pa_inn_n_anchors: PositiveInt = 50
    pa_inn_min_samples: PositiveInt = 10
    pa_inn_regularization: float = 0.01
    localization_metric: Literal["euclidean", "cosine"] = "euclidean"
    scatter_class_balance_gamma: float = 0.0
    score_support_penalty: bool = False
    score_support_penalty_fn: Literal["log", "sqrt"] = "log"
    score_support_penalty_alpha: float = 1.0
    enable_ncdr_classification: bool = False
    ncdr_threshold: float = 0.85
    enable_support_radius_shrink: bool = False
    radius_shrink_rate: float = 0.08
    enable_class_cond_temperature: bool = False
    temperature_support_gamma: float = 0.4
    novel_threshold_mult: float = 1.0
    c_sep_singleton: float = 1.25
    lambda_decay: float = 0.0
    w_drop: float = 0.0

    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)

    @field_validator(
        "radius_quantile",
        "alpha_false_known",
        "beta_true_support",
        "threshold_quantile",
        "evt_target_coverage",
        "evm_inclusion_threshold",
        "ghost_alpha",
        "per_proto_evt_coverage",
        "per_class_threshold_quantile",
    )
    @classmethod
    def validate_unit_interval(cls, value: float) -> float:
        """Require threshold-related hyperparameters to stay in the unit interval."""

        if not 0.0 < value <= 1.0:
            raise ValueError("Expected a value in the interval (0, 1].")
        return value

    @field_validator(
        "c_range",
        "c_null",
        "c_aff_local",
        "c_aff_global",
        "c_margin",
        "c_rho",
        "c_sep",
        "c_sep_singleton",
        "fragile_tau_mult",
        "initial_tau_global",
        "eig_tol",
        "energy_temperature",
        "reciprocal_lambda",
        "per_proto_tau_fallback_mult",
        "singleton_tau_mult",
        "tiny_tau_mult",
        "prov_tau_mult",
        "teen_temperature",
    )
    @classmethod
    def validate_positive_float(cls, value: float) -> float:
        """Require strictly positive continuous hyperparameters."""

        if value <= 0.0:
            raise ValueError("Expected a strictly positive floating-point value.")
        return value

    @field_validator("lambda_decay", "w_drop", "c_reg", "pa_inn_regularization", "scatter_class_balance_gamma", "radius_shrink_rate")
    @classmethod
    def validate_non_negative_float(cls, value: float) -> float:
        """Require non-negative decay-related hyperparameters."""

        if value < 0.0:
            raise ValueError("Expected a non-negative floating-point value.")
        return value

    @field_validator(
        "alpha_warmstart",
        "teen_lambda_base",
        "c_squeeze",
        "refresh_validation_min_acc",
        "dual_space_alpha",
        "novel_threshold_mult",
        "score_support_penalty_alpha",
        "ncdr_threshold",
        "temperature_support_gamma",
    )
    @classmethod
    def validate_non_negative_unit_interval(cls, value: float) -> float:
        """Require bounded blend-style hyperparameters in the unit interval."""

        if not 0.0 <= value <= 1.0:
            raise ValueError("Expected a value in the interval [0, 1].")
        return value

    def explicit_map_config(self) -> ExplicitMapConfig:
        """Convert the fit config into an explicit-map config."""

        return ExplicitMapConfig(
            kind=self.map_kind,
            n_components=self.map_n_components,
            gamma=self.map_gamma,
            kernel=self.map_kernel,
            random_state=self.seed,
            n_landmarks=self.map_n_landmarks,
            landmark_strategy=self.nystrom_landmark_strategy,
        )
