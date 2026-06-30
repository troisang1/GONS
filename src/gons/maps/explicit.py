"""Frozen explicit map implementations and state serialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.kernel_approximation import Nystroem, RBFSampler

MapArray: TypeAlias = NDArray[np.float64]
MapKind: TypeAlias = Literal["linear", "rff", "nystrom"]
MapStateKind: TypeAlias = Literal[
    "linear",
    "rff",
    "nystrom",
    "laplacian_rff",
    "pre_whitened",
    "whitened",
]
LandmarkStrategy: TypeAlias = Literal["random", "kmeans", "supervised"]


@dataclass(frozen=True, slots=True)
class ExplicitMapConfig:
    """Configuration for a frozen explicit map."""

    kind: MapKind
    n_components: int = 256
    gamma: float = 1.0
    kernel: str = "rbf"
    random_state: int = 0
    n_landmarks: int | None = None
    landmark_strategy: LandmarkStrategy = "random"


@dataclass(frozen=True, slots=True)
class ExplicitMapState:
    """Serializable explicit-map state."""

    kind: MapStateKind
    params: dict[str, object]
    fitted: dict[str, object]


def _as_float_array(values: NDArray[np.float64] | np.ndarray) -> MapArray:
    return np.asarray(values, dtype=np.float64)


def _state_int(value: object) -> int:
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"Expected an integer state value, got {type(value)!r}.")
    return int(value)


def _state_float(value: object) -> float:
    if not isinstance(value, (float, int, np.floating, np.integer)):
        raise TypeError(f"Expected a floating-point state value, got {type(value)!r}.")
    return float(value)


def _state_str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected a string state value, got {type(value)!r}.")
    return value


def _state_array(value: object, *, dtype: np.dtype[np.float64] | np.dtype[np.int64]) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _state_explicit_map_state(value: object) -> ExplicitMapState:
    if not isinstance(value, ExplicitMapState):
        raise TypeError(f"Expected an ExplicitMapState, got {type(value)!r}.")
    return value


def compute_within_class_whitening(
    X: NDArray[np.float64],
    labels: NDArray[np.object_],
    *,
    epsilon: float = 1e-10,
) -> NDArray[np.float64]:
    """Compute a within-class whitening matrix for one feature space."""

    features = _as_float_array(X)
    if features.ndim != 2:
        raise ValueError("Expected a 2D feature matrix.")
    if features.shape[0] != len(labels):
        raise ValueError("Expected labels to align with the feature rows.")
    if features.shape[0] == 0:
        return np.eye(features.shape[1], dtype=np.float64)

    n_features = int(features.shape[1])
    scatter = np.zeros((n_features, n_features), dtype=np.float64)
    total_count = 0
    for class_label in np.unique(labels):
        class_rows = np.asarray(features[labels == class_label], dtype=np.float64)
        if class_rows.shape[0] < 2:
            continue
        center = np.mean(class_rows, axis=0, dtype=np.float64)
        centered = class_rows - center[None, :]
        scatter += centered.T @ centered
        total_count += int(class_rows.shape[0])

    if total_count == 0:
        return np.eye(n_features, dtype=np.float64)

    scatter /= float(total_count)
    eigvals, eigvecs = np.linalg.eigh(scatter)
    eigvals = np.maximum(eigvals, epsilon)
    whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return np.asarray(whitening, dtype=np.float64)


def _landmark_allocation_by_class(
    labels: NDArray[np.object_],
    n_landmarks: int,
) -> dict[str, int]:
    """Allocate landmark budget across classes while keeping minority coverage."""

    if n_landmarks < 1:
        return {}
    unique_labels = [str(label) for label in np.unique(labels)]
    class_counts = {
        class_label: int(np.sum(labels == class_label))
        for class_label in unique_labels
    }
    if not class_counts:
        return {}

    if n_landmarks < len(class_counts):
        ranked = sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))
        return {
            class_label: 1 if index < n_landmarks else 0
            for index, (class_label, _) in enumerate(ranked)
        }

    allocation = {class_label: 1 for class_label in class_counts}
    remaining = n_landmarks - len(class_counts)
    if remaining == 0:
        return allocation

    total_count = sum(class_counts.values())
    fractions = {
        class_label: (remaining * class_counts[class_label]) / max(total_count, 1)
        for class_label in class_counts
    }
    for class_label, share in fractions.items():
        extra = int(np.floor(share))
        allocation[class_label] += extra
        remaining -= extra

    remainders = sorted(
        (
            fractions[class_label] - np.floor(fractions[class_label]),
            class_label,
        )
        for class_label in class_counts
    )
    while remaining > 0 and remainders:
        _, class_label = remainders.pop()
        allocation[class_label] += 1
        remaining -= 1

    return allocation


def _select_supervised_landmarks(
    x_fit: NDArray[np.float64],
    labels: NDArray[np.object_],
    n_landmarks: int,
    random_state: int,
) -> NDArray[np.int64]:
    """Select Nyström landmarks with class-aware representative coverage."""

    features = _as_float_array(x_fit)
    if features.shape[0] != len(labels):
        raise ValueError("Supervised landmarks require labels aligned to feature rows.")
    if n_landmarks >= features.shape[0]:
        return np.arange(features.shape[0], dtype=np.int64)

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import pairwise_distances_argmin

    allocation = _landmark_allocation_by_class(labels, n_landmarks)
    rng = np.random.default_rng(random_state)
    selected: list[int] = []

    for class_label in np.unique(labels):
        class_name = str(class_label)
        target = min(allocation.get(class_name, 0), int(np.sum(labels == class_label)))
        if target <= 0:
            continue
        class_indices = np.flatnonzero(labels == class_label).astype(np.int64)
        if target >= len(class_indices):
            selected.extend(class_indices.tolist())
            continue
        class_rows = np.asarray(features[class_indices], dtype=np.float64)
        kmeans = MiniBatchKMeans(
            n_clusters=target,
            batch_size=min(1024, len(class_rows)),
            n_init=3,
            random_state=random_state,
        )
        kmeans.fit(class_rows)
        nearest = pairwise_distances_argmin(kmeans.cluster_centers_, class_rows)
        chosen = class_indices[np.asarray(nearest, dtype=np.int64)]
        unique_chosen = np.unique(chosen)
        if unique_chosen.size < target:
            remaining = np.setdiff1d(class_indices, unique_chosen, assume_unique=False)
            fill = rng.choice(
                remaining,
                size=target - unique_chosen.size,
                replace=False,
            ).astype(np.int64)
            unique_chosen = np.concatenate([unique_chosen, fill])
        selected.extend(np.asarray(unique_chosen[:target], dtype=np.int64).tolist())

    selected_array = np.unique(np.asarray(selected, dtype=np.int64))
    if selected_array.size >= n_landmarks:
        return np.sort(selected_array[:n_landmarks])

    remaining_indices = np.setdiff1d(
        np.arange(features.shape[0], dtype=np.int64),
        selected_array,
        assume_unique=False,
    )
    fill = rng.choice(
        remaining_indices,
        size=n_landmarks - selected_array.size,
        replace=False,
    ).astype(np.int64)
    return np.sort(np.concatenate([selected_array, fill]))


@dataclass(slots=True)
class IdentityExplicitMap:
    """Linear identity explicit map."""

    n_features_in_: int | None = None
    kind: Literal["linear"] = "linear"

    def fit(self, x_fit: MapArray) -> IdentityExplicitMap:
        self.n_features_in_ = int(_as_float_array(x_fit).shape[1])
        return self

    def transform(self, x: MapArray) -> MapArray:
        transformed = _as_float_array(x)
        if self.n_features_in_ is not None and transformed.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} input features, got {transformed.shape[1]}."
            )
        return transformed

    def to_state(self) -> ExplicitMapState:
        if self.n_features_in_ is None:
            raise ValueError("Identity map must be fitted before serialization.")
        return ExplicitMapState(
            kind=self.kind,
            params={},
            fitted={"n_features_in_": self.n_features_in_},
        )

    @classmethod
    def from_state(cls, state: ExplicitMapState) -> IdentityExplicitMap:
        return cls(n_features_in_=_state_int(state.fitted["n_features_in_"]))


@dataclass(slots=True)
class RFFExplicitMap:
    """Frozen random Fourier feature map."""

    gamma: float = 1.0
    n_components: int = 256
    random_state: int = 0
    estimator: RBFSampler | None = None
    kind: Literal["rff"] = "rff"

    def fit(self, x_fit: MapArray) -> RFFExplicitMap:
        estimator = RBFSampler(
            gamma=self.gamma,
            n_components=self.n_components,
            random_state=self.random_state,
        )
        estimator.fit(_as_float_array(x_fit))
        self.estimator = estimator
        return self

    def transform(self, x: MapArray) -> MapArray:
        if self.estimator is None:
            raise ValueError("RFF map must be fitted before transform.")
        return _as_float_array(self.estimator.transform(_as_float_array(x)))

    def to_state(self) -> ExplicitMapState:
        if self.estimator is None:
            raise ValueError("RFF map must be fitted before serialization.")
        return ExplicitMapState(
            kind=self.kind,
            params={
                "gamma": self.gamma,
                "n_components": self.n_components,
                "random_state": self.random_state,
            },
            fitted={
                "n_features_in_": int(self.estimator.n_features_in_),
                "random_offset_": _as_float_array(self.estimator.random_offset_),
                "random_weights_": _as_float_array(self.estimator.random_weights_),
            },
        )

    @classmethod
    def from_state(cls, state: ExplicitMapState) -> RFFExplicitMap:
        restored = cls(
            gamma=_state_float(state.params["gamma"]),
            n_components=_state_int(state.params["n_components"]),
            random_state=_state_int(state.params["random_state"]),
        )
        estimator = RBFSampler(
            gamma=restored.gamma,
            n_components=restored.n_components,
            random_state=restored.random_state,
        )
        estimator.n_features_in_ = _state_int(state.fitted["n_features_in_"])
        estimator.random_offset_ = _state_array(
            state.fitted["random_offset_"],
            dtype=np.dtype(np.float64),
        )
        estimator.random_weights_ = _state_array(
            state.fitted["random_weights_"],
            dtype=np.dtype(np.float64),
        )
        restored.estimator = estimator
        return restored


@dataclass(slots=True)
class LaplacianRFFExplicitMap:
    """Random Fourier features for the Laplacian kernel."""

    gamma: float = 1.0
    n_components: int = 256
    random_state: int = 0
    random_weights_: NDArray[np.float64] | None = None
    random_offset_: NDArray[np.float64] | None = None
    n_features_in_: int | None = None
    kind: Literal["laplacian_rff"] = "laplacian_rff"

    def fit(self, x_fit: MapArray) -> LaplacianRFFExplicitMap:
        x_fit = _as_float_array(x_fit)
        rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = int(x_fit.shape[1])
        self.random_weights_ = np.asarray(
            rng.standard_cauchy(size=(self.n_components, self.n_features_in_)) * self.gamma,
            dtype=np.float64,
        )
        self.random_offset_ = np.asarray(
            rng.uniform(0.0, 2.0 * np.pi, size=self.n_components),
            dtype=np.float64,
        )
        return self

    def transform(self, x: MapArray) -> MapArray:
        if self.random_weights_ is None or self.random_offset_ is None:
            raise ValueError("Laplacian RFF map must be fitted before transform.")
        transformed = _as_float_array(x)
        projection = transformed @ self.random_weights_.T + self.random_offset_[None, :]
        scale = np.sqrt(2.0 / max(self.n_components, 1))
        return np.asarray(scale * np.cos(projection), dtype=np.float64)

    def to_state(self) -> ExplicitMapState:
        if (
            self.random_weights_ is None
            or self.random_offset_ is None
            or self.n_features_in_ is None
        ):
            raise ValueError("Laplacian RFF map must be fitted before serialization.")
        return ExplicitMapState(
            kind=self.kind,
            params={
                "gamma": self.gamma,
                "n_components": self.n_components,
                "random_state": self.random_state,
            },
            fitted={
                "n_features_in_": self.n_features_in_,
                "random_offset_": _as_float_array(self.random_offset_),
                "random_weights_": _as_float_array(self.random_weights_),
            },
        )

    @classmethod
    def from_state(cls, state: ExplicitMapState) -> LaplacianRFFExplicitMap:
        return cls(
            gamma=_state_float(state.params["gamma"]),
            n_components=_state_int(state.params["n_components"]),
            random_state=_state_int(state.params["random_state"]),
            random_weights_=_state_array(
                state.fitted["random_weights_"],
                dtype=np.dtype(np.float64),
            ),
            random_offset_=_state_array(
                state.fitted["random_offset_"],
                dtype=np.dtype(np.float64),
            ),
            n_features_in_=_state_int(state.fitted["n_features_in_"]),
        )


@dataclass(slots=True)
class NystromExplicitMap:
    """Frozen Nyström explicit map."""

    gamma: float = 1.0
    n_components: int = 256
    kernel: str = "rbf"
    random_state: int = 0
    n_landmarks: int | None = None
    landmark_strategy: LandmarkStrategy = "random"
    landmark_indices_: NDArray[np.int64] | None = None
    estimator: Nystroem | None = None
    kind: Literal["nystrom"] = "nystrom"

    def fit(
        self,
        x_fit: MapArray,
        *,
        labels: NDArray[np.object_] | None = None,
    ) -> NystromExplicitMap:
        x_fit = _as_float_array(x_fit)
        sample_count = x_fit.shape[0]
        target_landmarks = min(sample_count, self.n_landmarks or self.n_components)
        if self.landmark_strategy == "kmeans" and sample_count > target_landmarks:
            from sklearn.cluster import MiniBatchKMeans

            kmeans = MiniBatchKMeans(
                n_clusters=target_landmarks,
                random_state=self.random_state,
                batch_size=min(1024, sample_count),
                n_init=3,
            )
            kmeans.fit(x_fit)
            centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
            from sklearn.metrics import pairwise_distances_argmin

            landmark_indices = np.sort(
                pairwise_distances_argmin(centers, x_fit).astype(np.int64),
            )
        elif self.landmark_strategy == "supervised" and sample_count > target_landmarks:
            if labels is None:
                raise ValueError("Supervised landmark selection requires labels.")
            landmark_indices = _select_supervised_landmarks(
                x_fit,
                labels,
                target_landmarks,
                self.random_state,
            )
        else:
            rng = np.random.default_rng(self.random_state)
            landmark_indices = np.sort(
                rng.choice(sample_count, size=target_landmarks, replace=False).astype(np.int64),
            )
        estimator = Nystroem(
            kernel=self.kernel,
            gamma=self.gamma,
            n_components=target_landmarks,
            random_state=self.random_state,
        )
        estimator.fit(x_fit[landmark_indices])
        self.landmark_indices_ = landmark_indices
        self.estimator = estimator
        return self

    def transform(self, x: MapArray) -> MapArray:
        if self.estimator is None:
            raise ValueError("Nyström map must be fitted before transform.")
        return _as_float_array(self.estimator.transform(_as_float_array(x)))

    def to_state(self) -> ExplicitMapState:
        if self.estimator is None or self.landmark_indices_ is None:
            raise ValueError("Nyström map must be fitted before serialization.")
        return ExplicitMapState(
            kind=self.kind,
            params={
                "gamma": self.gamma,
                "kernel": self.kernel,
                "landmark_strategy": self.landmark_strategy,
                "n_components": int(self.estimator.n_components),
                "n_landmarks": int(len(self.landmark_indices_)),
                "random_state": self.random_state,
            },
            fitted={
                "components_": _as_float_array(self.estimator.components_),
                "component_indices_": np.asarray(self.estimator.component_indices_, dtype=np.int64),
                "landmark_indices_": np.asarray(self.landmark_indices_, dtype=np.int64),
                "n_features_in_": int(self.estimator.n_features_in_),
                "normalization_": _as_float_array(self.estimator.normalization_),
            },
        )

    @classmethod
    def from_state(cls, state: ExplicitMapState) -> NystromExplicitMap:
        landmark_strategy: LandmarkStrategy = "random"
        if "landmark_strategy" in state.params:
            landmark_strategy = str(state.params["landmark_strategy"])  # type: ignore[assignment]
        restored = cls(
            gamma=_state_float(state.params["gamma"]),
            kernel=_state_str(state.params["kernel"]),
            n_components=_state_int(state.params["n_components"]),
            n_landmarks=_state_int(state.params["n_landmarks"]),
            random_state=_state_int(state.params["random_state"]),
            landmark_strategy=landmark_strategy,
        )
        estimator = Nystroem(
            kernel=restored.kernel,
            gamma=restored.gamma,
            n_components=restored.n_components,
            random_state=restored.random_state,
        )
        estimator.components_ = _state_array(
            state.fitted["components_"],
            dtype=np.dtype(np.float64),
        )
        estimator.component_indices_ = _state_array(
            state.fitted["component_indices_"],
            dtype=np.dtype(np.int64),
        )
        estimator.n_features_in_ = _state_int(state.fitted["n_features_in_"])
        estimator.normalization_ = _state_array(
            state.fitted["normalization_"],
            dtype=np.dtype(np.float64),
        )
        restored.landmark_indices_ = _state_array(
            state.fitted["landmark_indices_"],
            dtype=np.dtype(np.int64),
        )
        restored.estimator = estimator
        return restored


@dataclass(slots=True)
class PreWhitenedExplicitMap:
    """Wrap a fitted explicit map with raw-space whitening before the map."""

    base_map: ExplicitMap
    whitening_matrix: NDArray[np.float64] | None = None
    kind: Literal["pre_whitened"] = "pre_whitened"

    def transform(self, x: MapArray) -> MapArray:
        if self.whitening_matrix is None:
            raise ValueError("Pre-whitened map must be fitted before transform.")
        whitened = np.asarray(_as_float_array(x) @ self.whitening_matrix.T, dtype=np.float64)
        return np.asarray(self.base_map.transform(whitened), dtype=np.float64)

    def to_state(self) -> ExplicitMapState:
        if self.whitening_matrix is None:
            raise ValueError("Pre-whitened map must be fitted before serialization.")
        return ExplicitMapState(
            kind=self.kind,
            params={},
            fitted={
                "base_state": self.base_map.to_state(),
                "whitening_matrix": _as_float_array(self.whitening_matrix),
            },
        )

    @classmethod
    def from_state(cls, state: ExplicitMapState) -> PreWhitenedExplicitMap:
        return cls(
            base_map=explicit_map_from_state(_state_explicit_map_state(state.fitted["base_state"])),
            whitening_matrix=_state_array(
                state.fitted["whitening_matrix"],
                dtype=np.dtype(np.float64),
            ),
        )


@dataclass(slots=True)
class WhitenedExplicitMap:
    """Wrap a fitted explicit map with within-class whitening."""

    base_map: ExplicitMap
    whitening_matrix: NDArray[np.float64] | None = None
    kind: Literal["whitened"] = "whitened"

    @classmethod
    def from_fitted_base(
        cls,
        base_map: ExplicitMap,
        transformed_train: MapArray,
        labels: NDArray[np.object_] | None = None,
        *,
        epsilon: float = 1e-10,
    ) -> WhitenedExplicitMap:
        whitening = np.eye(transformed_train.shape[1], dtype=np.float64)
        if labels is not None and transformed_train.shape[0] > 0:
            whitening = compute_within_class_whitening(
                np.asarray(transformed_train, dtype=np.float64),
                labels,
                epsilon=epsilon,
            )
        return cls(base_map=base_map, whitening_matrix=whitening)

    def transform(self, x: MapArray) -> MapArray:
        if self.whitening_matrix is None:
            raise ValueError("Whitened map must be fitted before transform.")
        transformed = np.asarray(self.base_map.transform(x), dtype=np.float64)
        return np.asarray(transformed @ self.whitening_matrix.T, dtype=np.float64)

    def to_state(self) -> ExplicitMapState:
        if self.whitening_matrix is None:
            raise ValueError("Whitened map must be fitted before serialization.")
        return ExplicitMapState(
            kind=self.kind,
            params={},
            fitted={
                "base_state": self.base_map.to_state(),
                "whitening_matrix": _as_float_array(self.whitening_matrix),
            },
        )

    @classmethod
    def from_state(cls, state: ExplicitMapState) -> WhitenedExplicitMap:
        return cls(
            base_map=explicit_map_from_state(_state_explicit_map_state(state.fitted["base_state"])),
            whitening_matrix=_state_array(
                state.fitted["whitening_matrix"],
                dtype=np.dtype(np.float64),
            ),
        )


ExplicitMap: TypeAlias = (
    IdentityExplicitMap
    | RFFExplicitMap
    | LaplacianRFFExplicitMap
    | NystromExplicitMap
    | PreWhitenedExplicitMap
    | WhitenedExplicitMap
)


def _compute_median_heuristic_gamma(
    x_fit: MapArray,
    *,
    max_sample: int = 5000,
    random_state: int = 0,
) -> float:
    """Compute gamma via the median heuristic: gamma = 1 / median(||xi - xj||^2).

    Uses a random subsample to avoid O(n^2) cost on large datasets.
    """
    rng = np.random.default_rng(random_state)
    n = x_fit.shape[0]
    if n > max_sample:
        idx = rng.choice(n, max_sample, replace=False)
        sample = x_fit[idx]
    else:
        sample = x_fit
    # Compute pairwise squared distances on a further subsample for speed
    n_half = min(len(sample) // 2, 2000)
    a, b = sample[:n_half], sample[n_half : 2 * n_half]
    sq_dists = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2).ravel()
    median_sq = float(np.median(sq_dists))
    if median_sq < 1e-12:
        return 1.0
    return 1.0 / median_sq


def fit_explicit_map(
    x_fit: MapArray,
    config: ExplicitMapConfig,
    *,
    labels: NDArray[np.object_] | None = None,
) -> ExplicitMap:
    """Fit one of the supported frozen explicit maps.

    If ``config.gamma`` is negative (e.g. -1.0), the median heuristic is used
    to automatically determine gamma from the training data.
    """

    effective_gamma = config.gamma
    if effective_gamma < 0:
        effective_gamma = _compute_median_heuristic_gamma(
            x_fit, random_state=config.random_state,
        )

    if config.kind == "linear":
        return IdentityExplicitMap().fit(x_fit)
    if config.kind == "rff":
        if config.kernel == "laplacian":
            return LaplacianRFFExplicitMap(
                gamma=effective_gamma,
                n_components=config.n_components,
                random_state=config.random_state,
            ).fit(x_fit)
        return RFFExplicitMap(
            gamma=effective_gamma,
            n_components=config.n_components,
            random_state=config.random_state,
        ).fit(x_fit)
    if config.kind == "nystrom":
        return NystromExplicitMap(
            gamma=effective_gamma,
            n_components=config.n_components,
            kernel=config.kernel,
            random_state=config.random_state,
            n_landmarks=config.n_landmarks,
            landmark_strategy=config.landmark_strategy,
        ).fit(x_fit, labels=labels)
    raise ValueError(f"Unsupported explicit map kind: {config.kind}")


def explicit_map_from_state(state: ExplicitMapState) -> ExplicitMap:
    """Rehydrate a fitted explicit map from serialized state."""

    if state.kind == "linear":
        return IdentityExplicitMap.from_state(state)
    if state.kind == "rff":
        return RFFExplicitMap.from_state(state)
    if state.kind == "laplacian_rff":
        return LaplacianRFFExplicitMap.from_state(state)
    if state.kind == "nystrom":
        return NystromExplicitMap.from_state(state)
    if state.kind == "pre_whitened":
        return PreWhitenedExplicitMap.from_state(state)
    if state.kind == "whitened":
        return WhitenedExplicitMap.from_state(state)
    raise ValueError(f"Unsupported explicit map kind: {state.kind}")


def save_explicit_map_state(explicit_map: ExplicitMap, path: Path) -> None:
    """Persist a serialized explicit-map state with joblib."""

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(explicit_map.to_state(), path)


def load_explicit_map_state(path: Path) -> ExplicitMap:
    """Load a serialized explicit-map state."""

    state = joblib.load(path)
    if not isinstance(state, ExplicitMapState):
        raise TypeError(f"Unexpected explicit map state payload in {path}")
    return explicit_map_from_state(state)
