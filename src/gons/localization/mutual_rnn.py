"""Exact nearest-neighbor and mutual-rNN localization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

LabelArray: TypeAlias = NDArray[np.str_]
FeatureArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ResidualComponent:
    """Low-support residual component that must be preserved."""

    class_label: str
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """Localization output for regular units plus preserved residuals."""

    unit_ids: NDArray[np.object_]
    unit_to_class: dict[str, str]
    residual_components: tuple[ResidualComponent, ...]
    effective_r_by_class: dict[str, int]


def build_exact_rnn_graph(
    features: FeatureArray,
    r: int,
    *,
    metric: str = "euclidean",
) -> csr_matrix:
    """Build the exact directed r-nearest-neighbor graph.

    Parameters
    ----------
    metric:
        Distance metric for neighbour search.  ``"euclidean"`` (default) or
        ``"cosine"``.
    """

    features = np.asarray(features, dtype=np.float64)
    sample_count = int(features.shape[0])
    if sample_count == 0:
        return csr_matrix((0, 0), dtype=np.int8)
    if sample_count == 1:
        return csr_matrix((1, 1), dtype=np.int8)
    if r < 1:
        raise ValueError("r must be at least 1 for nearest-neighbor graph construction.")

    # `kneighbors(X=None)` already accounts for self-neighbors internally, so
    # request only the desired external neighbor count here.
    n_neighbors = min(r, sample_count - 1)
    model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    model.fit(features)
    neighbor_indices = model.kneighbors(return_distance=False)

    rows: list[int] = []
    cols: list[int] = []
    for row_index, neighbors in enumerate(neighbor_indices):
        filtered_neighbors = [
            int(neighbor) for neighbor in neighbors if int(neighbor) != row_index
        ][:r]
        rows.extend([row_index] * len(filtered_neighbors))
        cols.extend(filtered_neighbors)

    data = np.ones(len(rows), dtype=np.int8)
    return csr_matrix((data, (rows, cols)), shape=(sample_count, sample_count))


def mutual_rnn_graph(
    features: FeatureArray,
    r: int,
    *,
    metric: str = "euclidean",
) -> csr_matrix:
    """Keep only the mutual edges from the exact rNN graph."""

    directed_graph = build_exact_rnn_graph(features, r, metric=metric)
    return directed_graph.minimum(directed_graph.transpose()).tocsr()


def graph_connected_components(graph: csr_matrix) -> list[NDArray[np.int64]]:
    """Return connected components as index arrays."""

    if graph.shape[0] == 0:
        return []
    _, component_labels = connected_components(graph, directed=False, return_labels=True)
    return [
        np.flatnonzero(component_labels == component_index).astype(np.int64)
        for component_index in np.unique(component_labels)
    ]


def localize_mutual_rnn_units(
    features: FeatureArray,
    labels: NDArray[np.object_],
    *,
    r: int,
    min_microclass_size: int = 2,
    namespace: str = "temp",
    metric: str = "euclidean",
) -> LocalizationResult:
    """Localize class-wise mutual-rNN units while preserving residuals."""

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)
    unit_ids = np.full(labels.shape[0], None, dtype=object)
    unit_to_class: dict[str, str] = {}
    residual_components: list[ResidualComponent] = []
    effective_r_by_class: dict[str, int] = {}
    local_counter = 0

    for class_label in np.unique(labels):
        class_name = str(class_label)
        class_indices = np.flatnonzero(labels == class_label).astype(np.int64)
        class_size = int(class_indices.size)
        if class_size == 0:
            effective_r_by_class[class_name] = 0
            continue
        if class_size == 1:
            effective_r_by_class[class_name] = 0
            residual_components.append(
                ResidualComponent(class_label=class_name, indices=tuple(class_indices.tolist()))
            )
            continue

        r_eff = min(r, class_size - 1)
        effective_r_by_class[class_name] = r_eff
        if r_eff < 1:
            residual_components.append(
                ResidualComponent(class_label=class_name, indices=tuple(class_indices.tolist()))
            )
            continue

        class_graph = mutual_rnn_graph(features[class_indices], r_eff, metric=metric)
        for component in graph_connected_components(class_graph):
            component_indices = class_indices[component]
            if len(component_indices) >= min_microclass_size:
                unit_id = f"{namespace}:regular:{local_counter}"
                local_counter += 1
                unit_ids[component_indices] = unit_id
                unit_to_class[unit_id] = class_name
            else:
                residual_components.append(
                    ResidualComponent(
                        class_label=class_name,
                        indices=tuple(component_indices.astype(int).tolist()),
                    )
                )

    return LocalizationResult(
        unit_ids=unit_ids,
        unit_to_class=unit_to_class,
        residual_components=tuple(residual_components),
        effective_r_by_class=effective_r_by_class,
    )
