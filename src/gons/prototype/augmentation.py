"""PA-iNN: Prototype-Augmented iNN for low-support scatter stabilization.

Generates synthetic anchors from local Gaussians for classes with very few
samples per localized component.  Anchors are used only for scatter matrix
computation and are discarded before test-time inference.

Transferred from the Few-Shot MND sub-project (industrial-defect-detection).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def augment_support_for_scatter(
    phi_support: NDArray[np.float64],
    labels: NDArray[np.object_],
    *,
    min_samples_per_class: int = 10,
    n_anchors: int = 50,
    regularization: float = 0.01,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.object_]]:
    """Add synthetic anchors for under-represented classes.

    For each class whose support size is below *min_samples_per_class*, sample
    *n_anchors* synthetic points from N(mu_c, reg * I + Sigma_c) and append
    them.  The returned arrays include both real and synthetic samples and can
    be passed directly to scatter computation.

    Parameters
    ----------
    phi_support:
        Explicit-space feature matrix (N, D).
    labels:
        Class labels aligned with *phi_support* (N,).
    min_samples_per_class:
        Classes with fewer samples than this receive augmentation.
    n_anchors:
        Number of synthetic anchors to generate per augmented class.
    regularization:
        Diagonal regularization added to the estimated covariance to
        avoid singular sampling distributions.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    augmented_features:
        (N + M, D) feature matrix with synthetic anchors appended.
    augmented_labels:
        (N + M,) label array with corresponding class labels.
    """

    phi_support = np.asarray(phi_support, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)
    rng = np.random.default_rng(seed)

    augmented_phi: list[NDArray[np.float64]] = [phi_support]
    augmented_labels: list[NDArray[np.object_]] = [labels]

    unique_labels = np.unique(labels)
    dim = phi_support.shape[1]

    for class_label in unique_labels:
        class_mask = labels == class_label
        class_features = phi_support[class_mask]
        n_class = class_features.shape[0]

        if n_class >= min_samples_per_class:
            continue

        mu_c = class_features.mean(axis=0)

        if n_class < 2:
            # Single sample: use isotropic Gaussian around the point.
            cov_c = regularization * np.eye(dim, dtype=np.float64)
        else:
            cov_c = np.cov(class_features, rowvar=False, bias=True)
            cov_c = np.asarray(cov_c, dtype=np.float64)
            if cov_c.ndim == 0:
                cov_c = cov_c.reshape(1, 1)
            cov_c += regularization * np.eye(dim, dtype=np.float64)

        # Cholesky decomposition for efficient sampling.
        try:
            L = np.linalg.cholesky(cov_c)
        except np.linalg.LinAlgError:
            # Fallback to eigendecomposition if Cholesky fails.
            eigvals, eigvecs = np.linalg.eigh(cov_c)
            eigvals = np.maximum(eigvals, regularization)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        z = rng.standard_normal((n_anchors, dim))
        synthetic = mu_c[None, :] + z @ L.T

        augmented_phi.append(synthetic.astype(np.float64))
        augmented_labels.append(
            np.full(n_anchors, class_label, dtype=object)
        )

    return (
        np.vstack(augmented_phi),
        np.concatenate(augmented_labels),
    )
