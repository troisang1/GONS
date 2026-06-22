"""Core spectral anomaly-scoring primitive (NumPy-only, framework-agnostic).

A closed-form discriminant-vector anomaly scorer. This primitive backs the
optional gate mechanism, which is disabled in GONS; package-specific adapters
live in `bcmrnfst.projection.dvmad_two_tailed`.
"""

from __future__ import annotations

import numpy as np

try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False


class DVMADCore:
    """Closed-form spectral one-class anomaly detector.

    Parameters
    ----------
    mode : {"min", "max", "both"}, default="both"
        Two-tailed eigenvector selection. ``"both"`` keeps the smallest and
        largest discriminant directions; ``"min"`` / ``"max"`` keep only one
        tail.
    eps : float, default=0.1
        Tail threshold in (0, 0.5). Eigenvalues with ``< eps`` (low tail) or
        ``> 1 - eps`` (high tail) are kept.
    artificial_mode : {"max", "min", "farthest"}, default="max"
        How to construct the single artificial reference point used to make
        the within-class scatter non-degenerate in the one-class regime.
    use_faiss : bool, default=True
        Use FAISS for exact NN distance lookup at predict time, when
        available. Falls back to a chunked NumPy implementation otherwise.
    faiss_float32 : bool, default=True
        Cast projected features to float32 (required by FAISS).
    chunk_size : int, default=4096
        Row chunk size for the NumPy NN fallback.
    """

    def __init__(
        self,
        mode: str = "both",
        eps: float = 0.1,
        artificial_mode: str = "max",
        use_faiss: bool = True,
        faiss_float32: bool = True,
        chunk_size: int = 4096,
    ):
        if mode not in {"min", "max", "both"}:
            raise ValueError("mode must be one of {'min', 'max', 'both'}")
        if artificial_mode not in {"max", "min", "farthest"}:
            raise ValueError(
                "artificial_mode must be one of {'max', 'min', 'farthest'}"
            )
        if not (0.0 < eps < 0.5):
            raise ValueError("eps must lie in (0, 0.5) for two-tailed selection")

        self.mode = mode
        self.eps = float(eps)
        self.artificial_mode = artificial_mode
        self.use_faiss = bool(use_faiss) and FAISS_AVAILABLE
        self.faiss_float32 = bool(faiss_float32)
        self.chunk_size = int(chunk_size)

        self.npd_ = None
        self.basepoint_X_ = None
        self.filtered_eigvals_ = None
        self.faiss_index_ = None

    def _construct_reference_point(self, X: np.ndarray) -> np.ndarray:
        if self.artificial_mode == "max":
            return np.max(X, axis=0)
        if self.artificial_mode == "min":
            return np.min(X, axis=0)
        mu = np.mean(X, axis=0)
        idx = int(np.argmax(np.linalg.norm(X - mu, axis=1)))
        return X[idx]

    def compute_discriminants(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        classes = np.unique(y)
        mean_total = X.mean(axis=0)

        P_W_blocks = []
        for c in classes:
            Xc = X[y == c]
            mc = Xc.mean(axis=0)
            P_W_blocks.append((Xc - mc).T)
        P_W = np.hstack(P_W_blocks)

        P_S = (X - mean_total).T
        S_S = P_S @ P_S.T

        eigvals_S, Q_S = np.linalg.eigh(S_S)
        D_pinv = np.linalg.pinv(np.diag(eigvals_S))

        M = P_W @ P_W.T
        T = D_pinv @ (Q_S.T @ M @ Q_S)

        eigvals, eigvecs = np.linalg.eigh(T)

        eps = self.eps
        if self.mode == "max":
            mask = eigvals > (1.0 - eps)
        elif self.mode == "min":
            mask = eigvals < eps
        else:
            mask = (eigvals < eps) | (eigvals > (1.0 - eps))

        if not np.any(mask):
            mask = np.ones_like(eigvals, dtype=bool)

        self.filtered_eigvals_ = eigvals[mask]
        return Q_S @ eigvecs[:, mask]

    def fit(self, X_train: np.ndarray, y_train: np.ndarray | None = None):
        X_train = np.asarray(X_train)
        n, _ = X_train.shape

        if y_train is None:
            y_train = np.zeros(n, dtype=int)
        else:
            y_train = np.asarray(y_train)

        x_ref = self._construct_reference_point(X_train)
        new_label = (int(np.max(y_train)) + 1) if y_train.size else 1

        X_aug = np.vstack([X_train, x_ref])
        y_aug = np.hstack([y_train, new_label])

        self.npd_ = self.compute_discriminants(X_aug, y_aug)

        Z_aug = X_aug @ self.npd_
        if self.faiss_float32:
            Z_aug = Z_aug.astype(np.float32)
        self.basepoint_X_ = Z_aug

        if self.use_faiss:
            m = self.basepoint_X_.shape[1]
            index = faiss.IndexFlatL2(m)
            index.add(self.basepoint_X_.astype(np.float32))
            self.faiss_index_ = index

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.npd_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return np.asarray(X) @ self.npd_

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.basepoint_X_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        Z = self.transform(X_test)
        if self.faiss_float32:
            Z = Z.astype(np.float32)

        if self.use_faiss and self.faiss_index_ is not None:
            D, _ = self.faiss_index_.search(Z.astype(np.float32), k=1)
            return np.sqrt(D[:, 0])

        return self._min_l2_to_set_chunked(
            Z, self.basepoint_X_, chunk_size=self.chunk_size
        )

    @staticmethod
    def _min_l2_to_set_chunked(
        A: np.ndarray, B: np.ndarray, chunk_size: int = 4096
    ) -> np.ndarray:
        A = np.asarray(A)
        B = np.asarray(B)
        nA = A.shape[0]
        norm_B = np.sum(B ** 2, axis=1)[None, :]
        out = np.empty(nA, dtype=A.dtype)
        for s in range(0, nA, chunk_size):
            e = min(s + chunk_size, nA)
            Ac = A[s:e]
            norm_A = np.sum(Ac ** 2, axis=1)[:, None]
            dot = Ac @ B.T
            dist2 = norm_A + norm_B - 2.0 * dot
            out[s:e] = np.sqrt(np.maximum(dist2.min(axis=1), 0.0))
        return out

    def convergence_stats(self, X: np.ndarray) -> dict:
        if self.basepoint_X_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        centroid = self.basepoint_X_.mean(axis=0)
        Z = self.transform(X)
        if self.faiss_float32:
            Z = Z.astype(np.float32)
        dists = np.linalg.norm(Z - centroid, axis=1)
        sq_train = np.mean(np.sum((self.basepoint_X_ - centroid) ** 2, axis=1))
        sq_proj = np.mean(np.sum((Z - centroid) ** 2, axis=1))
        return {
            "mean": float(dists.mean()),
            "max": float(dists.max()),
            "min": float(dists.min()),
            "std": float(dists.std()),
            "convergence": (
                float("nan") if sq_train == 0 else float(sq_proj / sq_train)
            ),
            "n_selected": (
                None
                if self.filtered_eigvals_ is None
                else int(self.filtered_eigvals_.size)
            ),
        }
