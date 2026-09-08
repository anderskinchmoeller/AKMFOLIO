from __future__ import annotations

import numpy as np
import pandas as pd

from akm_hrp.cov.jit_kernels import pca_residual_reconstruction_blas_jit


_EPS = 1e-12


def pca_covariance_from_sample(
    sample_cov: np.ndarray,
    explained_variance: float = 0.90,
) -> tuple[np.ndarray, int]:
    """
    PCA covariance shrinkage from a sample covariance matrix.

    NumPy/LAPACK performs the eigendecomposition; the retained-factor
    reconstruction and residual-diagonal shrinkage run inside cached Numba.
    """
    if not 0.0 < explained_variance <= 1.0:
        raise ValueError("explained_variance must be in (0, 1].")

    cov = np.asarray(sample_cov, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]

    eigvals = np.ascontiguousarray(
        np.clip(eigvals[order], 0.0, None),
        dtype=np.float64,
    )
    eigvecs = np.ascontiguousarray(eigvecs[:, order], dtype=np.float64)

    total = max(float(eigvals.sum()), _EPS)
    cumulative = np.cumsum(eigvals) / total
    k = int(np.searchsorted(cumulative, explained_variance, side="left") + 1)
    k = max(1, min(k, len(eigvals)))

    out = pca_residual_reconstruction_blas_jit(
        eigvals,
        eigvecs,
        np.ascontiguousarray(np.diag(cov), dtype=np.float64),
        k,
    )

    return 0.5 * (out + out.T), k


def pca_covariance(
    returns: pd.DataFrame,
    explained_variance: float = 0.90,
) -> tuple[pd.DataFrame, int]:
    """Estimate PCA covariance directly from returns."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    x = returns.copy().astype(float).replace([np.inf, -np.inf], np.nan)
    means = x.mean(axis=0)
    x = x.loc[:, means.notna()]
    means = means.loc[x.columns]
    x = x.fillna(means)

    sample_cov = np.atleast_2d(
        np.cov(x.to_numpy(dtype=np.float64), rowvar=False, ddof=1)
    )
    cov, k = pca_covariance_from_sample(sample_cov, explained_variance)
    return pd.DataFrame(cov, index=x.columns, columns=x.columns), k
