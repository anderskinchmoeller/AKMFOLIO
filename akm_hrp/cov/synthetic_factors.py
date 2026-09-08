from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from akm_hrp.cov.jit_kernels import (
    sparse_factor_covariance_blas_jit,
    sparse_factor_shrinkage_jit,
)


_EPS = 1e-12


@dataclass(frozen=True)
class SparseFactorConfig:
    explained_variance: float = 0.80
    loading_threshold: float = 0.025
    loading_shrinkage: float = 0.15
    max_factors: int | None = None


def sparse_factor_covariance(
    returns: pd.DataFrame,
    config: SparseFactorConfig | None = None,
) -> pd.DataFrame:
    """
    Sparse PCA-factor covariance estimator with Numba-accelerated loading
    shrinkage and covariance reconstruction.

    Eigensolver work stays in NumPy/LAPACK. Numba accelerates the O(NK) soft
    thresholding and O(N^2 K) covariance reconstruction.
    """
    cfg = config or SparseFactorConfig()

    if not 0.0 < cfg.explained_variance <= 1.0:
        raise ValueError("explained_variance must be in (0, 1].")
    if cfg.loading_threshold < 0.0:
        raise ValueError("loading_threshold must be non-negative.")
    if not 0.0 <= cfg.loading_shrinkage <= 1.0:
        raise ValueError("loading_shrinkage must be in [0, 1].")

    x = (
        returns.copy()
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )

    means = x.mean(axis=0)
    x = x.loc[:, means.notna()]
    means = means.loc[x.columns]
    x = x.fillna(means)

    std = x.std(ddof=1)
    x = x.loc[:, std.notna() & (std > _EPS)]

    if x.shape[1] < 2:
        raise ValueError("Too few usable assets for sparse factor covariance.")

    values = x.to_numpy(dtype=np.float64)
    sample_cov = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    sample_cov = 0.5 * (sample_cov + sample_cov.T)

    eigvals, eigvecs = np.linalg.eigh(sample_cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]

    total = max(float(eigvals.sum()), _EPS)
    cumulative = np.cumsum(eigvals) / total
    k = int(np.searchsorted(cumulative, cfg.explained_variance, side="left") + 1)

    if cfg.max_factors is not None:
        k = min(k, max(1, int(cfg.max_factors)))
    k = max(1, min(k, len(eigvals)))

    # B B' reproduces the retained PCA factor covariance.
    loadings = eigvecs[:, :k] * np.sqrt(eigvals[:k])[None, :]

    sparse_loadings = sparse_factor_shrinkage_jit(
        np.ascontiguousarray(loadings, dtype=np.float64),
        float(cfg.loading_threshold),
        float(cfg.loading_shrinkage),
    )

    factor_diag = np.sum(sparse_loadings * sparse_loadings, axis=1)
    idio = np.clip(np.diag(sample_cov) - factor_diag, _EPS, None)

    cov = sparse_factor_covariance_blas_jit(
        sparse_loadings,
        np.asarray(idio, dtype=np.float64),
    )

    return pd.DataFrame(cov, index=x.columns, columns=x.columns)
