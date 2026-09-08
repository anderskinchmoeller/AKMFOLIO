from __future__ import annotations

import numpy as np
import pandas as pd

from akm_hrp.cov.jit_kernels import ewma_covariance_jit_blas


def ewma_covariance_array(
    values: np.ndarray,
    halflife: float = 26.0,
) -> np.ndarray:
    """Numba/BLAS-accelerated EWMA covariance for a dense return matrix."""
    x = np.ascontiguousarray(values, dtype=np.float64)
    cov = ewma_covariance_jit_blas(x, float(halflife))
    return 0.5 * (cov + cov.T)


def ewma_covariance(
    returns: pd.DataFrame,
    halflife: float = 26.0,
) -> pd.DataFrame:
    """DataFrame wrapper around the JIT EWMA estimator."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    x = returns.copy().astype(float).replace([np.inf, -np.inf], np.nan)
    means = x.mean(axis=0)
    x = x.loc[:, means.notna()]
    means = means.loc[x.columns]
    x = x.fillna(means)

    cov = ewma_covariance_array(x.to_numpy(dtype=np.float64), halflife)
    return pd.DataFrame(cov, index=x.columns, columns=x.columns)
