from __future__ import annotations

import math

import numpy as np
try:
    from numba import njit, prange
except ImportError:  # pragma: no cover - exercised in minimal deployments
    prange = range

    def njit(*decorator_args, **decorator_kwargs):
        """No-op compatibility decorator when optional Numba is absent."""
        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1:
            return decorator_args[0]

        def decorate(function):
            return function

        return decorate


_EPS = 1e-12
_UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


@njit(cache=True, fastmath=True)
def ewma_covariance_jit(values: np.ndarray, halflife: float) -> np.ndarray:
    """
    JIT EWMA covariance using the same normalized exponential weights and
    effective-sample correction as the NumPy implementation.
    """
    n_obs, n_assets = values.shape

    if n_obs < 2:
        raise ValueError("At least two observations are required.")
    if n_assets < 2:
        raise ValueError("At least two assets are required.")
    if halflife <= 0.0:
        raise ValueError("halflife must be positive.")

    decay = math.exp(math.log(0.5) / halflife)

    weights = np.empty(n_obs, dtype=np.float64)
    total_weight = 0.0
    w = 1.0

    # Newest observation gets weight 1; walk backward geometrically.
    for rev in range(n_obs):
        t = n_obs - 1 - rev
        weights[t] = w
        total_weight += w
        w *= decay

    for t in range(n_obs):
        weights[t] /= total_weight

    mean = np.zeros(n_assets, dtype=np.float64)
    for t in range(n_obs):
        wt = weights[t]
        for j in range(n_assets):
            mean[j] += wt * values[t, j]

    sum_w2 = 0.0
    for t in range(n_obs):
        sum_w2 += weights[t] * weights[t]

    denom = 1.0 - sum_w2
    if denom < _EPS:
        denom = _EPS

    cov = np.zeros((n_assets, n_assets), dtype=np.float64)

    for i in range(n_assets):
        for j in range(i, n_assets):
            acc = 0.0
            for t in range(n_obs):
                di = values[t, i] - mean[i]
                dj = values[t, j] - mean[j]
                acc += weights[t] * di * dj

            value = acc / denom
            cov[i, j] = value
            cov[j, i] = value

    return cov


@njit(cache=True, fastmath=True)
def pca_residual_reconstruction_jit(
    eigvals_desc: np.ndarray,
    eigvecs_desc: np.ndarray,
    original_diag: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Reconstruct low-rank PCA covariance plus diagonal residual variance.

    The eigendecomposition remains NumPy/LAPACK; this JIT kernel accelerates
    the dense retained-factor reconstruction and residual shrinkage step.
    """
    n_assets = eigvecs_desc.shape[0]

    if k < 1:
        k = 1
    if k > eigvals_desc.shape[0]:
        k = eigvals_desc.shape[0]

    out = np.zeros((n_assets, n_assets), dtype=np.float64)

    for factor in range(k):
        lam = eigvals_desc[factor]
        if lam <= 0.0:
            continue

        for i in range(n_assets):
            vi = eigvecs_desc[i, factor]
            for j in range(i, n_assets):
                contribution = lam * vi * eigvecs_desc[j, factor]
                out[i, j] += contribution
                if i != j:
                    out[j, i] += contribution

    # Preserve the original marginal variances through idiosyncratic diagonal
    # residuals, matching the previous PCA estimator semantics.
    for i in range(n_assets):
        residual = original_diag[i] - out[i, i]
        if residual < _EPS:
            residual = _EPS
        out[i, i] += residual

    return out


@njit(cache=True, fastmath=True)
def sparse_factor_shrinkage_jit(
    loadings: np.ndarray,
    threshold: float,
    shrinkage: float,
) -> np.ndarray:
    """
    Soft-threshold and shrink factor loadings.

    threshold is an absolute loading threshold. shrinkage in [0, 1] controls
    how aggressively surviving loadings are attenuated after thresholding.
    """
    n_assets, n_factors = loadings.shape
    out = np.empty_like(loadings)

    if threshold < 0.0:
        threshold = 0.0
    if shrinkage < 0.0:
        shrinkage = 0.0
    if shrinkage > 1.0:
        shrinkage = 1.0

    scale = 1.0 - shrinkage

    for i in range(n_assets):
        for j in range(n_factors):
            x = loadings[i, j]
            ax = abs(x)

            if ax <= threshold:
                out[i, j] = 0.0
            else:
                shrunk = (ax - threshold) * scale
                out[i, j] = shrunk if x >= 0.0 else -shrunk

    return out


@njit(cache=True, fastmath=True)
def sparse_factor_covariance_jit(
    sparse_loadings: np.ndarray,
    idiosyncratic_variance: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct B B' + D from sparse factor loadings and idiosyncratic risk.
    """
    n_assets, n_factors = sparse_loadings.shape
    cov = np.zeros((n_assets, n_assets), dtype=np.float64)

    for i in range(n_assets):
        for j in range(i, n_assets):
            acc = 0.0
            for factor in range(n_factors):
                acc += sparse_loadings[i, factor] * sparse_loadings[j, factor]

            cov[i, j] = acc
            cov[j, i] = acc

    for i in range(n_assets):
        residual = idiosyncratic_variance[i]
        if residual < _EPS:
            residual = _EPS
        cov[i, i] += residual

    return cov


@njit(cache=True)
def _splitmix64_next(state: np.uint64) -> tuple[np.uint64, np.uint64]:
    """Small deterministic PRNG suitable for bootstrap-index generation."""
    state = np.uint64(state + np.uint64(0x9E3779B97F4A7C15))
    z = state
    z = np.uint64((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9))
    z = np.uint64((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB))
    z = np.uint64(z ^ (z >> np.uint64(31)))
    return state, z


@njit(cache=True)
def block_bootstrap_indices_jit(
    n_obs: int,
    block_size: int,
    n_bootstraps: int,
    seed: int,
) -> np.ndarray:
    """
    Circular moving-block bootstrap index generation.

    Every row contains n_obs indices. Blocks wrap around the sample boundary,
    eliminating branch-heavy Python list construction.
    """
    if n_obs <= 0:
        raise ValueError("n_obs must be positive.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if n_bootstraps <= 0:
        raise ValueError("n_bootstraps must be positive.")

    indices = np.empty((n_bootstraps, n_obs), dtype=np.int64)
    state = np.uint64(seed if seed >= 0 else -seed)
    state = np.uint64(state + np.uint64(0xD1B54A32D192ED03))

    for b in range(n_bootstraps):
        pos = 0

        while pos < n_obs:
            state, rnd = _splitmix64_next(state)
            start = int(rnd % np.uint64(n_obs))

            take = block_size
            if pos + take > n_obs:
                take = n_obs - pos

            for k in range(take):
                indices[b, pos + k] = (start + k) % n_obs

            pos += take

    return indices

@njit(cache=True, fastmath=True)
def ewma_covariance_jit_blas(values: np.ndarray, halflife: float) -> np.ndarray:
    """
    BLAS-backed JIT EWMA covariance. Numba removes Python dispatch and weight
    construction overhead while matrix multiplication is delegated to LAPACK/BLAS.
    """
    n_obs, n_assets = values.shape
    if n_obs < 2:
        raise ValueError("At least two observations are required.")
    if n_assets < 2:
        raise ValueError("At least two assets are required.")
    if halflife <= 0.0:
        raise ValueError("halflife must be positive.")

    decay = math.exp(math.log(0.5) / halflife)
    weights = np.empty(n_obs, dtype=np.float64)
    total_weight = 0.0
    w = 1.0
    for rev in range(n_obs):
        t = n_obs - 1 - rev
        weights[t] = w
        total_weight += w
        w *= decay
    weights /= total_weight

    mean = weights @ values
    centered = values - mean
    sqrt_w = np.sqrt(weights)
    weighted = centered * sqrt_w.reshape((n_obs, 1))

    sum_w2 = weights @ weights
    denom = 1.0 - sum_w2
    if denom < _EPS:
        denom = _EPS

    return (weighted.T @ weighted) / denom

@njit(cache=True, fastmath=True, parallel=True)
def pca_residual_reconstruction_parallel_jit(
    eigvals_desc: np.ndarray,
    eigvecs_desc: np.ndarray,
    original_diag: np.ndarray,
    k: int,
) -> np.ndarray:
    n_assets = eigvecs_desc.shape[0]
    if k < 1:
        k = 1
    if k > eigvals_desc.shape[0]:
        k = eigvals_desc.shape[0]
    out = np.zeros((n_assets, n_assets), dtype=np.float64)
    for i in prange(n_assets):
        for j in range(n_assets):
            acc = 0.0
            for factor in range(k):
                lam = eigvals_desc[factor]
                if lam > 0.0:
                    acc += lam * eigvecs_desc[i, factor] * eigvecs_desc[j, factor]
            out[i, j] = acc
        residual = original_diag[i] - out[i, i]
        if residual < _EPS:
            residual = _EPS
        out[i, i] += residual
    return out

@njit(cache=True, fastmath=True)
def pca_residual_reconstruction_blas_jit(
    eigvals_desc: np.ndarray,
    eigvecs_desc: np.ndarray,
    original_diag: np.ndarray,
    k: int,
) -> np.ndarray:
    n_assets = eigvecs_desc.shape[0]
    if k < 1:
        k = 1
    if k > eigvals_desc.shape[0]:
        k = eigvals_desc.shape[0]
    scaled = eigvecs_desc[:, :k] * np.sqrt(np.maximum(eigvals_desc[:k], 0.0))
    out = scaled @ scaled.T
    for i in range(n_assets):
        residual = original_diag[i] - out[i, i]
        if residual < _EPS:
            residual = _EPS
        out[i, i] += residual
    return out


@njit(cache=True, fastmath=True)
def sparse_factor_covariance_blas_jit(
    sparse_loadings: np.ndarray,
    idiosyncratic_variance: np.ndarray,
) -> np.ndarray:
    """BLAS-backed sparse-factor reconstruction inside cached JIT code."""
    cov = sparse_loadings @ sparse_loadings.T
    n_assets = cov.shape[0]
    for i in range(n_assets):
        residual = idiosyncratic_variance[i]
        if residual < _EPS:
            residual = _EPS
        cov[i, i] += residual
    return cov
