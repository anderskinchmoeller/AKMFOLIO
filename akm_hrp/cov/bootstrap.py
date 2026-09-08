from __future__ import annotations

import numpy as np
import pandas as pd

from akm_hrp.cov.jit_kernels import block_bootstrap_indices_jit


def generate_block_bootstrap_indices(
    n_obs: int,
    block_size: int,
    n_bootstraps: int,
    seed: int = 42,
) -> np.ndarray:
    """
    Numba-accelerated circular moving-block bootstrap index generator.
    """
    return block_bootstrap_indices_jit(
        int(n_obs),
        int(block_size),
        int(n_bootstraps),
        int(seed),
    )


def block_bootstrap_returns(
    returns: pd.DataFrame,
    block_size: int,
    n_bootstraps: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return both bootstrap indices and the sampled return tensor with shape:
        (n_bootstraps, n_obs, n_assets)
    """
    values = returns.to_numpy(dtype=np.float64)
    indices = generate_block_bootstrap_indices(
        len(returns),
        block_size,
        n_bootstraps,
        seed,
    )
    samples = values[indices]
    return indices, samples
