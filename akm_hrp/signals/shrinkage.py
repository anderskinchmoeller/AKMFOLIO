from __future__ import annotations

import numpy as np
import pandas as pd


_EPS = 1e-12


def bayesian_shrink_signal(
    score: pd.Series,
    lambda_: float = 0.50,
) -> pd.Series:
    """
    Bayesian-style shrinkage of a cross-sectional alpha score.

        posterior = score / (1 + lambda * vol(score))

    where vol(score) is the current cross-sectional standard deviation.

    Higher cross-sectional dispersion therefore produces stronger shrinkage,
    reducing signal amplitude in noisy / unstable universes.
    """
    if not isinstance(score, pd.Series):
        raise TypeError("score must be a pandas Series.")

    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative.")

    s = (
        score.copy()
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )

    if s.isna().all():
        return pd.Series(0.0, index=s.index, dtype=float)

    # Missing alpha is neutral.
    s = s.fillna(0.0)

    vol = float(s.std(ddof=1))

    if not np.isfinite(vol) or vol <= _EPS:
        return s

    shrink_factor = 1.0 / (1.0 + float(lambda_) * vol)

    return s * shrink_factor


def shrink_signal(
    series: pd.Series,
    shrinkage: float = 0.30,
) -> pd.Series:
    """
    Backward-compatible alias.

    The old implementation used fixed linear shrinkage:
        (1 - shrinkage) * score

    It now interprets `shrinkage` as the Bayesian lambda parameter.
    """
    return bayesian_shrink_signal(
        series,
        lambda_=shrinkage,
    )
