from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import zscore

from akm_hrp.signals.shrinkage import bayesian_shrink_signal


_EPS = 1e-12


def winsorize(series: pd.Series, z: float = 3.0) -> pd.Series:
    """
    Winsorize extreme cross-sectional values at +/- z standard deviations.
    """
    s = (
        series.copy()
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )

    if s.isna().all():
        return pd.Series(0.0, index=s.index, dtype=float)

    s = s.fillna(0.0)

    std = float(s.std(ddof=1))

    if not np.isfinite(std) or std <= _EPS:
        return s * 0.0

    mean = float(s.mean())
    zs = (s - mean) / std

    upper = mean + z * std
    lower = mean - z * std

    return s.clip(lower=lower, upper=upper)


def compute_momentum(
    returns: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """
    Cross-sectional momentum over a given horizon.
    """
    if len(returns) < horizon:
        return pd.Series(0.0, index=returns.columns)

    r = returns.iloc[-horizon:]
    return (1.0 + r).prod() - 1.0


def compute_reversal(
    returns: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """
    Short-term reversal signal.
    """
    if len(returns) < horizon:
        return pd.Series(0.0, index=returns.columns)

    r = returns.iloc[-horizon:].sum()
    return -r


def compute_volatility(
    returns: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """
    Low-volatility signal: lower realized volatility receives a higher score.
    """
    if len(returns) < horizon:
        return pd.Series(0.0, index=returns.columns)

    vol = returns.iloc[-horizon:].std()
    return -vol


def compute_carry(
    returns: pd.DataFrame,
) -> pd.Series:
    """
    Simple carry proxy: average weekly return.
    """
    if len(returns) < 4:
        return pd.Series(0.0, index=returns.columns)

    return returns.mean()


def normalize_cross_section(
    series: pd.Series,
) -> pd.Series:
    """
    Z-score normalize cross-sectionally.
    """
    s = (
        series.copy()
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    std = float(s.std(ddof=1))

    if not np.isfinite(std) or std <= _EPS:
        return s * 0.0

    return pd.Series(
        zscore(s, nan_policy="omit"),
        index=s.index,
        dtype=float,
    ).fillna(0.0)


def combine_signals(
    signals: dict[str, pd.Series],
) -> pd.Series:
    """
    Combine multiple normalized signals by simple average.
    """
    if not signals:
        raise ValueError("signals is empty.")

    df = pd.DataFrame(signals)
    return df.mean(axis=1)


def compute_regime_confidence(
    returns: pd.DataFrame,
) -> float:
    """
    Regime confidence:
      - high volatility -> lower confidence
      - stable / stronger common correlation -> higher confidence
    """
    if len(returns) < 52:
        return 0.5

    vol = float(returns.std().mean())
    corr = float(returns.corr().abs().mean().mean())

    if not np.isfinite(vol):
        vol = 0.0

    if not np.isfinite(corr):
        corr = 0.0

    vol_score = np.exp(-vol * 10.0)
    corr_score = corr

    confidence = 0.5 * vol_score + 0.5 * corr_score

    return float(np.clip(confidence, 0.25, 1.0))


def alpha_stack(
    returns: pd.DataFrame,
    winsor_z: float = 3.0,
    bayes_lambda: float = 0.50,
    horizons: tuple[int, ...] = (4, 13, 26, 52),
    *,
    shrinkage: float | None = None,
) -> pd.Series:
    """
    Multi-horizon alpha stack:
      - momentum: 4 / 13 / 26 / 52 weeks
      - short-term reversal
      - low volatility
      - carry
      - Bayesian shrinkage of the combined alpha
      - regime-confidence gating

    Bayesian shrinkage:
        posterior = score / (1 + lambda * vol(score))

    Notes
    -----
    The old implementation applied a fixed linear shrinkage to every
    normalized component. That made signal amplitude insensitive to the
    current universe-level noise.

    Shrinkage is now applied once to the combined cross-sectional alpha.
    This preserves relative rankings while automatically reducing absolute
    signal strength when combined alpha dispersion becomes large.

    `shrinkage=` is retained as a backwards-compatible alias for
    `bayes_lambda=`.
    """
    if shrinkage is not None:
        bayes_lambda = float(shrinkage)

    if bayes_lambda < 0:
        raise ValueError("bayes_lambda must be non-negative.")

    signals: dict[str, pd.Series] = {}

    for h in horizons:
        signals[f"mom_{h}"] = compute_momentum(
            returns,
            h,
        )

    signals["rev_4"] = compute_reversal(
        returns,
        4,
    )

    signals["low_vol_26"] = compute_volatility(
        returns,
        26,
    )

    signals["carry"] = compute_carry(
        returns,
    )

    # Normalize and winsorize each component, but do NOT apply fixed
    # per-signal linear shrinkage.
    for name in list(signals):
        s = normalize_cross_section(
            signals[name]
        )
        s = winsorize(
            s,
            z=winsor_z,
        )
        signals[name] = s

    raw_alpha = combine_signals(
        signals
    )

    posterior_alpha = bayesian_shrink_signal(
        raw_alpha,
        lambda_=bayes_lambda,
    )

    confidence = compute_regime_confidence(
        returns
    )

    final_alpha = posterior_alpha * confidence

    return (
        final_alpha
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
