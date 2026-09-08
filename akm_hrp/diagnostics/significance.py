from __future__ import annotations

"""Selection-aware statistical diagnostics for strategy returns."""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

_EPS = 1e-12
_EULER_GAMMA = 0.5772156649015329


def sharpe_significance(
    returns: pd.Series,
    *,
    number_of_trials: int = 1,
    periods_per_year: float = 52.0,
) -> dict[str, float]:
    """Return Probabilistic and Deflated Sharpe diagnostics.

    ``number_of_trials`` must be the honest count of strategies or parameter
    variants searched, including discarded attempts. The Deflated Sharpe
    benchmark is the expected best null Sharpe across those trials and the
    probability accounts for observed skewness and kurtosis.
    """

    values = (
        pd.Series(returns, dtype=float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    trials = max(int(number_of_trials), 1)
    if len(values) < 3:
        return {
            "probabilistic_sharpe_ratio": 0.0,
            "deflated_sharpe_ratio": 0.0,
            "deflated_sharpe_p_value": 1.0,
            "deflated_sharpe_benchmark": 0.0,
            "significance_trials": float(trials),
        }
    volatility = float(np.std(values, ddof=1))
    if not np.isfinite(volatility) or volatility <= _EPS:
        return {
            "probabilistic_sharpe_ratio": 0.0,
            "deflated_sharpe_ratio": 0.0,
            "deflated_sharpe_p_value": 1.0,
            "deflated_sharpe_benchmark": 0.0,
            "significance_trials": float(trials),
        }

    weekly_sharpe = float(np.mean(values) / volatility)
    sample_skew = float(skew(values, bias=False))
    sample_kurtosis = float(kurtosis(values, fisher=False, bias=False))
    variance = (
        1.0
        - sample_skew * weekly_sharpe
        + 0.25 * (sample_kurtosis - 1.0) * weekly_sharpe**2
    ) / (len(values) - 1.0)
    sharpe_standard_error = float(np.sqrt(max(variance, _EPS)))

    if trials == 1:
        null_benchmark = 0.0
    else:
        expected_max_z = (
            (1.0 - _EULER_GAMMA) * norm.ppf(1.0 - 1.0 / trials)
            + _EULER_GAMMA
            * norm.ppf(1.0 - 1.0 / (trials * np.e))
        )
        null_benchmark = float(sharpe_standard_error * expected_max_z)

    psr = float(norm.cdf(weekly_sharpe / sharpe_standard_error))
    dsr = float(
        norm.cdf((weekly_sharpe - null_benchmark) / sharpe_standard_error)
    )
    return {
        "probabilistic_sharpe_ratio": psr,
        "deflated_sharpe_ratio": dsr,
        "deflated_sharpe_p_value": 1.0 - dsr,
        "deflated_sharpe_benchmark": (
            null_benchmark * np.sqrt(periods_per_year)
        ),
        "significance_trials": float(trials),
    }


def newey_west_mean_test(
    returns: pd.Series,
    *,
    lags: int | None = None,
    periods_per_year: float = 52.0,
) -> dict[str, float]:
    """HAC t-test for a return or benchmark-active-return mean."""

    values = (
        pd.Series(returns, dtype=float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    count = len(values)
    if count < 3:
        return {"annualized_mean": 0.0, "hac_t_stat": 0.0, "hac_p_value": 1.0}
    lag_count = (
        min(count - 1, max(1, int(np.floor(4.0 * (count / 100.0) ** (2.0 / 9.0)))))
        if lags is None
        else min(count - 1, max(0, int(lags)))
    )
    centred = values - values.mean()
    long_run_variance = float(centred @ centred / count)
    for lag in range(1, lag_count + 1):
        covariance = float(centred[lag:] @ centred[:-lag] / count)
        weight = 1.0 - lag / (lag_count + 1.0)
        long_run_variance += 2.0 * weight * covariance
    standard_error = np.sqrt(max(long_run_variance, _EPS) / count)
    statistic = float(values.mean() / standard_error)
    return {
        "annualized_mean": float(values.mean() * periods_per_year),
        "hac_t_stat": statistic,
        "hac_p_value": float(2.0 * norm.sf(abs(statistic))),
    }
