from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from akm_hrp.cov.ensemble import covariance_to_correlation
from akm_hrp.hrp.allocation import hrp_allocate
from akm_hrp.hrp.trees import build_tree_ensemble
from akm_hrp.overlay.bounds import apply_bounds

_EPS = 1e-12
_WEEKS_PER_YEAR = 52.0


@dataclass(frozen=True)
class HRPAlphaV1Config:
    """Configuration for the deliberately simple, retail-oriented HRP model."""

    # Approximately two years and six months of weekly observations.  The two
    # shrunk estimates are blended rather than selecting one fragile horizon.
    long_covariance_weeks: int = 104
    short_covariance_weeks: int = 26
    long_covariance_weight: float = 0.75

    linkage_methods: tuple[str, ...] = ("average", "complete", "ward")
    risk_contribution_cap: float = 0.20

    momentum_fast_weeks: int = 26
    momentum_slow_weeks: int = 52
    momentum_skip_recent_weeks: int = 4
    momentum_fast_weight: float = 0.50
    momentum_tilt_strength: float = 0.25
    negative_trend_multiplier: float = 0.50

    # Positions whose target change is smaller than this amount are frozen at
    # the previous target.  The engine separately controls total drift and L1
    # turnover of the actually executed rebalance.
    no_trade_band: float = 0.02
    no_trade_band_equal_weight_fraction: float = 0.50

    min_weight: float = 0.0
    max_weight: float = 0.20

    # Volatility targeting is only meaningful when the input contains a real
    # cash/T-bill return column.  No synthetic zero-return asset is invented.
    annual_target_volatility: float | None = None
    cash_asset: str | None = None
    max_cash_weight: float = 1.0


@dataclass(frozen=True)
class HRPAlphaV1Diagnostics:
    eligible_asset_count: int
    risky_asset_count: int
    long_observations: int
    short_observations: int
    long_ledoit_wolf_shrinkage: float
    short_ledoit_wolf_shrinkage: float
    covariance_condition_number: float
    tree_count: int
    negative_trend_asset_count: int
    momentum_score_dispersion: float
    alpha_tilt_active_share: float
    frozen_by_no_trade_band: int
    effective_no_trade_band: float
    target_turnover_l1: float
    predicted_annual_volatility_before_scaling: float
    predicted_annual_volatility_after_scaling: float
    risky_exposure: float
    volatility_target_applied: bool
    effective_asset_count: float
    maximum_weight: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "eligible_asset_count": self.eligible_asset_count,
            "risky_asset_count": self.risky_asset_count,
            "long_observations": self.long_observations,
            "short_observations": self.short_observations,
            "long_ledoit_wolf_shrinkage": self.long_ledoit_wolf_shrinkage,
            "short_ledoit_wolf_shrinkage": self.short_ledoit_wolf_shrinkage,
            "covariance_condition_number": self.covariance_condition_number,
            "tree_count": self.tree_count,
            "negative_trend_asset_count": self.negative_trend_asset_count,
            "momentum_score_dispersion": self.momentum_score_dispersion,
            "alpha_tilt_active_share": self.alpha_tilt_active_share,
            "frozen_by_no_trade_band": self.frozen_by_no_trade_band,
            "effective_no_trade_band": self.effective_no_trade_band,
            "target_turnover_l1": self.target_turnover_l1,
            "predicted_annual_volatility_before_scaling": (
                self.predicted_annual_volatility_before_scaling
            ),
            "predicted_annual_volatility_after_scaling": (
                self.predicted_annual_volatility_after_scaling
            ),
            "risky_exposure": self.risky_exposure,
            "volatility_target_applied": int(self.volatility_target_applied),
            "effective_asset_count": self.effective_asset_count,
            "maximum_weight": self.maximum_weight,
        }


def _clean_window(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    window = (
        returns.copy()
        .astype(float)
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
    )
    if window.index.has_duplicates:
        raise ValueError("returns index contains duplicate dates.")
    valid = (
        (window.notna().sum(axis=0) >= 3)
        & window.std(axis=0, skipna=True).notna()
        & (window.std(axis=0, skipna=True) > _EPS)
    )
    window = window.loc[:, valid]
    if window.shape[1] < 2:
        raise ValueError("HRP Alpha v1 requires at least two varying assets.")
    # The engine already screens interior missingness.  Mean filling here is a
    # neutral dense-matrix repair, not a return forecast.
    return window.fillna(window.mean(axis=0))


def _ledoit_wolf_covariance(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    estimator = LedoitWolf(assume_centered=False).fit(
        returns.to_numpy(dtype=float)
    )
    values = np.asarray(estimator.covariance_, dtype=float)
    values = 0.5 * (values + values.T)
    return (
        pd.DataFrame(values, index=returns.columns, columns=returns.columns),
        float(estimator.shrinkage_),
    )


def blended_ledoit_wolf_covariance(
    returns: pd.DataFrame,
    config: HRPAlphaV1Config,
) -> tuple[pd.DataFrame, float, float, int, int]:
    """Blend long- and short-memory Ledoit-Wolf covariance estimates."""
    long_observations = min(config.long_covariance_weeks, len(returns))
    short_observations = min(config.short_covariance_weeks, len(returns))
    long_window = returns.iloc[-long_observations:]
    short_window = returns.iloc[-short_observations:]
    long_covariance, long_shrinkage = _ledoit_wolf_covariance(long_window)
    short_covariance, short_shrinkage = _ledoit_wolf_covariance(short_window)
    long_weight = float(np.clip(config.long_covariance_weight, 0.0, 1.0))
    blended = long_weight * long_covariance + (1.0 - long_weight) * short_covariance
    blended = 0.5 * (blended + blended.T)
    return (
        blended,
        long_shrinkage,
        short_shrinkage,
        long_observations,
        short_observations,
    )


def _compounded_return(
    returns: pd.DataFrame,
    lookback_weeks: int,
    skip_recent_weeks: int = 0,
) -> pd.Series:
    end = len(returns) - max(skip_recent_weeks, 0)
    start = max(0, end - lookback_weeks)
    if end <= start:
        return pd.Series(0.0, index=returns.columns)
    return (1.0 + returns.iloc[start:end]).prod(axis=0) - 1.0


def _rank_zscore(values: pd.Series) -> pd.Series:
    ranks = values.rank(method="average", pct=True)
    dispersion = float(ranks.std(ddof=0))
    if not np.isfinite(dispersion) or dispersion <= _EPS:
        return pd.Series(0.0, index=values.index)
    return (ranks - float(ranks.mean())) / dispersion


def momentum_and_trend_scores(
    returns: pd.DataFrame,
    config: HRPAlphaV1Config,
) -> tuple[pd.Series, pd.Series]:
    """Combine cross-sectional 6/12-month momentum and absolute 12m trend."""
    fast = _compounded_return(
        returns,
        config.momentum_fast_weeks,
        config.momentum_skip_recent_weeks,
    )
    slow = _compounded_return(
        returns,
        config.momentum_slow_weeks,
        config.momentum_skip_recent_weeks,
    )
    fast_weight = float(np.clip(config.momentum_fast_weight, 0.0, 1.0))
    score = fast_weight * _rank_zscore(fast) + (1.0 - fast_weight) * _rank_zscore(slow)

    # The absolute trend includes the newest observation.  It answers a
    # different question from the cross-sectional score and acts only as a
    # soft exposure penalty, never as a hard sell signal.
    absolute_trend = _compounded_return(
        returns,
        config.momentum_slow_weeks,
        skip_recent_weeks=0,
    )
    return score.astype(float), absolute_trend.astype(float)


def ensemble_hrp_weights(
    covariance: pd.DataFrame,
    config: HRPAlphaV1Config,
) -> tuple[pd.Series, int]:
    """Average HRP allocations across several plausible dendrograms."""
    correlation = covariance_to_correlation(covariance)
    assets = list(covariance.columns)
    trees = build_tree_ensemble(correlation, methods=config.linkage_methods)
    allocations = [
        hrp_allocate(
            correlation,
            covariance,
            tree,
            assets,
            risk_cap=config.risk_contribution_cap,
        ).reindex(assets)
        for tree in trees
    ]
    ensemble = pd.concat(allocations, axis=1).mean(axis=1)
    ensemble = ensemble.clip(lower=0.0)
    return ensemble / ensemble.sum(), len(trees)


def _project_with_bounds(
    weights: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> pd.Series:
    """Project onto a simplex with per-asset lower and upper bounds."""
    weights = weights.astype(float)
    lower = lower.reindex(weights.index).astype(float)
    upper = upper.reindex(weights.index).astype(float)
    if (lower > upper).any():
        raise ValueError("Lower weight bound exceeds upper bound.")
    if float(lower.sum()) > 1.0 + 1e-10 or float(upper.sum()) < 1.0 - 1e-10:
        raise ValueError("Per-asset weight bounds are infeasible.")

    values = weights.to_numpy(dtype=float)
    lo = lower.to_numpy(dtype=float)
    hi = upper.to_numpy(dtype=float)
    lambda_low = float(np.min(values - hi)) - 1.0
    lambda_high = float(np.max(values - lo)) + 1.0
    for _ in range(200):
        lagrange = 0.5 * (lambda_low + lambda_high)
        projected = np.clip(values - lagrange, lo, hi)
        if float(projected.sum()) > 1.0:
            lambda_low = lagrange
        else:
            lambda_high = lagrange
    projected = np.clip(values - 0.5 * (lambda_low + lambda_high), lo, hi)
    residual = 1.0 - float(projected.sum())
    if abs(residual) > 1e-10:
        room = hi - projected if residual > 0.0 else projected - lo
        for position in np.flatnonzero(room > _EPS):
            step = min(abs(residual), float(room[position]))
            projected[position] += step if residual > 0.0 else -step
            residual += -step if residual > 0.0 else step
            if abs(residual) <= 1e-10:
                break
    result = pd.Series(projected, index=weights.index, dtype=float)
    if not np.isclose(result.sum(), 1.0, atol=1e-9):
        raise RuntimeError("Per-asset bound projection failed.")
    return result


def _apply_no_trade_band(
    candidate: pd.Series,
    previous_weights: pd.Series | None,
    lower: pd.Series,
    upper: pd.Series,
    band: float,
) -> tuple[pd.Series, int, float]:
    if previous_weights is None or band <= 0.0:
        return _project_with_bounds(candidate, lower, upper), 0, 0.0

    previous = previous_weights.reindex(candidate.index).fillna(0.0).clip(lower=0.0)
    if float(previous.sum()) <= _EPS:
        return _project_with_bounds(candidate, lower, upper), 0, 0.0
    previous /= previous.sum()
    previous = _project_with_bounds(previous, lower, upper)
    difference = candidate - previous
    frozen = difference.abs() < band

    frozen_lower = lower.copy()
    frozen_upper = upper.copy()
    frozen_lower.loc[frozen] = previous.loc[frozen]
    frozen_upper.loc[frozen] = previous.loc[frozen]
    result = _project_with_bounds(candidate, frozen_lower, frozen_upper)
    turnover = float((result - previous).abs().sum())
    return result, int(frozen.sum()), turnover


class HRPAlphaV1Allocator:
    """Shrunk ensemble HRP with a modest momentum/trend expected-return tilt."""

    def __init__(self, config: HRPAlphaV1Config | None = None) -> None:
        self.config = config or HRPAlphaV1Config()
        self.last_diagnostics: HRPAlphaV1Diagnostics | None = None
        self.last_covariance: pd.DataFrame | None = None
        self.last_momentum_score: pd.Series | None = None
        self._previous_weights: pd.Series | None = None
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.long_covariance_weeks < 2:
            raise ValueError("long_covariance_weeks must be at least two.")
        if self.config.short_covariance_weeks < 2:
            raise ValueError("short_covariance_weeks must be at least two.")
        if not self.config.linkage_methods:
            raise ValueError("At least one linkage method is required.")
        if not 0.0 < self.config.risk_contribution_cap <= 1.0:
            raise ValueError("risk_contribution_cap must be in (0, 1].")
        if self.config.momentum_fast_weeks < 1 or self.config.momentum_slow_weeks < 1:
            raise ValueError("Momentum lookbacks must be positive.")
        if self.config.momentum_tilt_strength < 0.0:
            raise ValueError("momentum_tilt_strength cannot be negative.")
        if not 0.0 <= self.config.negative_trend_multiplier <= 1.0:
            raise ValueError("negative_trend_multiplier must be in [0, 1].")
        if self.config.no_trade_band < 0.0:
            raise ValueError("no_trade_band cannot be negative.")
        if self.config.no_trade_band_equal_weight_fraction < 0.0:
            raise ValueError("no_trade_band_equal_weight_fraction cannot be negative.")
        if self.config.min_weight < 0.0 or self.config.max_weight <= 0.0:
            raise ValueError("Invalid weight bounds.")
        if not 0.0 <= self.config.max_cash_weight <= 1.0:
            raise ValueError("max_cash_weight must be in [0, 1].")
        if (
            self.config.annual_target_volatility is not None
            and self.config.annual_target_volatility <= 0.0
        ):
            raise ValueError("annual_target_volatility must be positive.")

    def reset_state(self) -> None:
        self.last_diagnostics = None
        self.last_covariance = None
        self.last_momentum_score = None
        self._previous_weights = None

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        window = _clean_window(returns)
        all_assets = list(window.columns)
        cash_asset = self.config.cash_asset
        volatility_target_requested = (
            self.config.annual_target_volatility is not None
            and cash_asset is not None
            and cash_asset in window.columns
        )

        risky_window = (
            window.drop(columns=[cash_asset])
            if volatility_target_requested
            else window
        )
        if risky_window.shape[1] < 2:
            raise ValueError("At least two risky assets are required.")

        (
            covariance,
            long_shrinkage,
            short_shrinkage,
            long_observations,
            short_observations,
        ) = blended_ledoit_wolf_covariance(risky_window, self.config)
        base_weights, tree_count = ensemble_hrp_weights(covariance, self.config)
        momentum_score, absolute_trend = momentum_and_trend_scores(
            risky_window,
            self.config,
        )

        tilted = base_weights * np.exp(
            self.config.momentum_tilt_strength * momentum_score
        )
        negative_trend = absolute_trend < 0.0
        tilted.loc[negative_trend] *= self.config.negative_trend_multiplier
        tilted = tilted.clip(lower=0.0)
        tilted /= tilted.sum()
        alpha_tilt_active_share = 0.5 * float((tilted - base_weights).abs().sum())

        # Apply risky-asset bounds before any cash scaling.  With no explicit
        # cash asset this is the final fully invested risky portfolio.
        risky_weights = apply_bounds(
            tilted,
            min_weight=self.config.min_weight,
            max_weight=self.config.max_weight,
        )
        risky_values = risky_weights.to_numpy(dtype=float)
        covariance_values = covariance.to_numpy(dtype=float)
        annual_volatility_before = float(
            np.sqrt(
                max(risky_values @ covariance_values @ risky_values, 0.0)
                * _WEEKS_PER_YEAR
            )
        )

        risky_exposure = 1.0
        if volatility_target_requested:
            target = float(self.config.annual_target_volatility)
            if annual_volatility_before > _EPS:
                risky_exposure = min(1.0, target / annual_volatility_before)
            risky_exposure = max(
                risky_exposure,
                1.0 - self.config.max_cash_weight,
            )
            candidate = pd.Series(0.0, index=all_assets, dtype=float)
            candidate.loc[risky_weights.index] = risky_exposure * risky_weights
            candidate.loc[cash_asset] = 1.0 - risky_exposure
            lower = pd.Series(self.config.min_weight, index=all_assets, dtype=float)
            upper = pd.Series(self.config.max_weight, index=all_assets, dtype=float)
            lower.loc[cash_asset] = 0.0
            upper.loc[cash_asset] = self.config.max_cash_weight
        else:
            candidate = risky_weights.reindex(all_assets).fillna(0.0)
            lower = pd.Series(self.config.min_weight, index=all_assets, dtype=float)
            upper = pd.Series(self.config.max_weight, index=all_assets, dtype=float)

        # A literal two-point band is sensible for 12–20 exposures but would
        # freeze every sub-1% position in a broad stock panel.  Cap it at half
        # an equal-weight position so the rule scales with universe breadth.
        effective_no_trade_band = min(
            self.config.no_trade_band,
            self.config.no_trade_band_equal_weight_fraction / len(candidate),
        )
        final, frozen_count, target_turnover = _apply_no_trade_band(
            candidate,
            self._previous_weights,
            lower,
            upper,
            effective_no_trade_band,
        )
        # If the no-trade projection moved the cash sleeve, report the realised
        # risky exposure rather than the pre-band proposal.
        if volatility_target_requested:
            risky_exposure = float(final.drop(index=cash_asset).sum())
        annual_volatility_after = annual_volatility_before * risky_exposure

        self._previous_weights = final.copy()
        self.last_covariance = covariance.copy()
        self.last_momentum_score = momentum_score.copy()
        self.last_diagnostics = HRPAlphaV1Diagnostics(
            eligible_asset_count=len(all_assets),
            risky_asset_count=risky_window.shape[1],
            long_observations=long_observations,
            short_observations=short_observations,
            long_ledoit_wolf_shrinkage=long_shrinkage,
            short_ledoit_wolf_shrinkage=short_shrinkage,
            covariance_condition_number=float(
                np.linalg.cond(covariance.to_numpy(dtype=float))
            ),
            tree_count=tree_count,
            negative_trend_asset_count=int(negative_trend.sum()),
            momentum_score_dispersion=float(momentum_score.std(ddof=0)),
            alpha_tilt_active_share=alpha_tilt_active_share,
            frozen_by_no_trade_band=frozen_count,
            effective_no_trade_band=effective_no_trade_band,
            target_turnover_l1=target_turnover,
            predicted_annual_volatility_before_scaling=annual_volatility_before,
            predicted_annual_volatility_after_scaling=annual_volatility_after,
            risky_exposure=risky_exposure,
            volatility_target_applied=volatility_target_requested,
            effective_asset_count=float(1.0 / np.sum(final.to_numpy() ** 2)),
            maximum_weight=float(final.max()),
        )
        return final
