from __future__ import annotations

"""A deliberately small, fixed-parameter HRP model.

The allocator implements the low-overfitting design described in the project
notes: one shrinkage estimator, one linkage rule, one momentum definition and
one weak tilt.  It intentionally has no regime classifier, linkage selection,
volatility target, leverage, or fitted hyperparameters.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from akm_hrp.cov.ensemble import covariance_to_correlation
from akm_hrp.hrp.allocation import hrp_allocate
from akm_hrp.overlay.bounds import apply_bounds

_EPS = 1e-12


@dataclass(frozen=True)
class LowOverfitHRPConfig:
    """Economically fixed choices; avoid optimizing these to a sharp maximum."""

    covariance_lookback_weeks: int = 104
    minimum_history_weeks: int = 104
    momentum_short_weeks: int = 26
    momentum_long_weeks: int = 52
    momentum_tilt: float = 0.20
    no_trade_band: float = 0.005
    min_weight: float = 0.0
    max_weight: float = 0.10
    risk_contribution_cap: float = 0.20
    linkage_method: str = "average"


@dataclass(frozen=True)
class LowOverfitHRPDiagnostics:
    covariance_observations: int
    covariance_shrinkage: float
    momentum_tilt: float
    target_month: str
    effective_asset_count: float
    maximum_weight: float
    target_turnover_l1: float
    frozen_by_monthly_schedule: bool

    def as_dict(self) -> dict[str, float | int | str]:
        return self.__dict__.copy()


def _momentum_score(
    returns: pd.DataFrame,
    short_weeks: int,
    long_weeks: int,
) -> pd.Series:
    """Cross-sectional standardized rank of average 6- and 12-month returns."""

    short_return = (1.0 + returns.iloc[-short_weeks:]).prod(axis=0) - 1.0
    long_return = (1.0 + returns.iloc[-long_weeks:]).prod(axis=0) - 1.0
    momentum = 0.5 * (short_return + long_return)
    ranks = momentum.rank(method="average", pct=True)
    centred = ranks - ranks.mean()
    scale = float(centred.std(ddof=0))
    if not np.isfinite(scale) or scale <= _EPS:
        return pd.Series(0.0, index=returns.columns)
    return centred / scale


def _freeze_small_trades(
    target: pd.Series,
    previous: pd.Series | None,
    band: float,
) -> pd.Series:
    """Keep prior targets when the proposed position change is immaterial."""

    if previous is None or band <= 0.0:
        return target
    prior = previous.reindex(target.index).fillna(0.0)
    frozen = target.where((target - prior).abs() >= band, prior)
    if float(frozen.sum()) <= _EPS:
        return target
    return frozen / frozen.sum()


class LowOverfitHRPAllocator:
    """Ledoit–Wolf HRP with one weak momentum tilt and monthly target updates."""

    def __init__(self, config: LowOverfitHRPConfig | None = None) -> None:
        self.config = config or LowOverfitHRPConfig()
        self.last_diagnostics: LowOverfitHRPDiagnostics | None = None
        self.last_covariance: pd.DataFrame | None = None
        self._previous_target: pd.Series | None = None
        self._last_target_month: pd.Period | None = None
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.covariance_lookback_weeks < 3:
            raise ValueError("covariance_lookback_weeks must be at least three.")
        if not 3 <= self.config.momentum_short_weeks <= self.config.momentum_long_weeks:
            raise ValueError("Momentum horizons must satisfy 3 <= short <= long.")
        if self.config.momentum_tilt < 0.0:
            raise ValueError("momentum_tilt cannot be negative.")
        if self.config.no_trade_band < 0.0:
            raise ValueError("no_trade_band cannot be negative.")
        if not 0.0 < self.config.max_weight <= 1.0:
            raise ValueError("max_weight must be in (0, 1].")

    def reset_state(self) -> None:
        self.last_diagnostics = None
        self.last_covariance = None
        self._previous_target = None
        self._last_target_month = None

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        window = (
            returns.copy().astype(float).sort_index().replace([np.inf, -np.inf], np.nan)
        )
        if not isinstance(window.index, pd.DatetimeIndex):
            raise TypeError("Low-overfit HRP requires a DatetimeIndex.")
        valid = (
            window.notna().sum(axis=0)
            >= max(
                self.config.minimum_history_weeks,
                self.config.momentum_long_weeks,
            )
        ) & (window.std(axis=0, skipna=True) > _EPS)
        window = window.loc[:, valid]
        if window.shape[1] < 2:
            raise ValueError("Too few assets have complete momentum history.")
        window = window.fillna(window.mean(axis=0))

        target_month = window.index[-1].to_period("M")
        if (
            self._previous_target is not None
            and self._last_target_month == target_month
        ):
            frozen = self._previous_target.reindex(window.columns).fillna(0.0)
            if float(frozen.sum()) > _EPS:
                frozen /= frozen.sum()
                frozen = apply_bounds(
                    frozen,
                    self.config.min_weight,
                    self.config.max_weight,
                )
                self._record_diagnostics(
                    frozen,
                    target_month,
                    covariance=None,
                    shrinkage=float("nan"),
                    turnover=0.0,
                    frozen=True,
                )
                return frozen

        covariance_window = window.iloc[-self.config.covariance_lookback_weeks :]
        estimator = LedoitWolf().fit(covariance_window.to_numpy(dtype=float))
        covariance = pd.DataFrame(
            estimator.covariance_,
            index=window.columns,
            columns=window.columns,
        )
        correlation = covariance_to_correlation(covariance)
        distance = np.sqrt(
            np.maximum(0.5 * (1.0 - correlation.to_numpy(dtype=float)), 0.0)
        )
        np.fill_diagonal(distance, 0.0)
        tree = linkage(
            squareform(distance, checks=False),
            method=self.config.linkage_method,
        )
        base = hrp_allocate(
            correlation,
            covariance,
            tree,
            list(covariance.columns),
            risk_cap=self.config.risk_contribution_cap,
        )
        score = _momentum_score(
            window,
            self.config.momentum_short_weeks,
            self.config.momentum_long_weeks,
        ).reindex(base.index)
        tilted = base * np.exp(self.config.momentum_tilt * score)
        tilted /= tilted.sum()
        target = _freeze_small_trades(
            tilted,
            self._previous_target,
            self.config.no_trade_band,
        )
        target = apply_bounds(
            target,
            self.config.min_weight,
            self.config.max_weight,
        )

        turnover = 0.0
        if self._previous_target is not None:
            previous = self._previous_target.reindex(target.index).fillna(0.0)
            turnover = float((target - previous).abs().sum())
        self._previous_target = target.copy()
        self._last_target_month = target_month
        self._record_diagnostics(
            target,
            target_month,
            covariance=covariance,
            shrinkage=float(estimator.shrinkage_),
            turnover=turnover,
            frozen=False,
        )
        return target

    def _record_diagnostics(
        self,
        target: pd.Series,
        target_month: pd.Period,
        *,
        covariance: pd.DataFrame | None,
        shrinkage: float,
        turnover: float,
        frozen: bool,
    ) -> None:
        values = target.to_numpy(dtype=float)
        if covariance is not None:
            self.last_covariance = covariance
        self.last_diagnostics = LowOverfitHRPDiagnostics(
            covariance_observations=self.config.covariance_lookback_weeks,
            covariance_shrinkage=shrinkage,
            momentum_tilt=self.config.momentum_tilt,
            target_month=str(target_month),
            effective_asset_count=float(1.0 / np.sum(values**2)),
            maximum_weight=float(target.max()),
            target_turnover_l1=turnover,
            frozen_by_monthly_schedule=frozen,
        )
