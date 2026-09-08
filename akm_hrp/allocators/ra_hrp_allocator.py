from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from akm_hrp.cov.ensemble import (
    AdaptiveCovarianceConfig,
    CovarianceBlendDiagnostics,
    adaptive_covariance_ensemble,
    covariance_to_correlation,
)
from akm_hrp.hrp.return_adjusted import (
    ReturnAdjustedHRPDiagnostics,
    return_adjusted_hrp_allocate,
)
from akm_hrp.hrp.trees import build_tree_ensemble
from akm_hrp.overlay.bounds import apply_bounds

_EPS = 1e-12
_WEEKS_PER_YEAR = 52.0


@dataclass(frozen=True)
class RAHRPConfig:
    linkage_method: str = "single"
    score_floor: float = 1e-4
    interpolation: float = 1.0
    mean_halflife: float | None = None
    annual_risk_free_rate: float = 0.0
    cov_ewma_halflife: float = 26.0
    cov_pca_explained_variance: float = 0.90
    min_weight: float = 0.0
    max_weight: float = 0.10


class RAHRPAllocator:
    """Project adapter for fixed-tree Return-Adjusted HRP."""

    def __init__(self, config: RAHRPConfig | None = None) -> None:
        self.config = config or RAHRPConfig()
        self.last_ra_diagnostics: ReturnAdjustedHRPDiagnostics | None = None
        self.last_covariance_diagnostics: CovarianceBlendDiagnostics | None = None

    def reset_state(self) -> None:
        self.last_ra_diagnostics = None
        self.last_covariance_diagnostics = None

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame.")
        if returns.shape[1] < 2:
            raise ValueError("RA-HRP requires at least two assets.")

        window = (
            returns.copy()
            .astype(float)
            .sort_index()
            .replace([np.inf, -np.inf], np.nan)
        )
        valid = (
            window.mean(axis=0).notna()
            & window.std(axis=0, skipna=True).notna()
            & (window.std(axis=0, skipna=True) > _EPS)
        )
        window = window.loc[:, valid]
        if window.shape[1] < 2:
            raise ValueError("Too few non-constant assets remain for RA-HRP.")

        # The engine has already screened interior missingness. Mean imputation
        # is used only for the residual gaps entering moment estimation.
        window = window.fillna(window.mean(axis=0))
        covariance, covariance_diagnostics = adaptive_covariance_ensemble(
            window,
            AdaptiveCovarianceConfig(
                ewma_halflife=self.config.cov_ewma_halflife,
                pca_explained_variance=self.config.cov_pca_explained_variance,
            ),
        )
        correlation = covariance_to_correlation(covariance)
        tree = build_tree_ensemble(
            correlation,
            methods=(self.config.linkage_method,),
        )[0]

        expected_returns = _estimate_expected_excess_returns(
            window,
            mean_halflife=self.config.mean_halflife,
            annual_risk_free_rate=self.config.annual_risk_free_rate,
        )

        raw_weights, ra_diagnostics = return_adjusted_hrp_allocate(
            expected_returns,
            covariance,
            tree,
            list(covariance.columns),
            score_floor=self.config.score_floor,
            interpolation=self.config.interpolation,
        )
        weights = apply_bounds(
            raw_weights,
            min_weight=self.config.min_weight,
            max_weight=self.config.max_weight,
        )
        self.last_ra_diagnostics = ra_diagnostics
        self.last_covariance_diagnostics = covariance_diagnostics
        return weights


def _estimate_expected_excess_returns(
    returns: pd.DataFrame,
    *,
    mean_halflife: float | None,
    annual_risk_free_rate: float,
) -> pd.Series:
    """Estimate weekly expected excess returns without changing their units."""
    if mean_halflife is None:
        expected_returns = returns.mean(axis=0)
    else:
        if mean_halflife <= 0.0:
            raise ValueError("mean_halflife must be positive when provided.")
        expected_returns = returns.ewm(
            halflife=mean_halflife,
            adjust=False,
        ).mean().iloc[-1]

    weekly_risk_free_rate = float(annual_risk_free_rate) / _WEEKS_PER_YEAR
    return expected_returns - weekly_risk_free_rate
