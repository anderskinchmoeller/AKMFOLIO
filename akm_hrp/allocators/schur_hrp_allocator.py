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
from akm_hrp.hrp.schur import SchurHRPDiagnostics, schur_hrp_allocate
from akm_hrp.hrp.trees import build_tree_ensemble
from akm_hrp.overlay.bounds import apply_bounds

_EPS = 1e-12


@dataclass(frozen=True)
class SchurHRPConfig:
    """gamma = 0 is classic HRP; gamma = 1 is minimum variance, long-only here."""

    gamma: float = 0.5
    linkage_method: str = "single"
    min_b: float = 1e-3
    long_only: bool = True
    split: str = "midpoint"
    cov_ewma_halflife: float = 26.0
    cov_pca_explained_variance: float = 0.90
    min_weight: float = 0.0
    max_weight: float = 0.10


class SchurHRPAllocator:
    """Project adapter for fixed-tree Schur complementary HRP (Cotton, 2024).

    Uses no expected-return input, so it is the risk-only counterpart to the
    RA-HRP allocators and shares their covariance estimator and bounds.
    """

    def __init__(self, config: SchurHRPConfig | None = None) -> None:
        self.config = config or SchurHRPConfig()
        self.last_schur_diagnostics: SchurHRPDiagnostics | None = None
        self.last_covariance_diagnostics: CovarianceBlendDiagnostics | None = None

    def reset_state(self) -> None:
        self.last_schur_diagnostics = None
        self.last_covariance_diagnostics = None

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame.")
        if returns.shape[1] < 2:
            raise ValueError("Schur HRP requires at least two assets.")

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
            raise ValueError("Too few non-constant assets remain for Schur HRP.")

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

        raw_weights, schur_diagnostics = schur_hrp_allocate(
            covariance,
            tree,
            list(covariance.columns),
            gamma=self.config.gamma,
            min_b=self.config.min_b,
            long_only=self.config.long_only,
            split=self.config.split,
        )
        weights = apply_bounds(
            raw_weights,
            min_weight=self.config.min_weight,
            max_weight=self.config.max_weight,
        )
        self.last_schur_diagnostics = schur_diagnostics
        self.last_covariance_diagnostics = covariance_diagnostics
        return weights
