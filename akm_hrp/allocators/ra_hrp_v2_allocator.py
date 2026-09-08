from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from akm_hrp.cov.ensemble import (
    AdaptiveCovarianceConfig,
    CovarianceBlendDiagnostics,
    covariance_scenarios,
    covariance_to_correlation,
)
from akm_hrp.hrp.robust_return_adjusted import (
    RobustReturnAdjustedDiagnostics,
    robust_return_adjusted_hrp_allocate,
)
from akm_hrp.hrp.stability import (
    ClusterStabilityDiagnostics,
    cluster_stability,
    ensemble_coclustering,
    gating_parameters,
    transition_spike_score,
)
from akm_hrp.hrp.trees import build_consensus_tree, build_tree_ensemble
from akm_hrp.overlay.bounds import apply_bounds

_EPS = 1e-12
_WEEKS_PER_YEAR = 52.0


@dataclass(frozen=True)
class RAHRPV2Config:
    """Robust extensions around the paper-defined RA-HRP return target."""

    tree_methods: tuple[str, ...] = ("single", "average", "complete")
    consensus_linkage_method: str = "average"
    downside_covariance_quantile: float = 0.25
    cov_ewma_halflife: float = 26.0
    cov_pca_explained_variance: float = 0.90

    mean_halflife: float | None = 26.0
    annual_risk_free_rate: float = 0.0
    return_shrinkage: float = 0.50
    return_winsor_quantile: float = 0.05
    score_floor: float = 1e-4
    interpolation: float = 0.75

    return_target_weight: float = 1.0
    risk_regret_weight: float = 1.0
    split_turnover_penalty: float = 0.02
    minimum_split: float = 0.10
    maximum_split: float = 0.90
    split_grid_size: int = 101
    risk_cap: float = 0.20

    stability_threshold: float = 0.70
    cluster_min_common_assets: int = 4
    transition_ema_alpha: float = 0.25
    min_bisection_aggressiveness: float = 0.35
    base_risk_budget_weight: float = 0.00
    max_risk_budget_weight: float = 0.45

    min_weight: float = 0.0
    max_weight: float = 0.10


@dataclass(frozen=True)
class RAHRPV2Diagnostics:
    tree_count: int
    eligible_asset_count: int
    covariance: CovarianceBlendDiagnostics
    clustering: ClusterStabilityDiagnostics
    allocation: RobustReturnAdjustedDiagnostics

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tree_count": self.tree_count,
            "eligible_asset_count": self.eligible_asset_count,
            **{f"cov_{key}": value for key, value in self.covariance.as_dict().items()},
            **{
                f"cluster_{key}": value
                for key, value in self.clustering.as_dict().items()
            },
            **{
                f"allocation_{key}": value
                for key, value in self.allocation.as_dict().items()
            },
        }


def robust_expected_excess_returns(
    returns: pd.DataFrame,
    *,
    mean_halflife: float | None = 26.0,
    annual_risk_free_rate: float = 0.0,
    shrinkage: float = 0.50,
    winsor_quantile: float = 0.05,
) -> pd.Series:
    """Estimate causal weekly excess returns with shrinkage and winsorization."""
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be between zero and one.")
    if not 0.0 <= winsor_quantile < 0.5:
        raise ValueError("winsor_quantile must be in [0, 0.5).")
    if mean_halflife is not None and mean_halflife <= 0.0:
        raise ValueError("mean_halflife must be positive when provided.")

    if mean_halflife is None:
        estimate = returns.mean(axis=0)
    else:
        estimate = returns.ewm(halflife=mean_halflife, adjust=False).mean().iloc[-1]
    estimate = estimate - float(annual_risk_free_rate) / _WEEKS_PER_YEAR

    centre = float(estimate.median())
    estimate = (1.0 - shrinkage) * estimate + shrinkage * centre
    if winsor_quantile > 0.0 and len(estimate) >= 4:
        lower = float(estimate.quantile(winsor_quantile))
        upper = float(estimate.quantile(1.0 - winsor_quantile))
        estimate = estimate.clip(lower=lower, upper=upper)
    return estimate.astype(float)


class RAHRPV2Allocator:
    """Consensus, scenario-robust, stability-aware Return-Adjusted HRP."""

    def __init__(self, config: RAHRPV2Config | None = None) -> None:
        self.config = config or RAHRPV2Config()
        self.last_diagnostics: RAHRPV2Diagnostics | None = None
        self._previous_coclustering: pd.DataFrame | None = None
        self._transition_ema: float | None = None
        self._previous_weights: pd.Series | None = None

    def reset_state(self) -> None:
        self.last_diagnostics = None
        self._previous_coclustering = None
        self._transition_ema = None
        self._previous_weights = None

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame.")

        window = (
            returns.copy()
            .astype(float)
            .sort_index()
            .replace([np.inf, -np.inf], np.nan)
        )
        valid = (
            (window.notna().sum(axis=0) >= 2)
            & window.std(axis=0, skipna=True).notna()
            & (window.std(axis=0, skipna=True) > _EPS)
        )
        window = window.loc[:, valid]
        if window.shape[1] < 2:
            raise ValueError("Too few non-constant assets remain for RA-HRP v2.")

        # Eligibility was screened by the engine; mean fill handles only the
        # small residual gaps entering dense covariance operations.
        window = window.fillna(window.mean(axis=0))
        scenarios, covariance_diagnostics = covariance_scenarios(
            window,
            config=AdaptiveCovarianceConfig(
                ewma_halflife=self.config.cov_ewma_halflife,
                pca_explained_variance=self.config.cov_pca_explained_variance,
            ),
            downside_quantile=self.config.downside_covariance_quantile,
        )
        reference = scenarios["blend"]
        assets = list(reference.columns)

        trees: list[np.ndarray] = []
        for scenario in scenarios.values():
            trees.extend(
                build_tree_ensemble(
                    covariance_to_correlation(scenario),
                    methods=self.config.tree_methods,
                )
            )
        coclustering = ensemble_coclustering(trees, assets)
        stability, n_common = cluster_stability(
            self._previous_coclustering,
            coclustering,
            min_common_assets=self.config.cluster_min_common_assets,
        )
        transition_rate = 1.0 - stability
        spike, transition_ema = transition_spike_score(
            transition_rate,
            self._transition_ema,
            ema_alpha=self.config.transition_ema_alpha,
        )
        aggressiveness, risk_budget_weight = gating_parameters(
            stability,
            spike,
            stability_threshold=self.config.stability_threshold,
            min_bisection_aggressiveness=self.config.min_bisection_aggressiveness,
            base_risk_budget_weight=self.config.base_risk_budget_weight,
            max_risk_budget_weight=self.config.max_risk_budget_weight,
        )
        cluster_diagnostics = ClusterStabilityDiagnostics(
            stability=stability,
            transition_rate=transition_rate,
            transition_ema=transition_ema,
            transition_spike=spike,
            bisection_aggressiveness=aggressiveness,
            risk_budget_weight=risk_budget_weight,
            n_common_assets=n_common,
            n_clusters=max(1, round(np.sqrt(len(assets)))),
        )

        expected_returns = robust_expected_excess_returns(
            window,
            mean_halflife=self.config.mean_halflife,
            annual_risk_free_rate=self.config.annual_risk_free_rate,
            shrinkage=self.config.return_shrinkage,
            winsor_quantile=self.config.return_winsor_quantile,
        )
        consensus_tree = build_consensus_tree(
            coclustering,
            method=self.config.consensus_linkage_method,
        )
        raw_weights, allocation_diagnostics = (
            robust_return_adjusted_hrp_allocate(
                expected_returns,
                reference,
                scenarios,
                consensus_tree,
                assets,
                previous_weights=self._previous_weights,
                score_floor=self.config.score_floor,
                interpolation=self.config.interpolation,
                return_target_weight=self.config.return_target_weight,
                risk_regret_weight=self.config.risk_regret_weight,
                turnover_penalty=self.config.split_turnover_penalty,
                minimum_split=self.config.minimum_split,
                maximum_split=self.config.maximum_split,
                grid_size=self.config.split_grid_size,
                bisection_aggressiveness=aggressiveness,
                risk_budget_weight=risk_budget_weight,
                risk_cap=self.config.risk_cap,
            )
        )
        final_weights = apply_bounds(
            raw_weights,
            min_weight=self.config.min_weight,
            max_weight=self.config.max_weight,
        ).reindex(assets).fillna(0.0)

        self._previous_coclustering = coclustering
        self._transition_ema = transition_ema
        self._previous_weights = final_weights.copy()
        self.last_diagnostics = RAHRPV2Diagnostics(
            tree_count=len(trees),
            eligible_asset_count=len(assets),
            covariance=covariance_diagnostics,
            clustering=cluster_diagnostics,
            allocation=allocation_diagnostics,
        )
        return final_weights
