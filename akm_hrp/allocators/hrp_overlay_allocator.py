from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from akm_hrp.cov.ensemble import (
    AdaptiveCovarianceConfig,
    covariance_scenarios,
    covariance_to_correlation,
)
from akm_hrp.hrp.allocation import hrp_allocate
from akm_hrp.hrp.robust_allocation import (
    RobustAllocationDiagnostics,
    regret_aware_hrp_allocate,
)
from akm_hrp.hrp.stability import (
    ClusterStabilityDiagnostics,
    cluster_stability,
    ensemble_coclustering,
    gating_parameters,
    transition_spike_score,
)
from akm_hrp.hrp.trees import build_consensus_tree, build_tree_ensemble
from akm_hrp.overlay.overlay import apply_overlays
from akm_hrp.signals.alpha_stack import alpha_stack


@dataclass(frozen=True)
class AllocatorConfig:
    # HRP
    tree_methods: tuple[str, ...] = ("single", "average", "complete")
    risk_cap: float = 0.20

    # Regret-aware consensus HRP
    use_regret_aware_hrp: bool = True
    downside_covariance_quantile: float = 0.25
    robust_minimum_split: float = 0.10
    robust_maximum_split: float = 0.90
    robust_split_grid_size: int = 101
    robust_turnover_penalty: float = 0.02
    consensus_linkage_method: str = "average"

    # Cluster-stability gating
    use_cluster_stability_gating: bool = True
    cluster_stability_threshold: float = 0.70
    cluster_min_common_assets: int = 4
    cluster_transition_ema_alpha: float = 0.25
    min_bisection_aggressiveness: float = 0.35
    base_risk_budget_weight: float = 0.00
    max_risk_budget_weight: float = 0.45

    # Final portfolio bounds
    min_weight: float = 0.00
    max_weight: float = 0.10

    # Adaptive covariance ensemble
    use_adaptive_covariance: bool = True
    cov_ewma_halflife: float = 26.0
    cov_pca_explained_variance: float = 0.90

    # Alpha shrinkage
    alpha_bayes_lambda: float = 0.50

    # Signal-budget overlay
    use_signal_budget: bool = True
    budget_beta: float = 0.20
    budget_tau: float = 0.75

    # Covariance-inverse overlay
    use_cov_inverse: bool = True
    beamforming_beta: float = 0.20
    beamforming_loading: float = 0.10
    beamforming_strength: float = 0.50

    # Limit active risk from the alpha layer relative to the robust core.
    overlay_max_active_share: float = 0.10


class HRPOverlayAllocator:
    """
    Ensemble-HRP allocator with:
      - shrinkage-adaptive covariance blending
      - cluster-stability gating
      - optional signal-budget and covariance-inverse overlays

    State is deliberately local to one allocator instance, so stability is
    measured only within the same chronological walk-forward/CPCV path.
    """

    def __init__(self, config: AllocatorConfig | None = None):
        self.config = config or AllocatorConfig()

        self.last_covariance_diagnostics = None
        self.last_cluster_diagnostics: ClusterStabilityDiagnostics | None = None
        self.last_robust_diagnostics: RobustAllocationDiagnostics | None = None

        self._previous_coclustering: pd.DataFrame | None = None
        self._transition_ema: float | None = None
        self._previous_weights: pd.Series | None = None

    def reset_state(self) -> None:
        """
        Reset path-dependent cluster diagnostics.
        Useful before starting a new independent backtest path.
        """
        self.last_cluster_diagnostics = None
        self.last_robust_diagnostics = None
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

        if window.shape[1] < 2:
            raise ValueError(
                f"HRP requires at least 2 assets; received {window.shape[1]}."
            )

        valid = (
            (window.notna().sum(axis=0) >= 2)
            & window.std(skipna=True).notna()
            & (window.std(skipna=True) > 1e-12)
        )
        window = window.loc[:, valid]

        if window.shape[1] < 2:
            raise ValueError(
                "Too few non-constant assets remain for HRP allocation."
            )

        if self.config.use_adaptive_covariance:
            cov_cfg = AdaptiveCovarianceConfig(
                ewma_halflife=self.config.cov_ewma_halflife,
                pca_explained_variance=self.config.cov_pca_explained_variance,
            )

            scenario_covariances, diagnostics = covariance_scenarios(
                window,
                config=cov_cfg,
                downside_quantile=self.config.downside_covariance_quantile,
            )
            cov = scenario_covariances["blend"]
            self.last_covariance_diagnostics = diagnostics
            corr = covariance_to_correlation(cov)
        else:
            cov = window.cov()
            corr = window.corr()
            scenario_covariances = {"sample": cov}
            self.last_covariance_diagnostics = None

        if not np.isfinite(cov.to_numpy(dtype=float)).all():
            bad = cov.columns[
                ~np.isfinite(cov.to_numpy(dtype=float)).all(axis=0)
            ].tolist()
            raise ValueError(
                "Covariance matrix contains non-finite values. "
                f"Problem assets: {bad}"
            )

        if not np.isfinite(corr.to_numpy(dtype=float)).all():
            bad = corr.columns[
                ~np.isfinite(corr.to_numpy(dtype=float)).all(axis=0)
            ].tolist()
            raise ValueError(
                "Correlation matrix contains non-finite values. "
                f"Problem assets: {bad}"
            )

        assets = list(cov.columns)
        window = window.loc[:, assets]

        trees = []
        for scenario_covariance in scenario_covariances.values():
            scenario_corr = covariance_to_correlation(scenario_covariance)
            trees.extend(
                build_tree_ensemble(
                    scenario_corr,
                    methods=self.config.tree_methods,
                )
            )

        if not trees:
            raise RuntimeError("No HRP trees were generated.")

        # ---------------------------------------------------------------
        # Cluster-stability gating
        # ---------------------------------------------------------------
        current_coclustering = ensemble_coclustering(
            trees,
            assets,
        )

        if self.config.use_cluster_stability_gating:

            stability, n_common = cluster_stability(
                self._previous_coclustering,
                current_coclustering,
                min_common_assets=self.config.cluster_min_common_assets,
            )

            transition_rate = 1.0 - stability

            spike, updated_ema = transition_spike_score(
                transition_rate,
                self._transition_ema,
                ema_alpha=self.config.cluster_transition_ema_alpha,
            )

            bisection_aggressiveness, risk_budget_weight = gating_parameters(
                stability,
                spike,
                stability_threshold=self.config.cluster_stability_threshold,
                min_bisection_aggressiveness=(
                    self.config.min_bisection_aggressiveness
                ),
                base_risk_budget_weight=self.config.base_risk_budget_weight,
                max_risk_budget_weight=self.config.max_risk_budget_weight,
            )

            self.last_cluster_diagnostics = ClusterStabilityDiagnostics(
                stability=stability,
                transition_rate=transition_rate,
                transition_ema=updated_ema,
                transition_spike=spike,
                bisection_aggressiveness=bisection_aggressiveness,
                risk_budget_weight=risk_budget_weight,
                n_common_assets=n_common,
                n_clusters=max(
                    1,
                    int(round(np.sqrt(len(assets)))),
                ),
            )

            # Update state only after diagnostics for this rebalance have
            # been computed against the previous rebalance.
            self._previous_coclustering = current_coclustering
            self._transition_ema = updated_ema

        else:
            bisection_aggressiveness = 1.0
            risk_budget_weight = 0.0
            self.last_cluster_diagnostics = None

        # ---------------------------------------------------------------
        # Regret-aware consensus HRP, with the legacy ensemble retained as a
        # reproducible benchmark switch.
        # ---------------------------------------------------------------
        if self.config.use_regret_aware_hrp:
            consensus_tree = build_consensus_tree(
                current_coclustering,
                method=self.config.consensus_linkage_method,
            )
            hrp_weights, robust_diagnostics = regret_aware_hrp_allocate(
                reference_covariance=cov,
                covariance_scenarios=scenario_covariances,
                tree=consensus_tree,
                asset_names=assets,
                previous_weights=self._previous_weights,
                risk_cap=self.config.risk_cap,
                bisection_aggressiveness=bisection_aggressiveness,
                risk_budget_weight=risk_budget_weight,
                minimum_split=self.config.robust_minimum_split,
                maximum_split=self.config.robust_maximum_split,
                grid_size=self.config.robust_split_grid_size,
                turnover_penalty=self.config.robust_turnover_penalty,
            )
            self.last_robust_diagnostics = robust_diagnostics
        else:
            hrp_members = []
            for tree in trees:
                member = hrp_allocate(
                    corr=corr,
                    cov=cov,
                    tree=tree,
                    asset_names=assets,
                    risk_cap=self.config.risk_cap,
                    bisection_aggressiveness=bisection_aggressiveness,
                    risk_budget_weight=risk_budget_weight,
                )
                hrp_members.append(
                    member.reindex(assets).fillna(0.0).astype(float)
                )
            hrp_weights = pd.concat(hrp_members, axis=1).mean(axis=1)
            self.last_robust_diagnostics = None

        hrp_weights = hrp_weights.clip(lower=0.0)

        total = float(hrp_weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError("HRP ensemble produced invalid weights.")

        hrp_weights /= total

        if self.config.use_signal_budget or self.config.use_cov_inverse:
            alpha = alpha_stack(
                window,
                bayes_lambda=self.config.alpha_bayes_lambda,
            )
            alpha = (
                alpha.reindex(assets)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
        else:
            alpha = pd.Series(0.0, index=assets, dtype=float)

        final_weights = apply_overlays(
            hrp_weights=hrp_weights,
            alpha=alpha,
            cov=cov,
            use_signal_budget=self.config.use_signal_budget,
            use_cov_inverse=self.config.use_cov_inverse,
            budget_beta=self.config.budget_beta,
            budget_tau=self.config.budget_tau,
            beamforming_beta=self.config.beamforming_beta,
            beamforming_loading=self.config.beamforming_loading,
            beamforming_strength=self.config.beamforming_strength,
            min_weight=self.config.min_weight,
            max_weight=self.config.max_weight,
            max_active_share=self.config.overlay_max_active_share,
        )

        final_weights = (
            final_weights.reindex(assets)
            .fillna(0.0)
            .astype(float)
        )

        if not np.isfinite(final_weights.to_numpy()).all():
            raise RuntimeError(
                "Allocator produced non-finite final portfolio weights."
            )

        if (final_weights < -1e-12).any():
            raise RuntimeError(
                "Allocator produced negative final portfolio weights."
            )

        total = float(final_weights.sum())
        if total <= 0.0:
            raise RuntimeError(
                "Allocator produced zero total portfolio weight."
            )

        final_weights = final_weights.clip(lower=0.0)
        final_weights /= final_weights.sum()

        self._previous_weights = final_weights.copy()

        return final_weights
