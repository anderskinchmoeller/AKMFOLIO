from __future__ import annotations

"""Causal nonlinear signal ensemble for the retail-capacity MPC model.

The machine-learning layer is deliberately smaller than a generic neural
network.  Two regularized online regressions learn a fixed set of economically
motivated signal interactions at different decay rates.  A stock receives an
ML score only when the fast and slow models agree on its sign.  This makes the
model suitable for a low-signal, non-stationary cross-section while preserving
the parent allocator's factor risk model, constraints, and execution costs.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from akm_hrp.allocators.dynamic_barra_alpha import _EPS, _rank_normal
from akm_hrp.allocators.retail_edge_mpc import (
    RetailEdgeMPCAllocator,
    RetailEdgeMPCConfig,
    capacity_edge_signal_panel,
)

_BASE_SIGNALS = (
    "residual_momentum",
    "neglected_earnings_drift",
    "patient_quality_value",
    "conservative_investment",
    "cost_hurdled_liquidity_supply",
)
_ML_SIGNAL = "ml_interaction_ensemble"
_ML_FEATURES = (
    *_BASE_SIGNALS,
    "momentum_x_earnings",
    "earnings_x_quality",
    "quality_x_investment",
    "momentum_x_reversal",
    "quality_conviction",
)


@dataclass(frozen=True)
class RetailEdgeMLMPCConfig(RetailEdgeMPCConfig):
    """Predeclared regularization and learning-speed controls.

    Cross-sections, rather than individual securities, receive equal weight in
    the sufficient statistics.  This prevents large universes from swamping
    the history and makes results less sensitive to membership changes.
    """

    ml_ridge_penalty: float = 8.0
    ml_fast_halflife_rebalances: float = 6.0
    ml_slow_halflife_rebalances: float = 24.0
    ml_minimum_training_cross_sections: int = 12
    ml_target_clip: float = 2.5
    ml_prediction_clip: float = 4.0
    maximum_signal_weight: float = 0.30


def _ml_design(signals: pd.DataFrame) -> pd.DataFrame:
    """Create a fixed, low-dimensional nonlinear design from cleaned signals."""

    base = signals.reindex(columns=_BASE_SIGNALS).fillna(0.0).clip(-4.0, 4.0)
    design = base.copy()
    design["momentum_x_earnings"] = (
        base["residual_momentum"] * base["neglected_earnings_drift"]
    )
    design["earnings_x_quality"] = (
        base["neglected_earnings_drift"] * base["patient_quality_value"]
    )
    design["quality_x_investment"] = (
        base["patient_quality_value"] * base["conservative_investment"]
    )
    design["momentum_x_reversal"] = (
        base["residual_momentum"] * base["cost_hurdled_liquidity_supply"]
    )
    # A signed square lets the model distinguish weak from unusually strong
    # quality/value observations without adding an unconstrained feature search.
    quality = base["patient_quality_value"]
    design["quality_conviction"] = quality * quality.abs()
    return design.reindex(columns=_ML_FEATURES).clip(-6.0, 6.0)


class RetailEdgeMLMPCAllocator(RetailEdgeMPCAllocator):
    """Retail Edge MPC with a causal, consensus-gated nonlinear ML signal."""

    signal_names = (*_BASE_SIGNALS, _ML_SIGNAL)
    ic_priors = {
        **RetailEdgeMPCAllocator.ic_priors,
        # The ML signal must earn its weight out of sample; unlike the economic
        # signals, it receives no positive information-coefficient prior.
        _ML_SIGNAL: 0.0,
    }

    def __init__(self, balanced_pit: pd.DataFrame, **kwargs) -> None:
        if kwargs.get("config") is None:
            kwargs["config"] = RetailEdgeMLMPCConfig()
        super().__init__(balanced_pit, **kwargs)
        if not isinstance(self.config, RetailEdgeMLMPCConfig):
            raise TypeError(
                "RetailEdgeMLMPCAllocator requires RetailEdgeMLMPCConfig."
            )
        self._initialize_ml_state()

    def _initialize_ml_state(self) -> None:
        size = len(_ML_FEATURES)
        self._ml_fast_gram = np.zeros((size, size), dtype=float)
        self._ml_fast_cross = np.zeros(size, dtype=float)
        self._ml_slow_gram = np.zeros((size, size), dtype=float)
        self._ml_slow_cross = np.zeros(size, dtype=float)
        self._ml_training_cross_sections = 0
        self._previous_ml_design: pd.DataFrame | None = None
        self._previous_ml_eligible: pd.Series | None = None
        self._previous_ml_date: pd.Timestamp | None = None
        self._last_ml_training_date: pd.Timestamp | None = None
        self.last_ml_coefficients = pd.DataFrame()
        self.last_ml_consensus_fraction = 0.0

    def reset_state(self) -> None:
        super().reset_state()
        self._initialize_ml_state()

    def _learn_previous_cross_section(
        self, returns: pd.DataFrame, current_date: pd.Timestamp
    ) -> None:
        """Update sufficient statistics after the old target is observable."""

        if (
            self._previous_ml_design is None
            or self._previous_ml_eligible is None
            or self._previous_ml_date is None
            or self._last_ml_training_date == current_date
        ):
            return
        source = returns
        if (
            self._engine_return_context is not None
            and current_date in self._engine_return_context.index
        ):
            source = self._engine_return_context
        forward_rows = source.loc[
            (source.index > self._previous_ml_date)
            & (source.index <= current_date)
        ]
        if forward_rows.empty:
            return
        forward = (1.0 + forward_rows).prod(min_count=1) - 1.0
        common = self._previous_ml_design.index.intersection(forward.dropna().index)
        eligible = self._previous_ml_eligible.reindex(common).fillna(False)
        common = common[eligible.to_numpy(dtype=bool)]
        if len(common) < 12:
            self._last_ml_training_date = current_date
            return

        x = self._previous_ml_design.loc[common].to_numpy(dtype=float)
        target = _rank_normal(forward.reindex(common)).clip(
            -self.config.ml_target_clip, self.config.ml_target_clip
        )
        y = target.fillna(0.0).to_numpy(dtype=float)
        # Each formation date contributes one normalized observation, avoiding
        # accidental overweighting when the eligible universe expands.
        gram = x.T @ x / len(common)
        cross = x.T @ y / len(common)
        fast_decay = float(
            np.exp(np.log(0.5) / self.config.ml_fast_halflife_rebalances)
        )
        slow_decay = float(
            np.exp(np.log(0.5) / self.config.ml_slow_halflife_rebalances)
        )
        self._ml_fast_gram = fast_decay * self._ml_fast_gram + gram
        self._ml_fast_cross = fast_decay * self._ml_fast_cross + cross
        self._ml_slow_gram = slow_decay * self._ml_slow_gram + gram
        self._ml_slow_cross = slow_decay * self._ml_slow_cross + cross
        self._ml_training_cross_sections += 1
        self._last_ml_training_date = current_date

    def _ridge_coefficients(self, gram: np.ndarray, cross: np.ndarray) -> np.ndarray:
        penalty = self.config.ml_ridge_penalty * np.eye(len(_ML_FEATURES))
        return np.linalg.solve(gram + penalty, cross)

    def _ml_signal_values(
        self, design: pd.DataFrame, eligible: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        score = pd.Series(0.0, index=design.index)
        mask = pd.Series(False, index=design.index)
        if (
            self._ml_training_cross_sections
            < self.config.ml_minimum_training_cross_sections
        ):
            return score, mask

        fast = self._ridge_coefficients(self._ml_fast_gram, self._ml_fast_cross)
        slow = self._ridge_coefficients(self._ml_slow_gram, self._ml_slow_cross)
        self.last_ml_coefficients = pd.DataFrame(
            {"fast": fast, "slow": slow}, index=_ML_FEATURES
        )
        values = design.to_numpy(dtype=float)
        fast_prediction = values @ fast
        slow_prediction = values @ slow
        agreement = (
            np.sign(fast_prediction) == np.sign(slow_prediction)
        ) & eligible.to_numpy(dtype=bool)
        self.last_ml_consensus_fraction = float(np.mean(agreement))
        if int(agreement.sum()) < 8:
            return score, mask
        combined = np.clip(
            0.5 * (fast_prediction + slow_prediction),
            -self.config.ml_prediction_clip,
            self.config.ml_prediction_clip,
        )
        names = design.index[agreement]
        score.loc[names] = _rank_normal(pd.Series(combined[agreement], index=names))
        mask.loc[names] = True
        return score, mask

    def _build_signal_panel(
        self,
        returns: pd.DataFrame,
        snapshot: pd.DataFrame,
        exposures: pd.DataFrame,
        market_caps: pd.Series,
        observed_returns: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        current_date = pd.Timestamp(returns.index[-1])
        self._learn_previous_cross_section(returns, current_date)
        signals, coverage, masks = capacity_edge_signal_panel(
            returns,
            snapshot,
            exposures,
            market_caps,
            observed_returns,
            self.config,
        )
        design = _ml_design(signals)
        eligible = masks.any(axis=1)
        ml_score, ml_mask = self._ml_signal_values(design, eligible)
        signals[_ML_SIGNAL] = ml_score
        masks[_ML_SIGNAL] = ml_mask
        coverage.loc[_ML_SIGNAL] = float(ml_mask.mean())

        # Save formation-time inputs only after producing today's prediction.
        # They cannot enter training until a later allocate call observes the
        # corresponding forward return interval.
        self._previous_ml_design = design.copy()
        self._previous_ml_eligible = eligible.copy()
        self._previous_ml_date = current_date
        return signals.reindex(columns=self.signal_names), coverage, masks
