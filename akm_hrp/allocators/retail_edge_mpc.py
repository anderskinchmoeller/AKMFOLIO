from __future__ import annotations

"""Capacity-conditioned alpha for portfolios too small to interest large funds.

The model deliberately does not equate "microcap" with alpha. A stock enters
the research niche only when a retail-sized order is implementable while a
minimum meaningful position for a multi-billion-dollar fund is awkward in
ownership or trading-time terms. Within that niche, the model combines a
small, predeclared set of persistent signals and lets causal rank-IC evidence
turn signals down when they stop working.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from akm_hrp.allocators.dynamic_barra_alpha import (
    _EPS,
    _neutralize,
    _rank_normal,
)
from akm_hrp.allocators.retail_alpha_mpc import (
    RetailAlphaMPCAllocator,
    RetailAlphaMPCConfig,
    _field,
    _mean_available,
    _spread_vector,
)

_EDGE_SIGNALS = (
    "residual_momentum",
    "neglected_earnings_drift",
    "patient_quality_value",
    "conservative_investment",
    "cost_hurdled_liquidity_supply",
)
_EDGE_IC_PRIORS = {
    "residual_momentum": 0.018,
    "neglected_earnings_drift": 0.020,
    "patient_quality_value": 0.014,
    "conservative_investment": 0.010,
    "cost_hurdled_liquidity_supply": 0.006,
}


@dataclass(frozen=True)
class RetailEdgeMPCConfig(RetailAlphaMPCConfig):
    """Conservative defaults for the capacity-limited opportunity set."""

    maximum_added_assets: int = 12
    planning_horizon: int = 1
    optimizer_max_iterations: int = 30
    allow_cvar_floor_relaxation: bool = True
    # Dust positions can lose usable estimation history after a terminal event.
    # The engine still targets them to zero and includes them in turnover/costs.
    maximum_unrepresentable_weight: float = 0.0001
    minimum_candidate_score: float = 0.50
    minimum_candidate_hold_score: float | None = -0.15
    maximum_signal_weight: float = 0.35
    minimum_signal_coverage: float = 0.20
    minimum_edge_market_cap: float = 20_000_000.0
    maximum_edge_market_cap: float = 1_500_000_000.0
    institutional_fund_size: float = 5_000_000_000.0
    institutional_meaningful_position_bps: float = 5.0
    institutional_max_adv_participation: float = 0.10
    institutional_minimum_execution_days: float = 2.0
    institutional_minimum_ownership: float = 0.005
    maximum_zero_return_fraction: float = 0.20
    reversal_round_trip_cost_multiple: float = 2.0
    alpha_strength: float = 0.08
    # A 6% weekly CVaR ceiling is below the attainable long-only risk floor in
    # broad-equity crisis windows. Twelve percent remains a meaningful stress
    # bound while avoiding routine emergency relaxation in 2008/2020 regimes.
    weekly_cvar_95_limit: float = 0.12


def _residual_returns(
    returns: pd.DataFrame,
    exposures: pd.DataFrame,
) -> pd.DataFrame:
    """Remove current market/industry structure from each weekly cross-section."""

    columns = [
        column
        for column in exposures
        if column == "MARKET" or column.startswith("IND_")
    ]
    if not columns:
        return returns.sub(returns.mean(axis=1), axis=0)
    design = exposures.loc[returns.columns, columns].to_numpy(dtype=float)
    # A tiny ridge handles a redundant market dummy/industry system without
    # allowing numerical rank choices to move the residual signal.
    gram = design.T @ design + 1e-8 * np.eye(design.shape[1])
    coefficients = np.linalg.solve(
        gram, design.T @ returns.to_numpy(dtype=float).T
    )
    fitted = (design @ coefficients).T
    return pd.DataFrame(
        returns.to_numpy(dtype=float) - fitted,
        index=returns.index,
        columns=returns.columns,
    )


def _clean_edge_signals(
    raw: pd.DataFrame,
    masks: pd.DataFrame,
    exposures: pd.DataFrame,
    market_caps: pd.Series,
) -> pd.DataFrame:
    """Neutralize signals without inventing scores for unavailable assets."""

    assets = raw.index
    neutral_columns = [
        column
        for column in exposures
        if column == "MARKET" or column.startswith("IND_")
    ]
    neutral_exposures = exposures.loc[:, neutral_columns]
    cleaned: dict[str, pd.Series] = {}
    for signal in raw:
        available = masks[signal].fillna(False).astype(bool)
        result = pd.Series(0.0, index=assets)
        if int(available.sum()) < 3:
            cleaned[signal] = result
            continue
        names = assets[available.to_numpy()]
        x_frame = neutral_exposures.loc[names]
        caps = market_caps.reindex(names)
        ranked = _neutralize(raw.loc[names, signal], x_frame, caps)
        x = x_frame.to_numpy(dtype=float)
        y = ranked.to_numpy(dtype=float)
        root_weight = np.sqrt(
            caps.fillna(caps.median()).clip(lower=_EPS).to_numpy(dtype=float)
        )
        coefficient = np.linalg.lstsq(
            x * root_weight[:, None], y * root_weight, rcond=1e-8
        )[0]
        residual = pd.Series(y - x @ coefficient, index=names)
        result.loc[names] = residual / max(float(residual.std(ddof=0)), _EPS)
        cleaned[signal] = result
    return pd.DataFrame(cleaned, index=assets)


def capacity_edge_signal_panel(
    returns: pd.DataFrame,
    snapshot: pd.DataFrame,
    exposures: pd.DataFrame,
    market_caps: pd.Series,
    observed_returns: pd.DataFrame,
    config: RetailEdgeMPCConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build five predeclared signals inside a two-sided capacity gate."""

    assets = returns.columns
    adv = _field(snapshot, "dollar_volume_20d", assets)
    price = _field(snapshot, "price", assets)
    zero_fraction = _field(snapshot, "zero_return_fraction_20d", assets)
    cap = market_caps.reindex(assets)
    meaningful_dollars = (
        config.institutional_fund_size
        * config.institutional_meaningful_position_bps
        / 10_000.0
    )
    large_fund_days = meaningful_dollars / (
        config.institutional_max_adv_participation * adv.clip(lower=1.0)
    )
    ownership = meaningful_dollars / cap.clip(lower=1.0)
    retail_days = (
        config.max_weight * config.portfolio_value
        / (config.maximum_adv_participation * adv.clip(lower=1.0))
    )
    capacity_gate = (
        cap.between(config.minimum_edge_market_cap, config.maximum_edge_market_cap)
        & price.ge(config.minimum_price)
        & adv.ge(config.minimum_dollar_volume)
        & zero_fraction.le(config.maximum_zero_return_fraction)
        & retail_days.le(config.execution_days)
        & (
            large_fund_days.ge(config.institutional_minimum_execution_days)
            | ownership.ge(config.institutional_minimum_ownership)
        )
    ).fillna(False)

    residual = _residual_returns(returns, exposures)
    momentum_window = residual.iloc[-52:-4] if len(residual) >= 52 else residual.iloc[:-4]
    residual_momentum = (1.0 + momentum_window).prod() - 1.0

    earnings_columns = [
        column
        for column in ("standardized_unexpected_earnings", "fundamental_momentum")
        if column in snapshot
    ]
    earnings_parts = []
    for column in earnings_columns:
        values = _field(snapshot, column, assets)
        earnings_parts.append(_rank_normal(values).where(values.notna()))
    earnings = _mean_available(earnings_parts, assets)

    quality_parts = []
    quality_columns: list[str] = []
    for column, direction in (
        ("book_to_market", 1.0),
        ("gross_profitability", 1.0),
        ("return_on_assets", 1.0),
        ("cash_return_on_assets", 1.0),
        ("accruals_to_assets", -1.0),
    ):
        if column in snapshot:
            quality_columns.append(column)
            values = _field(snapshot, column, assets)
            quality_parts.append(
                direction * _rank_normal(values).where(values.notna())
            )
    quality_value = _mean_available(quality_parts, assets)

    investment_parts = []
    investment_columns: list[str] = []
    for column in ("asset_growth", "sales_growth", "accruals_to_assets"):
        if column in snapshot:
            investment_columns.append(column)
            values = _field(snapshot, column, assets)
            investment_parts.append(-_rank_normal(values).where(values.notna()))
    conservative_investment = _mean_available(investment_parts, assets)

    recent_residual = (1.0 + residual.iloc[-4:]).prod() - 1.0
    reversal = -recent_residual
    spread = pd.Series(
        _spread_vector(snapshot, assets, config), index=assets, dtype=float
    )
    # A retail investor has little market impact, but still crosses the spread.
    # Admit liquidity provision only when the observed dislocation clears a
    # conservative round-trip spread hurdle before ranking.
    reversal_hurdle = reversal.abs().ge(
        config.reversal_round_trip_cost_multiple * spread
    )

    def any_available(columns: list[str]) -> pd.Series:
        if not columns:
            return pd.Series(False, index=assets)
        return snapshot.reindex(index=assets, columns=columns).notna().any(axis=1)

    observed_momentum = observed_returns.iloc[-len(momentum_window) :]
    required_momentum = max(1, int(np.ceil(0.90 * len(observed_momentum))))
    masks = pd.DataFrame(
        {
            "residual_momentum": (
                capacity_gate
                & observed_momentum.notna().sum().ge(required_momentum)
            ),
            "neglected_earnings_drift": (
                capacity_gate & any_available(earnings_columns)
            ),
            "patient_quality_value": (
                capacity_gate & any_available(quality_columns)
            ),
            "conservative_investment": (
                capacity_gate & any_available(investment_columns)
            ),
            "cost_hurdled_liquidity_supply": (
                capacity_gate
                & observed_returns.iloc[-4:].notna().all()
                & reversal_hurdle
            ),
        },
        index=assets,
    ).fillna(False).astype(bool)
    raw = pd.DataFrame(
        {
            "residual_momentum": _rank_normal(residual_momentum),
            "neglected_earnings_drift": _rank_normal(earnings),
            "patient_quality_value": _rank_normal(quality_value),
            "conservative_investment": _rank_normal(conservative_investment),
            "cost_hurdled_liquidity_supply": _rank_normal(reversal),
        },
        index=assets,
    )
    cleaned = _clean_edge_signals(raw, masks, exposures, cap)
    coverage = masks.mean(axis=0).reindex(_EDGE_SIGNALS).fillna(0.0)
    return cleaned, coverage, masks


class RetailEdgeMPCAllocator(RetailAlphaMPCAllocator):
    """Retail MPC specialized for small, implementable capacity gaps."""

    signal_names = _EDGE_SIGNALS
    ic_priors = _EDGE_IC_PRIORS

    def __init__(self, balanced_pit: pd.DataFrame, **kwargs) -> None:
        if kwargs.get("config") is None:
            kwargs["config"] = RetailEdgeMPCConfig()
        super().__init__(balanced_pit, **kwargs)
        if not isinstance(self.config, RetailEdgeMPCConfig):
            raise TypeError("RetailEdgeMPCAllocator requires RetailEdgeMPCConfig.")

    def _build_signal_panel(
        self,
        returns: pd.DataFrame,
        snapshot: pd.DataFrame,
        exposures: pd.DataFrame,
        market_caps: pd.Series,
        observed_returns: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        return capacity_edge_signal_panel(
            returns,
            snapshot,
            exposures,
            market_caps,
            observed_returns,
            self.config,
        )
