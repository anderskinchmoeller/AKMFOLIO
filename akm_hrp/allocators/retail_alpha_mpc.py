from __future__ import annotations

"""Retail-capacity alpha with predictive risk and multi-period execution.

The allocator is deliberately point-in-time and stateful.  It starts with the
balanced CRSP universe, admits a small number of liquid top-2500 candidates,
learns signal efficacy only after subsequent returns arrive, and solves a
receding-horizon portfolio problem.  Only the first planned portfolio is
traded; the remaining path is a forecast, not a commitment.

This is a transparent research model, not a claim to reproduce a proprietary
vendor risk model or an institutional execution system.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import LinearConstraint, linprog, minimize
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from akm_hrp.allocators.dynamic_barra_alpha import (
    DynamicBarraAlphaAllocator,
    DynamicBarraAlphaConfig,
    _EPS,
    _current_exposures,
    _ewma_covariance,
    _latest_pit_assets,
    _neutralize,
    _project_box_simplex,
    _rank_normal,
    _robust_zscore,
)
from akm_hrp.cov.ensemble import covariance_to_correlation
from akm_hrp.hrp.allocation import hrp_allocate
from akm_hrp.overlay.bounds import apply_bounds

_SIGNALS = (
    "momentum_12_1",
    "post_earnings_drift",
    "quality_value_carry",
    "liquid_reversal",
    "retail_agility",
)
_IC_PRIORS = {
    "momentum_12_1": 0.020,
    "post_earnings_drift": 0.025,
    "quality_value_carry": 0.012,
    "liquid_reversal": 0.010,
    "retail_agility": 0.006,
}


@dataclass(frozen=True)
class RetailAlphaMPCConfig(DynamicBarraAlphaConfig):
    """Conservative defaults for a weekly, long-only retail portfolio."""

    min_weight: float = 0.0
    maximum_added_assets: int = 30
    minimum_candidate_hold_score: float | None = None
    planning_horizon: int = 3
    alpha_decay: float = 0.70
    planning_step_weeks: int = 1
    ic_halflife_rebalances: float = 12.0
    ic_prior_strength: float = 8.0
    maximum_signal_weight: float = 0.40
    minimum_signal_coverage: float = 0.30
    signal_ic_noise_floor: float = 0.003
    signal_correlation_shrinkage: float = 0.50
    har_short_weeks: int = 4
    har_medium_weeks: int = 13
    har_long_weeks: int = 52
    cross_sectional_risk_blend: float = 0.35
    cross_sectional_ridge: float = 4.0
    base_half_spread_bps: float = 4.0
    temporary_impact_coefficient: float = 0.10
    temporary_impact_exponent: float = 0.50
    permanent_impact_coefficient: float = 0.05
    hrp_anchor_strength: float = 0.08
    turnover_penalty: float = 0.0
    optimizer_max_iterations: int = 450
    allow_cvar_floor_relaxation: bool = False
    maximum_unrepresentable_weight: float = 1e-8


@dataclass(frozen=True)
class RetailAlphaMPCDiagnostics:
    core_asset_count: int
    added_asset_count: int
    selected_asset_count: int
    factor_count: int
    industry_count: int
    planning_horizon: int
    predictive_specific_volatility_median: float
    specific_risk_cross_sectional_r2: float
    covariance_condition_number: float
    predicted_weekly_volatility: float
    parametric_weekly_cvar_95: float
    effective_asset_count: float
    maximum_weight: float
    maximum_sector_weight: float
    maximum_absolute_style_exposure: float
    target_turnover_l1: float
    estimated_spread_cost: float
    estimated_temporary_impact: float
    estimated_permanent_impact: float
    estimated_total_execution_cost: float
    maximum_participation_utilization: float
    optimizer_success: bool
    optimizer_repaired: bool
    risk_limit_relaxed: bool
    effective_weekly_cvar_limit: float
    unrepresentable_exit_weight: float
    signal_weights: dict[str, float]
    signal_rank_ics: dict[str, float]
    signal_ic_observations: dict[str, float]
    signal_coverages: dict[str, float]

    exposure_limit_relaxation: float = 0.0

    def as_dict(self) -> dict[str, float | int | str | bool]:
        result: dict[str, float | int | str | bool] = {
            key: value
            for key, value in self.__dict__.items()
            if key
            not in {
                "signal_weights",
                "signal_rank_ics",
                "signal_ic_observations",
                "signal_coverages",
            }
        }
        signal_names = tuple(self.signal_weights)
        for signal in signal_names:
            result[f"signal_weight__{signal}"] = self.signal_weights[signal]
            result[f"signal_rank_ic__{signal}"] = self.signal_rank_ics[signal]
            result[f"signal_ic_observations__{signal}"] = (
                self.signal_ic_observations[signal]
            )
            result[f"signal_coverage__{signal}"] = self.signal_coverages[signal]
        return result


def _field(
    snapshot: pd.DataFrame,
    name: str,
    index: pd.Index,
    default: float = np.nan,
) -> pd.Series:
    if name not in snapshot:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(snapshot[name], errors="coerce").reindex(index)


def _mean_available(parts: list[pd.Series], index: pd.Index) -> pd.Series:
    if not parts:
        return pd.Series(0.0, index=index)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True).fillna(0.0)


def _retail_signal_panel(
    returns: pd.DataFrame,
    snapshot: pd.DataFrame,
    exposures: pd.DataFrame,
    market_caps: pd.Series,
    observed_returns: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Construct economically distinct signals using only data known now."""

    assets = returns.columns
    observed = returns if observed_returns is None else observed_returns.reindex_like(returns)
    # Medium-term momentum deliberately skips the latest four weeks, reducing
    # contamination from short-horizon reversal and microstructure effects.
    momentum_window = returns.iloc[-52:-4] if len(returns) >= 52 else returns.iloc[:-4]
    momentum = (1.0 + momentum_window).prod() - 1.0

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

    structural_parts: list[pd.Series] = []
    structural_columns: list[str] = []
    for column, direction in (
        ("book_to_market", 1.0),
        ("earnings_yield", 1.0),
        ("gross_profitability", 1.0),
        ("return_on_assets", 1.0),
        ("cash_return_on_assets", 1.0),
        ("accruals_to_assets", -1.0),
        ("leverage", -1.0),
        ("shareholder_carry", 1.0),
    ):
        if column in snapshot:
            structural_columns.append(column)
            values = _field(snapshot, column, assets)
            structural_parts.append(
                direction * _rank_normal(values).where(values.notna())
            )
    structural = _mean_available(structural_parts, assets)

    adv = _field(snapshot, "dollar_volume_20d", assets).clip(lower=1.0)
    amihud = _field(snapshot, "amihud_20d", assets).clip(lower=0.0)
    zero_fraction = _field(snapshot, "zero_return_fraction_20d", assets, 0.0)
    liquidity = (
        _robust_zscore(np.log(adv))
        - 0.5 * _robust_zscore(np.log1p(amihud.fillna(amihud.median() if amihud.notna().any() else 0.0)))
        - 0.5 * _robust_zscore(zero_fraction)
    )
    recent_reversal = -((1.0 + returns.iloc[-4:]).prod() - 1.0)
    # Reversal is permitted only where trading quality is at least reasonable.
    liquidity_gate = (1.0 / (1.0 + np.exp(-liquidity))).clip(0.15, 0.95)
    reversal = _rank_normal(recent_reversal) * liquidity_gate

    size = _robust_zscore(np.log(market_caps.clip(lower=1.0)))
    # The retail-capacity sleeve favours smaller names without rewarding the
    # untradeable tail.  It therefore requires both a size edge and liquidity.
    agility = (-size).clip(lower=0.0) * liquidity_gate

    raw = pd.DataFrame(
        {
            "momentum_12_1": _rank_normal(momentum),
            "post_earnings_drift": _rank_normal(earnings),
            "quality_value_carry": _rank_normal(structural),
            "liquid_reversal": _rank_normal(reversal),
            "retail_agility": _rank_normal(agility),
        },
        index=assets,
    ).fillna(0.0)

    def any_feature_available(columns: list[str]) -> pd.Series:
        if not columns:
            return pd.Series(False, index=assets)
        return snapshot.reindex(index=assets, columns=columns).notna().any(axis=1)

    observed_momentum = observed.iloc[-len(momentum_window) :]
    required_momentum = max(1, int(np.ceil(0.90 * len(observed_momentum))))
    signal_masks = pd.DataFrame(
        {
            "momentum_12_1": observed_momentum.notna().sum().ge(required_momentum),
            "post_earnings_drift": any_feature_available(earnings_columns),
            "quality_value_carry": any_feature_available(structural_columns),
            "liquid_reversal": (
                observed.iloc[-4:].notna().all()
                & _field(snapshot, "dollar_volume_20d", assets).notna()
            ),
            "retail_agility": (
                _field(snapshot, "market_cap_usd", assets).notna()
                & _field(snapshot, "dollar_volume_20d", assets).notna()
            ),
        },
        index=assets,
    ).fillna(False).astype(bool)

    # Remove market and industry bets from each alpha sleeve.  Style exposures
    # remain visible to the optimizer and are constrained at portfolio level.
    neutral_columns = [
        column
        for column in exposures
        if column == "MARKET" or column.startswith("IND_")
    ]
    neutral_exposures = exposures.loc[:, neutral_columns]
    cleaned: dict[str, pd.Series] = {}
    for column in raw:
        available = signal_masks[column]
        clean = pd.Series(0.0, index=assets)
        if int(available.sum()) < 3:
            cleaned[column] = clean
            continue
        available_assets = assets[available.to_numpy()]
        available_exposures = neutral_exposures.loc[available_assets]
        available_caps = market_caps.reindex(available_assets)
        ranked_residual = _neutralize(
            raw.loc[available_assets, column], available_exposures, available_caps
        )
        # _neutralize rank-transforms its residual. Ranking is robust to
        # outliers but can reintroduce the exposures just removed, so perform
        # one final weighted linear projection without another nonlinear map.
        x = available_exposures.to_numpy(dtype=float)
        y = ranked_residual.to_numpy(dtype=float)
        root_weight = np.sqrt(
            available_caps
            .fillna(available_caps.median())
            .clip(lower=_EPS)
            .to_numpy(dtype=float)
        )
        coefficient = np.linalg.lstsq(
            x * root_weight[:, None], y * root_weight, rcond=1e-8
        )[0]
        residual = pd.Series(y - x @ coefficient, index=available_assets)
        scale = max(float(residual.std(ddof=0)), _EPS)
        clean.loc[available_assets] = residual / scale
        cleaned[column] = clean

    coverage = signal_masks.mean(axis=0).reindex(_SIGNALS).fillna(0.0)
    return pd.DataFrame(cleaned, index=assets), coverage, signal_masks


def _capped_normalize(values: pd.Series, cap: float) -> pd.Series:
    """Normalize non-negative scores while imposing a diversification cap."""

    result = values.clip(lower=0.0).astype(float)
    active = result > _EPS
    if not active.any():
        return pd.Series(0.0, index=result.index)
    result.loc[~active] = 0.0
    result /= result.sum()
    effective_cap = max(float(cap), 1.0 / int(active.sum()))
    for _ in range(len(result) + 2):
        over = active & (result > effective_cap + 1e-12)
        if not over.any():
            break
        result.loc[over] = effective_cap
        remaining = 1.0 - float(result.loc[over].sum())
        under = active & ~over
        if not under.any() or remaining <= 0.0:
            break
        under_sum = float(result.loc[under].sum())
        result.loc[under] = (
            remaining / int(under.sum())
            if under_sum <= _EPS
            else result.loc[under] * remaining / under_sum
        )
    result.loc[~active] = 0.0
    return result / result.sum()


def _predictive_factor_risk_model(
    returns: pd.DataFrame,
    exposures: pd.DataFrame,
    sectors: pd.Series,
    snapshot: pd.DataFrame,
    market_caps: pd.Series,
    config: RetailAlphaMPCConfig,
) -> tuple[pd.DataFrame, pd.Series, float]:
    """Forecast factor covariance and next-period asset-specific variance.

    Specific variance blends fixed-weight heterogeneous autoregressive (HAR)
    horizons with a ridge-regularized cross-sectional forecast from size,
    liquidity, zero-return frequency, and the latest residual-volatility shock.
    Fixed horizon weights and ridge shrinkage intentionally limit overfitting.
    """

    r = returns.iloc[-config.risk_lookback_weeks :].to_numpy(dtype=float)
    b = exposures.to_numpy(dtype=float)
    cap_weight = np.sqrt(
        market_caps.reindex(returns.columns)
        .fillna(market_caps.median())
        .clip(lower=1.0)
        .to_numpy(dtype=float)
    )
    cap_weight /= max(float(np.mean(cap_weight)), _EPS)
    gram = b.T @ (cap_weight[:, None] * b)
    ridge = 1e-6 * max(float(np.trace(gram) / max(len(gram), 1)), _EPS)
    projection = np.linalg.solve(
        gram + ridge * np.eye(len(gram)), b.T * cap_weight[None, :]
    )
    factor_returns = (projection @ r.T).T
    factor_covariance = _ewma_covariance(
        factor_returns, config.factor_covariance_halflife_weeks
    )
    residuals = r - factor_returns @ b.T
    squared = residuals**2

    def horizon_mean(weeks: int) -> np.ndarray:
        return np.mean(squared[-min(weeks, len(squared)) :], axis=0)

    short = horizon_mean(config.har_short_weeks)
    medium = horizon_mean(config.har_medium_weeks)
    long = horizon_mean(config.har_long_weeks)
    har_variance = np.maximum(0.50 * short + 0.30 * medium + 0.20 * long, _EPS)

    assets = returns.columns
    adv = _field(snapshot, "dollar_volume_20d", assets).clip(lower=1.0)
    amihud = _field(snapshot, "amihud_20d", assets).clip(lower=0.0)
    zero_fraction = _field(snapshot, "zero_return_fraction_20d", assets, 0.0)
    lag_end = max(len(squared) - config.har_short_weeks, 1)
    lag_start = max(lag_end - config.har_short_weeks, 0)
    lagged_short = np.mean(squared[lag_start:lag_end], axis=0)
    lagged_long_start = max(lag_end - config.har_long_weeks, 0)
    lagged_long = np.mean(squared[lagged_long_start:lag_end], axis=0)
    predictors = pd.DataFrame(
        {
            "size": _robust_zscore(np.log(market_caps.clip(lower=1.0))),
            "adv": _robust_zscore(np.log(adv)),
            "amihud": _robust_zscore(np.log1p(amihud.fillna(amihud.median() if amihud.notna().any() else 0.0))),
            "zero": _robust_zscore(zero_fraction),
            "shock": _robust_zscore(
                pd.Series(
                    np.log((lagged_short + _EPS) / (lagged_long + _EPS)),
                    index=assets,
                )
            ),
        },
        index=assets,
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = np.column_stack([np.ones(len(assets)), predictors.to_numpy(dtype=float)])
    y = np.log(np.maximum(short, _EPS))
    penalty = np.eye(x.shape[1]) * config.cross_sectional_ridge
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    fitted = x @ coefficient
    forecast_predictors = predictors.copy()
    forecast_predictors["shock"] = _robust_zscore(
        pd.Series(np.log((short + _EPS) / (long + _EPS)), index=assets)
    )
    forecast_x = np.column_stack(
        [np.ones(len(assets)), forecast_predictors.to_numpy(dtype=float)]
    )
    cross_sectional = np.exp(forecast_x @ coefficient)
    total_sum = float(np.sum((y - y.mean()) ** 2))
    risk_r2 = (
        1.0 - float(np.sum((y - fitted) ** 2)) / total_sum
        if total_sum > _EPS
        else 0.0
    )
    raw_specific = (
        (1.0 - config.cross_sectional_risk_blend) * har_variance
        + config.cross_sectional_risk_blend * cross_sectional
    )
    sector_target = (
        pd.Series(raw_specific, index=assets)
        .groupby(sectors.reindex(assets).fillna("UNKNOWN"))
        .transform("median")
        .to_numpy(dtype=float)
    )
    specific = np.maximum(
        (1.0 - config.specific_variance_shrinkage) * raw_specific
        + config.specific_variance_shrinkage * sector_target,
        _EPS,
    )

    # Correlated residual risk is retained within industry blocks using the
    # same parsimonious DCC recursion as the dynamic Barra benchmark.
    residual_correlation = np.eye(len(assets))
    standardized = residuals / np.sqrt(specific)[None, :]
    for members in sectors.groupby(sectors, observed=True).groups.values():
        positions = assets.get_indexer(pd.Index(members))
        positions = positions[positions >= 0]
        if len(positions) < 2:
            continue
        u = standardized[:, positions]
        q_bar = np.nan_to_num(np.corrcoef(u, rowvar=False), nan=0.0)
        np.fill_diagonal(q_bar, 1.0)
        q = q_bar.copy()
        for observation in u[-52:]:
            q = (
                (1.0 - config.dcc_a - config.dcc_b) * q_bar
                + config.dcc_a * np.outer(observation, observation)
                + config.dcc_b * q
            )
        scale = np.sqrt(np.clip(np.diag(q), _EPS, None))
        block = np.clip(q / np.outer(scale, scale), -0.95, 0.95)
        np.fill_diagonal(block, 1.0)
        residual_correlation[np.ix_(positions, positions)] = block

    covariance = (
        b @ factor_covariance @ b.T
        + np.sqrt(specific)[:, None]
        * residual_correlation
        * np.sqrt(specific)[None, :]
    )
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(np.median(np.diag(covariance))) * 1e-6, _EPS)
    covariance = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    return (
        pd.DataFrame(covariance, index=assets, columns=assets),
        pd.Series(np.sqrt(specific), index=assets),
        float(np.clip(risk_r2, -1.0, 1.0)),
    )


def _spread_vector(
    snapshot: pd.DataFrame, assets: pd.Index, config: RetailAlphaMPCConfig
) -> np.ndarray:
    amihud = _field(snapshot, "amihud_20d", assets).clip(lower=0.0)
    zero = _field(snapshot, "zero_return_fraction_20d", assets, 0.0)
    illiquidity = _robust_zscore(np.log1p(amihud.fillna(amihud.median() if amihud.notna().any() else 0.0))).clip(
        lower=0.0
    )
    zero_penalty = _robust_zscore(zero).clip(lower=0.0)
    bps = config.base_half_spread_bps * (1.0 + 0.35 * illiquidity + 0.35 * zero_penalty)
    return bps.fillna(config.base_half_spread_bps).to_numpy(dtype=float) / 10_000.0


def _impact_cost_arrays(
    change: np.ndarray,
    specific_volatility: np.ndarray,
    adv: np.ndarray,
    spread: np.ndarray,
    config: RetailAlphaMPCConfig,
) -> tuple[float, float, float]:
    """Return spread, temporary, and permanent cost as capital fractions."""

    absolute = np.abs(change)
    participation = absolute * config.portfolio_value / (
        np.maximum(adv, 1.0) * config.execution_days
    )
    spread_cost = float(np.sum(spread * absolute))
    temporary = float(
        np.sum(
            config.temporary_impact_coefficient
            * specific_volatility
            * np.power(participation + _EPS, config.temporary_impact_exponent)
            * absolute
        )
    )
    # A portfolio executing through the interval pays approximately half of
    # the terminal permanent price displacement on its average filled shares.
    permanent = float(
        np.sum(
            0.5
            * config.permanent_impact_coefficient
            * specific_volatility
            * participation
            * absolute
        )
    )
    return spread_cost, temporary, permanent


class RetailAlphaMPCAllocator(DynamicBarraAlphaAllocator):
    """Balanced-core alpha model with online IC learning and MPC execution."""

    signal_names = _SIGNALS
    ic_priors = _IC_PRIORS

    def __init__(
        self,
        balanced_pit: pd.DataFrame,
        *,
        structural_features: pd.DataFrame | None = None,
        sector_history: pd.DataFrame | None = None,
        config: RetailAlphaMPCConfig | None = None,
    ) -> None:
        super().__init__(
            balanced_pit,
            structural_features=structural_features,
            sector_history=sector_history,
            config=config or RetailAlphaMPCConfig(),
        )
        self.config: RetailAlphaMPCConfig
        self.last_diagnostics: RetailAlphaMPCDiagnostics | None = None
        self.last_planned_weights: pd.DataFrame | None = None
        self.last_signal_weights: pd.Series | None = None
        self.last_signal_ics: pd.Series | None = None
        self.last_signal_coverage: pd.Series | None = None
        self.last_specific_volatility: pd.Series | None = None
        self.last_execution_cost: float = 0.0
        self._last_snapshot = pd.DataFrame()
        self._engine_current_weights: pd.Series | None = None
        self._engine_return_context: pd.DataFrame | None = None
        self._forced_exit_assets = pd.Index([])
        # Matches the write-off floor computed in prepare_allocation_window
        # (max(min_weight, 5e-4)); kept here too so allocate() has a sane
        # value even on a first call that skipped prepare_allocation_window.
        self._held_materiality_threshold = max(
            float(getattr(self.config, "min_weight", 0.0)), 5e-4
        )
        self.last_exit_assets = pd.Index([])
        self._training_dates: pd.Index | None = None
        self._previous_signal_panel: pd.DataFrame | None = None
        self._previous_signal_availability: pd.DataFrame | None = None
        self._previous_signal_date: pd.Timestamp | None = None
        self._ic_sum = pd.Series(0.0, index=self.signal_names)
        self._ic_square_sum = pd.Series(0.0, index=self.signal_names)
        self._ic_weight = pd.Series(0.0, index=self.signal_names)
        self._validate_retail_config()

    def _validate_retail_config(self) -> None:
        if self.config.planning_horizon < 1:
            raise ValueError("planning_horizon must be positive.")
        if not 0.0 < self.config.alpha_decay <= 1.0:
            raise ValueError("alpha_decay must be in (0, 1].")
        if self.config.planning_step_weeks < 1:
            raise ValueError("planning_step_weeks must be positive.")
        if self.config.ic_prior_strength <= 0.0:
            raise ValueError("ic_prior_strength must be positive.")
        minimum_cap = 1.0 / len(self.signal_names)
        if not minimum_cap <= self.config.maximum_signal_weight <= 1.0:
            raise ValueError(
                f"maximum_signal_weight must be in [{minimum_cap:.2f}, 1]."
            )
        if not 0.0 <= self.config.cross_sectional_risk_blend <= 1.0:
            raise ValueError("cross_sectional_risk_blend must be in [0, 1].")
        if not 0.0 <= self.config.minimum_signal_coverage <= 1.0:
            raise ValueError("minimum_signal_coverage must be in [0, 1].")
        if not 0.0 <= self.config.signal_ic_noise_floor <= 0.05:
            raise ValueError("signal_ic_noise_floor must be in [0, 0.05].")
        if not 0.0 <= self.config.signal_correlation_shrinkage <= 1.0:
            raise ValueError("signal_correlation_shrinkage must be in [0, 1].")
        if not 0.0 < self.config.temporary_impact_exponent <= 1.0:
            raise ValueError("temporary_impact_exponent must be in (0, 1].")

    def reset_state(self) -> None:
        super().reset_state()
        self.last_diagnostics = None
        self.last_planned_weights = None
        self.last_signal_weights = None
        self.last_signal_ics = None
        self.last_signal_coverage = None
        self.last_specific_volatility = None
        self.last_execution_cost = 0.0
        self._last_snapshot = pd.DataFrame()
        self._engine_current_weights = None
        self._engine_return_context = None
        self._forced_exit_assets = pd.Index([])
        self._held_materiality_threshold = max(
            float(getattr(self.config, "min_weight", 0.0)), 5e-4
        )
        self.last_exit_assets = pd.Index([])
        self._training_dates: pd.Index | None = None
        self._previous_signal_panel = None
        self._previous_signal_availability = None
        self._previous_signal_date = None
        self._ic_sum = pd.Series(0.0, index=self.signal_names)
        self._ic_square_sum = pd.Series(0.0, index=self.signal_names)
        self._ic_weight = pd.Series(0.0, index=self.signal_names)

    requires_split_training = True

    def set_training_dates(self, dates: pd.Index) -> None:
        """Restrict learned labels (IC, ML, Kelly) for online split validation."""
        self._training_dates = pd.DatetimeIndex(dates).copy()

    def _learning_interval_allowed(self, formation_date, label_dates) -> bool:
        return self._training_dates is None or (
            formation_date in self._training_dates
            and len(label_dates) > 0
            and pd.Index(label_dates).isin(self._training_dates).all()
        )

    def set_current_weights(self, weights: pd.Series) -> None:
        """Receive the backtest engine's drifted pre-trade holdings."""

        self._engine_current_weights = pd.Series(weights, dtype=float).copy()

    def prepare_allocation_window(
        self,
        full_window: pd.DataFrame,
        eligible_window: pd.DataFrame,
        current_weights: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Restore active balanced sleeves and positions requiring liquidation."""

        full = full_window.copy().replace([np.inf, -np.inf], np.nan)
        self._engine_return_context = full
        as_of = pd.Timestamp(full.index[-1])
        core = _latest_pit_assets(self.balanced_pit, as_of, full.columns)
        live = current_weights
        if live is None:
            live = (
                self._engine_current_weights
                if self._engine_current_weights is not None
                else self._previous_weights
            )
        # A position only stops counting as "held" once it decays below a
        # meaningful floor, not just floating-point noise (_EPS). Reusing
        # min_weight here means anything the optimizer would never size on
        # purpose gets written off instead of being force-carried and ramped
        # out indefinitely by the turnover cap. Falls back to a small fixed
        # floor when min_weight is left at its 0.0 default so this still does
        # something.
        held_materiality_threshold = max(float(getattr(self.config, "min_weight", 0.0)), 5e-4)
        # Recorded so allocate()'s omitted-weight crash-check (below) applies
        # the SAME write-off floor used here to decide what counts as "held"
        # and worth force-carrying. Without this, a dust position just under
        # this floor (e.g. weight=1.57e-4, below the 5e-4 floor but well
        # above allocate()'s maximum_unrepresentable_weight=1e-8) was
        # correctly judged immaterial here -- so it was never force-carried
        # into the window -- but then allocate() still summed it as an
        # "omitted" live weight against its much tighter 1e-8 tolerance and
        # aborted the whole walk-forward run over a position the model had
        # already, deliberately, written off.
        self._held_materiality_threshold = held_materiality_threshold
        held = (
            pd.Index([])
            if live is None
            else pd.Index(
                pd.Series(live, dtype=float)
                .loc[lambda x: x > held_materiality_threshold]
                .index
            )
        )
        original_eligible = pd.Index(eligible_window.columns)
        # Every live holding must remain representable even when it leaves PIT
        # eligibility or loses enough observations to fail the entry filter.
        # This includes balanced-core names: being in today's core does not
        # make a pre-existing position disappear from turnover or costs.
        self._forced_exit_assets = held.difference(original_eligible).intersection(
            full.columns
        )

        core_additions = core.difference(eligible_window.columns).intersection(
            full.columns
        )
        if len(core_additions):
            supplement = full.loc[:, core_additions]
            valid = (
                supplement.notna().sum()
                >= int(self.config.minimum_history_weeks)
            ) & (supplement.std(skipna=True) > _EPS)
            supplement = supplement.loc[:, valid]
            eligible_window = pd.concat([eligible_window, supplement], axis=1)

        held_additions = self._forced_exit_assets.difference(
            eligible_window.columns
        )
        if len(held_additions):
            supplement = full.loc[:, held_additions].copy()
            # An exit-only position can have insufficient or terminally sparse
            # history. Use only information available inside this formation
            # window to supply a conservative market-risk proxy for missing
            # observations. Actual observed returns, including CRSP delisting
            # returns, are never overwritten or followed by a synthetic -100%.
            proxy = eligible_window.mean(axis=1, skipna=True)
            if proxy.isna().all():
                proxy = full.mean(axis=1, skipna=True)
            proxy = proxy.fillna(0.0)
            for asset in held_additions:
                repaired = supplement[asset].where(
                    supplement[asset].notna(), proxy
                )
                if not np.isfinite(float(repaired.std(ddof=1))) or float(
                    repaired.std(ddof=1)
                ) <= _EPS:
                    repaired = proxy.copy()
                supplement[asset] = repaired
            usable = supplement.std(ddof=1).gt(_EPS)
            supplement = supplement.loc[:, usable]
            eligible_window = pd.concat([eligible_window, supplement], axis=1)

        eligible_window = eligible_window.loc[
            :, ~eligible_window.columns.duplicated()
        ]
        self._forced_exit_assets = self._forced_exit_assets.intersection(
            eligible_window.columns
        )
        return eligible_window

    def _update_dynamic_ics(
        self, returns: pd.DataFrame, current_date: pd.Timestamp
    ) -> None:
        if self._previous_signal_panel is None or self._previous_signal_date is None:
            return
        source = returns
        if (
            self._engine_return_context is not None
            and current_date in self._engine_return_context.index
        ):
            source = self._engine_return_context
        forward_rows = source.loc[
            (source.index > self._previous_signal_date)
            & (source.index <= current_date)
        ]
        if forward_rows.empty or not self._learning_interval_allowed(
            self._previous_signal_date, forward_rows.index
        ):
            return
        forward = (1.0 + forward_rows).prod(min_count=1) - 1.0
        decay = float(np.exp(np.log(0.5) / self.config.ic_halflife_rebalances))
        self._ic_sum *= decay
        self._ic_square_sum *= decay
        self._ic_weight *= decay
        for signal in self.signal_names:
            score = self._previous_signal_panel[signal]
            common = score.index.intersection(forward.dropna().index)
            if self._previous_signal_availability is not None:
                observed = (
                    self._previous_signal_availability[signal]
                    .reindex(common)
                    .fillna(False)
                    .astype(bool)
                )
                common = common[observed.to_numpy()]
            if len(common) < 8 or score.reindex(common).nunique() < 3:
                continue
            statistic = spearmanr(
                score.reindex(common), forward.reindex(common)
            ).statistic
            if not np.isfinite(statistic):
                continue
            clipped = float(np.clip(statistic, -0.20, 0.20))
            self._ic_sum[signal] += clipped
            self._ic_square_sum[signal] += clipped**2
            self._ic_weight[signal] += 1.0

    def _signal_combination(
        self,
        availability: pd.Series | None = None,
        signal_panel: pd.DataFrame | None = None,
    ) -> tuple[pd.Series, pd.Series]:
        priors = pd.Series(self.ic_priors, dtype=float).reindex(self.signal_names)
        strength = self.config.ic_prior_strength
        posterior = (strength * priors + self._ic_sum) / (
            strength + self._ic_weight
        )
        observed_mean = self._ic_sum / self._ic_weight.replace(0.0, np.nan)
        observed_second = self._ic_square_sum / self._ic_weight.replace(0.0, np.nan)
        observed_variance = (observed_second - observed_mean**2).clip(lower=0.0)
        prior_variance = 0.03**2
        uncertainty = np.sqrt(
            observed_variance.fillna(prior_variance)
            + prior_variance / (strength + self._ic_weight)
        )
        confidence = (
            posterior - self.config.signal_ic_noise_floor
        ).clip(lower=0.0) / uncertainty.clip(lower=1e-4)
        if availability is not None:
            confidence *= (
                availability.reindex(self.signal_names).fillna(0.0).clip(0.0, 1.0)
            )
        live = confidence.index[confidence > _EPS]
        if signal_panel is not None and len(live) > 1:
            correlation = (
                signal_panel.reindex(columns=live)
                .corr(method="spearman")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(dtype=float, copy=True)
            )
            np.fill_diagonal(correlation, 1.0)
            shrinkage = self.config.signal_correlation_shrinkage
            cleaned_correlation = (
                (1.0 - shrinkage) * correlation
                + shrinkage * np.eye(len(live))
            )
            adjusted = np.linalg.solve(
                cleaned_correlation + 1e-8 * np.eye(len(live)),
                confidence.reindex(live).to_numpy(dtype=float),
            )
            confidence.loc[live] = np.maximum(adjusted, 0.0)
        weights = _capped_normalize(confidence, self.config.maximum_signal_weight)
        return posterior.clip(-0.05, 0.05), weights

    def estimate_execution_cost(
        self, previous: pd.Series, target: pd.Series
    ) -> float:
        """Estimate the cost of the engine's actual, possibly capped trade."""

        if float(previous.sum()) <= _EPS:
            return 0.0
        assets = previous.index.union(target.index)
        change = target.reindex(assets).fillna(0.0) - previous.reindex(assets).fillna(
            0.0
        )
        median_vol = (
            float(self.last_specific_volatility.median())
            if self.last_specific_volatility is not None
            else 0.03
        )
        specific = (
            self.last_specific_volatility.reindex(assets).fillna(median_vol)
            if self.last_specific_volatility is not None
            else pd.Series(median_vol, index=assets)
        )
        snapshot = self._last_snapshot.reindex(assets)
        adv = _field(
            snapshot, "dollar_volume_20d", assets, self.config.minimum_dollar_volume
        ).fillna(self.config.minimum_dollar_volume)
        spread = _spread_vector(snapshot, assets, self.config)
        costs = _impact_cost_arrays(
            change.to_numpy(dtype=float),
            specific.to_numpy(dtype=float),
            adv.to_numpy(dtype=float),
            spread,
            self.config,
        )
        return float(sum(costs))

    def _build_signal_panel(
        self,
        returns: pd.DataFrame,
        snapshot: pd.DataFrame,
        exposures: pd.DataFrame,
        market_caps: pd.Series,
        observed_returns: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Signal-construction hook used by research-oriented subclasses."""

        return _retail_signal_panel(
            returns,
            snapshot,
            exposures,
            market_caps,
            observed_returns,
        )

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        window = returns.copy().astype(float).sort_index()
        window = window.replace([np.inf, -np.inf], np.nan)
        valid = (window.notna().sum() >= self.config.minimum_history_weeks) & (
            window.std(skipna=True) > _EPS
        )
        observed_window = window.loc[:, valid].copy()
        window = observed_window.fillna(observed_window.mean())
        if window.shape[1] < 2:
            raise ValueError("Retail Alpha MPC has too few eligible assets.")
        if window.shape[1] * self.config.max_weight < 1.0 - 1e-9:
            raise ValueError("max_weight is infeasible for the eligible asset count.")

        as_of = pd.Timestamp(window.index[-1])
        snapshot = self.features.snapshot(as_of, window.columns)
        sectors = self.sectors.lookup(as_of, window.columns)
        exposures, _ = _current_exposures(window, snapshot, sectors)
        market_caps = _field(
            snapshot, "market_cap_usd", window.columns, 1.0
        ).fillna(1.0)
        broad_signals, signal_coverage, signal_masks = self._build_signal_panel(
            window, snapshot, exposures, market_caps, observed_window
        )
        self._update_dynamic_ics(window, as_of)
        signal_availability = (
            broad_signals.std(ddof=0).gt(1e-8)
            & signal_coverage.ge(self.config.minimum_signal_coverage)
        ).astype(float)
        signal_ics, signal_weights = self._signal_combination(
            signal_availability, broad_signals.where(signal_masks)
        )
        admission_score = _rank_normal(broad_signals @ signal_weights)

        core = _latest_pit_assets(self.balanced_pit, as_of, window.columns)
        if len(core) < 2:
            raise ValueError("Balanced PIT core contains fewer than two assets.")
        live_previous = (
            self._engine_current_weights
            if self._engine_current_weights is not None
            else self._previous_weights
        )
        has_previous = live_previous is not None and float(live_previous.sum()) > _EPS
        previously_held = (
            pd.Index([])
            if not has_previous
            else pd.Index(
                pd.Series(live_previous, dtype=float)
                .loc[lambda values: values > _EPS]
                .index
            ).intersection(window.columns)
        )
        price = _field(snapshot, "price", window.columns)
        adv_broad = _field(snapshot, "dollar_volume_20d", window.columns)
        # Availability alone is not alpha. If cleaning/noise filtering assigns
        # every signal zero weight, do not admit arbitrary assets whose tied
        # zero score would otherwise pass a permissive candidate threshold.
        has_live_signal = bool(signal_weights.sum() > _EPS)
        asset_has_live_signal = (
            signal_masks.astype(float) @ signal_weights
        ).gt(_EPS)
        hold_score = (
            self.config.minimum_candidate_score
            if self.config.minimum_candidate_hold_score is None
            else self.config.minimum_candidate_hold_score
        )
        required_score = pd.Series(
            self.config.minimum_candidate_score,
            index=window.columns,
            dtype=float,
        )
        required_score.loc[
            previously_held.difference(core).intersection(window.columns)
        ] = hold_score
        candidate_mask = (
            ~window.columns.isin(core)
            & ~window.columns.isin(self._forced_exit_assets)
            & has_live_signal
            & asset_has_live_signal.reindex(window.columns).fillna(False).to_numpy()
            & admission_score.ge(required_score).to_numpy()
            & price.ge(self.config.minimum_price).to_numpy()
            & adv_broad.ge(self.config.minimum_dollar_volume).to_numpy()
        )
        ranked = admission_score.loc[window.columns[candidate_mask]].sort_values(
            ascending=False
        )
        additions: list[str] = []
        sector_counts: dict[str, int] = {}
        for asset in ranked.index:
            sector = str(sectors.get(asset, "UNKNOWN"))
            if sector_counts.get(sector, 0) >= self.config.max_added_per_sector:
                continue
            additions.append(str(asset))
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if len(additions) >= self.config.maximum_added_assets:
                break

        continuing = core.append(pd.Index(additions)).drop_duplicates()
        held_assets = previously_held
        exiting = held_assets.difference(continuing)
        self.last_exit_assets = exiting.copy()
        selected = continuing.append(exiting).drop_duplicates()
        selected_returns = window.loc[:, selected]
        selected_snapshot = snapshot.reindex(selected)
        selected_sectors = sectors.reindex(selected).fillna("UNKNOWN")
        selected_exposures, _ = _current_exposures(
            selected_returns, selected_snapshot, selected_sectors
        )
        selected_caps = market_caps.reindex(selected).fillna(1.0)
        selected_signals = broad_signals.reindex(selected).fillna(0.0)
        covariance, specific_volatility, risk_r2 = _predictive_factor_risk_model(
            selected_returns,
            selected_exposures,
            selected_sectors,
            selected_snapshot,
            selected_caps,
            self.config,
        )
        alpha = pd.Series(0.0, index=selected)
        for signal in self.signal_names:
            alpha += (
                signal_weights[signal]
                * signal_ics[signal]
                * specific_volatility
                * selected_signals[signal]
            )
        alpha.loc[exiting] = 0.0
        # Repaired held assets are present to model liquidation and costs, not
        # to receive a new alpha-backed allocation from synthetic history.
        alpha.loc[self._forced_exit_assets.intersection(alpha.index)] = 0.0

        correlation = covariance_to_correlation(covariance)
        distance = np.sqrt(np.maximum(0.5 * (1.0 - correlation.to_numpy()), 0.0))
        np.fill_diagonal(distance, 0.0)
        tree = linkage(squareform(distance, checks=False), method="average")
        hrp = hrp_allocate(correlation, covariance, tree, list(selected), risk_cap=0.15)
        hrp = apply_bounds(hrp, 0.0, self.config.max_weight)
        if len(exiting):
            hrp.loc[exiting] = 0.0
            if float(hrp.sum()) <= _EPS:
                raise RuntimeError("No continuing assets remain after forced exits.")
            hrp /= hrp.sum()
        previous_actual = (
            live_previous.reindex(selected).fillna(0.0)
            if has_previous
            else hrp.copy()
        )
        if has_previous:
            # Only count weight lost above the same write-off floor that
            # prepare_allocation_window used to decide what to force-carry
            # (see the comment there). A live position at or below that
            # floor was already, deliberately, judged immaterial and left
            # off the window -- it is a write-off, not an omission -- so it
            # must not also be judged against this method's much tighter
            # maximum_unrepresentable_weight tolerance.
            omitted_weight = float(
                live_previous.drop(index=selected, errors="ignore")
                .clip(lower=0.0)
                .loc[lambda values: values > self._held_materiality_threshold]
                .sum()
            )
            if omitted_weight > self.config.maximum_unrepresentable_weight:
                raise RuntimeError(
                    "Live holdings could not be represented in the MPC window; "
                    f"omitted weight={omitted_weight:.6f}."
                )
        adv = _field(
            selected_snapshot,
            "dollar_volume_20d",
            selected,
            self.config.minimum_dollar_volume,
        ).fillna(self.config.minimum_dollar_volume)
        capacity = (
            self.config.maximum_adv_participation
            * self.config.execution_days
            * adv
            / self.config.portfolio_value
        ).clip(lower=0.0)
        first_lower = (previous_actual - capacity).clip(lower=0.0)
        first_upper = np.minimum(
            self.config.max_weight, previous_actual + capacity
        ).clip(lower=0.0)
        if not has_previous:
            first_lower[:] = 0.0
            first_upper[:] = self.config.max_weight
        if float(first_upper.sum()) < 1.0 - 1e-9:
            raise ValueError(
                "Participation limits cannot fund a fully invested first step."
            )

        covariance_values = covariance.to_numpy(dtype=float)
        alpha_values = alpha.to_numpy(dtype=float)
        prior = hrp.reindex(selected).to_numpy(dtype=float)
        previous_values = previous_actual.to_numpy(dtype=float)
        specific_values = specific_volatility.to_numpy(dtype=float)
        adv_values = adv.to_numpy(dtype=float)
        spread_values = _spread_vector(selected_snapshot, selected, self.config)
        horizon = self.config.planning_horizon
        asset_count = len(selected)
        exposure_columns = [
            column
            for column in ("SIZE", "VALUE", "MOMENTUM", "QUALITY", "LOW_VOL")
            if column in selected_exposures
        ]
        style_matrix = selected_exposures.loc[:, exposure_columns].to_numpy(dtype=float)
        sector_groups = list(
            selected_sectors.groupby(selected_sectors, observed=True).groups.items()
        )
        sector_members = [
            selected.get_indexer(pd.Index(members)) for _, members in sector_groups
        ]
        sector_labels = [str(label) for label, _ in sector_groups]

        def unpack(flat: np.ndarray) -> np.ndarray:
            return flat.reshape(horizon, asset_count)

        discounts = self.config.alpha_decay ** (
            np.arange(horizon, dtype=float) * self.config.planning_step_weeks
        )
        participation_scale = self.config.portfolio_value / (
            np.maximum(adv_values, 1.0) * self.config.execution_days
        )
        permanent_slope = (
            self.config.permanent_impact_coefficient
            * specific_values
            * participation_scale
        )
        lower_matrix = np.zeros((horizon, asset_count), dtype=float)
        upper_matrix = np.full(
            (horizon, asset_count), self.config.max_weight, dtype=float
        )
        lower_matrix[0] = first_lower.to_numpy(dtype=float)
        upper_matrix[0] = first_upper.to_numpy(dtype=float)
        exit_positions = selected.get_indexer(exiting)
        exit_positions = exit_positions[exit_positions >= 0]
        for position in exit_positions:
            schedule = np.maximum(
                previous_values[position]
                - capacity.to_numpy(dtype=float)[position]
                * np.arange(1, horizon + 1, dtype=float),
                0.0,
            )
            lower_matrix[:, position] = schedule
            upper_matrix[:, position] = schedule

        initial_rows: list[np.ndarray] = []
        for step in range(horizon):
            initial_rows.append(
                _project_box_simplex(
                    pd.Series(prior, index=selected),
                    pd.Series(lower_matrix[step], index=selected),
                    pd.Series(upper_matrix[step], index=selected),
                ).to_numpy(dtype=float)
            )
        initial_path = np.vstack(initial_rows)

        def objective(flat: np.ndarray) -> float:
            path = unpack(flat)
            total = 0.0
            predecessor = previous_values
            permanent_state = np.zeros(asset_count, dtype=float)
            for step, weights in enumerate(path):
                alpha_discount = discounts[step]
                trade = weights - predecessor
                spread_cost, temporary, _ = _impact_cost_arrays(
                    trade,
                    specific_values,
                    adv_values,
                    spread_values,
                    self.config,
                )
                signed_participation = trade * self.config.portfolio_value / (
                    np.maximum(adv_values, 1.0) * self.config.execution_days
                )
                impact_increment = (
                    self.config.permanent_impact_coefficient
                    * specific_values
                    * signed_participation
                )
                permanent = float(
                    np.sum(trade * (permanent_state + 0.5 * impact_increment))
                )
                permanent_state += impact_increment
                total += (
                    0.5
                    * self.config.risk_aversion
                    * float(weights @ covariance_values @ weights)
                    - alpha_discount
                    * self.config.alpha_strength
                    * float(alpha_values @ weights)
                    + spread_cost
                    + temporary
                    + permanent
                )
                predecessor = weights
            total += self.config.hrp_anchor_strength * float(
                np.sum((path[-1] - prior) ** 2)
            )
            return float(total)

        def objective_gradient(flat: np.ndarray) -> np.ndarray:
            """Exact gradient avoids O(horizon * assets) finite differences."""

            path = unpack(flat)
            gradient = np.zeros_like(path)
            predecessors = np.vstack([previous_values, path[:-1]])
            trades = path - predecessors
            trade_gradient = np.zeros_like(trades)

            for step, weights in enumerate(path):
                gradient[step] += (
                    self.config.risk_aversion * covariance_values @ weights
                    - discounts[step]
                    * self.config.alpha_strength
                    * alpha_values
                )
                trade = trades[step]
                absolute = np.abs(trade)
                sign = np.sign(trade)
                participation = participation_scale * absolute
                temporary_gradient = (
                    self.config.temporary_impact_coefficient
                    * specific_values
                    * sign
                    * (
                        np.power(
                            participation + _EPS,
                            self.config.temporary_impact_exponent,
                        )
                        + absolute
                        * self.config.temporary_impact_exponent
                        * np.power(
                            participation + _EPS,
                            self.config.temporary_impact_exponent - 1.0,
                        )
                        * participation_scale
                    )
                )
                trade_gradient[step] += spread_values * sign + temporary_gradient

            cumulative_trade = np.cumsum(trades, axis=0) - trades
            weighted_future_trade = np.zeros(asset_count, dtype=float)
            for step in range(horizon - 1, -1, -1):
                trade_gradient[step] += (
                    permanent_slope
                    * (cumulative_trade[step] + trades[step])
                    + permanent_slope * weighted_future_trade
                )
                weighted_future_trade += trades[step]

            for step in range(horizon):
                gradient[step] += trade_gradient[step]
                if step > 0:
                    gradient[step - 1] -= trade_gradient[step]
            gradient[-1] += (
                2.0 * self.config.hrp_anchor_strength * (path[-1] - prior)
            )
            return gradient.ravel()

        sum_jacobian = np.zeros((horizon, horizon * asset_count), dtype=float)
        for step in range(horizon):
            start = step * asset_count
            sum_jacobian[step, start : start + asset_count] = 1.0

        def cvar_constraint(flat: np.ndarray) -> np.ndarray:
            path = unpack(flat)
            variance = np.einsum("ij,jk,ik->i", path, covariance_values, path)
            return self.config.weekly_cvar_95_limit - 2.0627 * np.sqrt(
                np.maximum(variance, 0.0)
            )

        def cvar_jacobian(flat: np.ndarray) -> np.ndarray:
            path = unpack(flat)
            result = np.zeros((horizon, horizon * asset_count), dtype=float)
            for step, weights in enumerate(path):
                volatility = np.sqrt(
                    max(float(weights @ covariance_values @ weights), 0.0)
                )
                if volatility <= _EPS:
                    continue
                start = step * asset_count
                result[step, start : start + asset_count] = (
                    -2.0627 * covariance_values @ weights / volatility
                )
            return result

        constraints: list[dict[str, object]] = [
            {
                "type": "eq",
                "fun": lambda flat: unpack(flat).sum(axis=1) - 1.0,
                "jac": lambda _flat: sum_jacobian,
            },
            {
                "type": "ineq",
                "fun": cvar_constraint,
                "jac": cvar_jacobian,
            },
        ]
        capacity_steps = list(range(0 if has_previous else 1, horizon))
        if capacity_steps:
            capacity_values = capacity.to_numpy(dtype=float)
            capacity_jacobian = np.zeros(
                (2 * len(capacity_steps) * asset_count, horizon * asset_count),
                dtype=float,
            )
            row_start = 0
            for step in capacity_steps:
                trade_jacobian = np.zeros(
                    (asset_count, horizon * asset_count), dtype=float
                )
                column = step * asset_count
                trade_jacobian[:, column : column + asset_count] = np.eye(asset_count)
                if step > 0:
                    previous_column = (step - 1) * asset_count
                    trade_jacobian[
                        :, previous_column : previous_column + asset_count
                    ] = -np.eye(asset_count)
                capacity_jacobian[
                    row_start : row_start + asset_count
                ] = -trade_jacobian
                capacity_jacobian[
                    row_start + asset_count : row_start + 2 * asset_count
                ] = trade_jacobian
                row_start += 2 * asset_count

            def capacity_constraint(flat: np.ndarray) -> np.ndarray:
                path = unpack(flat)
                values: list[np.ndarray] = []
                for step in capacity_steps:
                    predecessor = previous_values if step == 0 else path[step - 1]
                    trade = path[step] - predecessor
                    values.extend([capacity_values - trade, capacity_values + trade])
                return np.concatenate(values)

            constraints.append(
                {
                    "type": "ineq",
                    "fun": capacity_constraint,
                    "jac": lambda _flat: capacity_jacobian,
                }
            )
        for column in range(style_matrix.shape[1]):
            vector = style_matrix[:, column].copy()
            style_jacobian = np.zeros(
                (horizon, horizon * asset_count), dtype=float
            )
            for step in range(horizon):
                start = step * asset_count
                style_jacobian[step, start : start + asset_count] = vector
            constraints.extend(
                [
                    {
                        "type": "ineq",
                        "fun": lambda flat, v=vector: (
                            self.config.max_absolute_style_exposure
                            - unpack(flat) @ v
                        ),
                        "jac": lambda _flat, j=style_jacobian: -j,
                    },
                    {
                        "type": "ineq",
                        "fun": lambda flat, v=vector: (
                            self.config.max_absolute_style_exposure
                            + unpack(flat) @ v
                        ),
                        "jac": lambda _flat, j=style_jacobian: j,
                    },
                ]
            )
        for positions in sector_members:
            sector_jacobian = np.zeros(
                (horizon, horizon * asset_count), dtype=float
            )
            for step in range(horizon):
                start = step * asset_count
                sector_jacobian[step, start + positions] = -1.0
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda flat, p=positions: (
                        self.config.max_sector_weight
                        - unpack(flat)[:, p].sum(axis=1)
                    ),
                    "jac": lambda _flat, j=sector_jacobian: j,
                }
            )

        flat_bounds = list(zip(lower_matrix.ravel(), upper_matrix.ravel(), strict=True))
        result = minimize(
            objective,
            initial_path.ravel(),
            method="SLSQP",
            jac=objective_gradient,
            bounds=flat_bounds,
            constraints=constraints,
            options={
                "maxiter": self.config.optimizer_max_iterations,
                "ftol": 1e-9,
                "disp": False,
            },
        )

        lower_flat = lower_matrix.ravel()
        upper_flat = upper_matrix.ravel()
        risk_limit_relaxed = False
        effective_cvar_limit = float(self.config.weekly_cvar_95_limit)

        def feasibility_violation(flat: np.ndarray) -> float:
            if flat.shape != lower_flat.shape or not np.isfinite(flat).all():
                return float("inf")
            violation = max(
                float(np.max(lower_flat - flat)),
                float(np.max(flat - upper_flat)),
                0.0,
            )
            for constraint in constraints:
                values = np.asarray(constraint["fun"](flat), dtype=float)
                if not np.isfinite(values).all():
                    return float("inf")
                if constraint["type"] == "eq":
                    violation = max(violation, float(np.max(np.abs(values))))
                else:
                    violation = max(violation, float(np.max(-values)), 0.0)
            return violation

        candidate = (
            np.asarray(result.x, dtype=float)
            if np.asarray(result.x).shape == lower_flat.shape
            else initial_path.ravel()
        )
        main_violation = feasibility_violation(candidate)
        optimizer_repaired = main_violation > 1e-7
        if optimizer_repaired:
            # First solve the affine feasibility problem exactly. SLSQP is
            # unreliable when started far outside simultaneous style, sector,
            # participation, and exit constraints; HiGHS gives it a valid
            # linear starting point. CVaR remains a convex nonlinear constraint
            # and is handled by the second-stage repair below.
            zero = np.zeros_like(lower_flat)
            a_ub: list[np.ndarray] = []
            b_ub: list[np.ndarray] = []
            a_eq: list[np.ndarray] = []
            b_eq: list[np.ndarray] = []
            for position, constraint in enumerate(constraints):
                if position == 1:  # nonlinear CVaR constraint
                    continue
                values_at_zero = np.atleast_1d(
                    np.asarray(constraint["fun"](zero), dtype=float)
                )
                jacobian = np.atleast_2d(
                    np.asarray(constraint["jac"](zero), dtype=float)
                )
                if constraint["type"] == "eq":
                    a_eq.append(jacobian)
                    b_eq.append(-values_at_zero)
                else:
                    a_ub.append(-jacobian)
                    b_ub.append(values_at_zero)
            linear_feasible = linprog(
                np.zeros_like(lower_flat),
                A_ub=np.vstack(a_ub) if a_ub else None,
                b_ub=np.concatenate(b_ub) if b_ub else None,
                A_eq=np.vstack(a_eq) if a_eq else None,
                b_eq=np.concatenate(b_eq) if b_eq else None,
                bounds=flat_bounds,
                method="highs",
            )
            if not linear_feasible.success:
                asset_cap_bound = asset_count * self.config.max_weight
                sector_cap_bound = (
                    len(sector_members) * self.config.max_sector_weight
                    if sector_members
                    else float("inf")
                )
                diagnostics: list[str] = []
                if asset_cap_bound < 1.0 - 1e-9:
                    diagnostics.append(
                        f"max_weight={self.config.max_weight:.4f} across "
                        f"{asset_count} selected assets caps total investable "
                        f"weight at {asset_cap_bound:.4f} (< 1.0)"
                    )
                if sector_cap_bound < 1.0 - 1e-9:
                    diagnostics.append(
                        f"max_sector_weight={self.config.max_sector_weight:.4f} "
                        f"across {len(sector_members)} represented sectors "
                        f"({', '.join(sector_labels)}) caps total investable "
                        f"weight at {sector_cap_bound:.4f} (< 1.0)"
                    )
                if not diagnostics:
                    diagnostics.append(
                        "no single cap (max_weight x count, max_sector_weight x "
                        "sector count) alone explains this; likely a combination "
                        "of sector, style-exposure, and/or participation caps "
                        "jointly excludes any fully-invested portfolio"
                    )
                raise RuntimeError(
                    "Retail Alpha MPC constraints are linearly infeasible: "
                    + "; ".join(diagnostics)
                    + f". Raw solver message={linear_feasible.message}"
                )
            repair_start = np.asarray(linear_feasible.x, dtype=float)

            # Among linearly feasible portfolios, the global minimum-variance
            # path is the strongest possible CVaR repair. Passing the affine
            # constraints as two dense matrices avoids SLSQP repeatedly
            # evaluating dozens of Python constraint callables.
            def repair_objective(flat: np.ndarray) -> float:
                path = unpack(flat)
                return 0.5 * float(
                    np.einsum("ij,jk,ik->", path, covariance_values, path)
                )

            def repair_gradient(flat: np.ndarray) -> np.ndarray:
                return (unpack(flat) @ covariance_values).ravel()

            matrix_constraints: list[LinearConstraint] = []
            if a_eq:
                equality_matrix = np.vstack(a_eq)
                equality_target = np.concatenate(b_eq)
                matrix_constraints.append(
                    LinearConstraint(
                        equality_matrix, equality_target, equality_target
                    )
                )
            if a_ub:
                inequality_matrix = np.vstack(a_ub)
                inequality_upper = np.concatenate(b_ub)
                matrix_constraints.append(
                    LinearConstraint(
                        inequality_matrix,
                        np.full_like(inequality_upper, -np.inf),
                        inequality_upper,
                    )
                )

            repaired = minimize(
                repair_objective,
                repair_start,
                method="SLSQP",
                jac=repair_gradient,
                bounds=flat_bounds,
                constraints=matrix_constraints,
                options={
                    "maxiter": max(self.config.optimizer_max_iterations, 100),
                    "ftol": 1e-10,
                    "disp": False,
                },
            )
            repaired_values = np.asarray(repaired.x, dtype=float)
            repaired_violation = feasibility_violation(repaired_values)
            if repaired_violation > 1e-7:
                linear_violation = max(
                    float(np.max(lower_flat - repaired_values)),
                    float(np.max(repaired_values - upper_flat)),
                    0.0,
                )
                for position, constraint in enumerate(constraints):
                    if position == 1:
                        continue
                    values = np.asarray(
                        constraint["fun"](repaired_values), dtype=float
                    )
                    if constraint["type"] == "eq":
                        linear_violation = max(
                            linear_violation, float(np.max(np.abs(values)))
                        )
                    else:
                        linear_violation = max(
                            linear_violation, float(np.max(-values)), 0.0
                        )
                if (
                    self.config.allow_cvar_floor_relaxation
                    and linear_violation <= 1e-7
                ):
                    repaired_path = unpack(repaired_values)
                    repaired_variance = np.einsum(
                        "ij,jk,ik->i",
                        repaired_path,
                        covariance_values,
                        repaired_path,
                    )
                    effective_cvar_limit = max(
                        effective_cvar_limit,
                        2.0627
                        * float(np.sqrt(np.maximum(repaired_variance, 0.0)).max()),
                    )
                    risk_limit_relaxed = True
                else:
                    raise RuntimeError(
                        "Retail Alpha MPC could not produce a constraint-feasible "
                        "portfolio after repair; "
                        f"maximum violation={repaired_violation:.3e}; "
                        f"solver message={repaired.message}"
                    )
            candidate = repaired_values

        optimizer_success = True
        planned = unpack(candidate)
        final = _project_box_simplex(
            pd.Series(planned[0], index=selected),
            pd.Series(lower_matrix[0], index=selected),
            pd.Series(upper_matrix[0], index=selected),
        )
        planned[0] = final.to_numpy(dtype=float)

        change = final - previous_actual
        spread_cost, temporary, permanent = _impact_cost_arrays(
            change.to_numpy(dtype=float),
            specific_values,
            adv_values,
            spread_values,
            self.config,
        )
        values = final.to_numpy(dtype=float)
        volatility = float(np.sqrt(max(values @ covariance_values @ values, 0.0)))
        sector_weight = final.groupby(selected_sectors).sum()
        style_exposure = selected_exposures.loc[:, exposure_columns].T @ final
        utilization = (
            change.abs()
            * self.config.portfolio_value
            / (
                self.config.maximum_adv_participation
                * self.config.execution_days
                * adv.replace(0.0, np.nan)
            )
        )

        self._previous_weights = final.copy()
        self._previous_signal_panel = broad_signals.copy()
        self._previous_signal_availability = signal_masks.copy()
        self._previous_signal_date = as_of
        self.last_covariance = covariance
        self.last_alpha = alpha
        self.last_selected_assets = selected
        self.last_specific_volatility = specific_volatility
        self.last_signal_weights = signal_weights
        self.last_signal_ics = signal_ics
        self.last_signal_coverage = signal_coverage
        self.last_planned_weights = pd.DataFrame(
            planned,
            index=pd.Index(range(1, horizon + 1), name="planning_step"),
            columns=selected,
        )
        self._last_snapshot = selected_snapshot.copy()
        self.last_execution_cost = spread_cost + temporary + permanent
        self.last_diagnostics = RetailAlphaMPCDiagnostics(
            core_asset_count=len(core),
            added_asset_count=len(additions),
            selected_asset_count=len(selected),
            factor_count=selected_exposures.shape[1],
            industry_count=int(selected_sectors.nunique()),
            planning_horizon=horizon,
            predictive_specific_volatility_median=float(specific_volatility.median()),
            specific_risk_cross_sectional_r2=risk_r2,
            covariance_condition_number=float(np.linalg.cond(covariance_values)),
            predicted_weekly_volatility=volatility,
            parametric_weekly_cvar_95=2.0627 * volatility,
            effective_asset_count=float(1.0 / np.sum(values**2)),
            maximum_weight=float(final.max()),
            maximum_sector_weight=float(sector_weight.max()),
            maximum_absolute_style_exposure=(
                float(style_exposure.abs().max()) if len(style_exposure) else 0.0
            ),
            target_turnover_l1=float(change.abs().sum()) if has_previous else 0.0,
            estimated_spread_cost=spread_cost if has_previous else 0.0,
            estimated_temporary_impact=temporary if has_previous else 0.0,
            estimated_permanent_impact=permanent if has_previous else 0.0,
            estimated_total_execution_cost=(
                spread_cost + temporary + permanent if has_previous else 0.0
            ),
            maximum_participation_utilization=(
                float(utilization.max(skipna=True))
                if has_previous and utilization.notna().any()
                else 0.0
            ),
            optimizer_success=optimizer_success,
            optimizer_repaired=optimizer_repaired,
            risk_limit_relaxed=risk_limit_relaxed,
            effective_weekly_cvar_limit=effective_cvar_limit,
            unrepresentable_exit_weight=omitted_weight if has_previous else 0.0,
            signal_weights=signal_weights.to_dict(),
            signal_rank_ics=signal_ics.to_dict(),
            signal_ic_observations=self._ic_weight.to_dict(),
            signal_coverages=signal_coverage.to_dict(),
        )
        self._engine_current_weights = None
        return final
