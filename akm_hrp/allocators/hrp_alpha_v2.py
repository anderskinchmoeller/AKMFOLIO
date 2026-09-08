from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from akm_hrp.allocators.hrp_alpha_v1 import (
    HRPAlphaV1Config,
    _apply_no_trade_band,
    _clean_window,
    _compounded_return,
    _project_with_bounds,
    _rank_zscore,
    blended_ledoit_wolf_covariance,
    ensemble_hrp_weights,
)

_EPS = 1e-12
_WEEKS_PER_YEAR = 52.0
_ML_FEATURE_NAMES = (
    "momentum_26_4",
    "momentum_52_4",
    "reversal_4",
    "low_volatility_26",
    "drawdown_26",
)


@dataclass(frozen=True)
class HRPAlphaV2Config(HRPAlphaV1Config):
    """Systematic, structural, and capacity-aware additions to HRP Alpha v1."""

    momentum_component_weight: float = 0.45
    microstructure_component_weight: float = 0.20
    machine_learning_component_weight: float = 0.20
    structural_component_weight: float = 0.15

    ml_forward_weeks: int = 4
    ml_training_weeks: int = 208
    ml_min_training_dates: int = 16
    ml_validation_fraction: float = 0.25
    ml_ridge_alpha: float = 10.0
    ml_ic_full_confidence: float = 0.05

    structural_max_age_days: int = 45
    microcap_market_cap_usd: float = 300_000_000.0
    microcap_max_weight: float = 0.02
    event_max_weight: float = 0.02
    minimum_dollar_volume: float = 100_000.0
    maximum_bid_ask_spread: float = 0.05
    portfolio_value: float = 100_000.0
    maximum_adv_participation: float = 0.01


@dataclass(frozen=True)
class HRPAlphaV2Diagnostics:
    eligible_asset_count: int
    tree_count: int
    covariance_condition_number: float
    long_ledoit_wolf_shrinkage: float
    short_ledoit_wolf_shrinkage: float
    momentum_active_share: float
    microstructure_score_dispersion: float
    ml_training_dates: int
    ml_validation_ic: float
    ml_confidence: float
    ml_coefficients: tuple[float, ...]
    structural_feature_assets: int
    structural_feature_coverage: float
    structural_score_dispersion: float
    microcap_asset_count: int
    liquidity_exclusion_count: int
    capacity_capped_asset_count: int
    event_signal_asset_count: int
    frozen_by_no_trade_band: int
    effective_no_trade_band: float
    target_turnover_l1: float
    effective_asset_count: float
    maximum_weight: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "eligible_asset_count": self.eligible_asset_count,
            "tree_count": self.tree_count,
            "covariance_condition_number": self.covariance_condition_number,
            "long_ledoit_wolf_shrinkage": self.long_ledoit_wolf_shrinkage,
            "short_ledoit_wolf_shrinkage": self.short_ledoit_wolf_shrinkage,
            "momentum_active_share": self.momentum_active_share,
            "microstructure_score_dispersion": (
                self.microstructure_score_dispersion
            ),
            "ml_training_dates": self.ml_training_dates,
            "ml_validation_ic": self.ml_validation_ic,
            "ml_confidence": self.ml_confidence,
            "ml_coefficients": "|".join(
                f"{name}:{value:.8g}"
                for name, value in zip(
                    _ML_FEATURE_NAMES,
                    self.ml_coefficients,
                    strict=True,
                )
            ),
            "structural_feature_assets": self.structural_feature_assets,
            "structural_feature_coverage": self.structural_feature_coverage,
            "structural_score_dispersion": self.structural_score_dispersion,
            "microcap_asset_count": self.microcap_asset_count,
            "liquidity_exclusion_count": self.liquidity_exclusion_count,
            "capacity_capped_asset_count": self.capacity_capped_asset_count,
            "event_signal_asset_count": self.event_signal_asset_count,
            "frozen_by_no_trade_band": self.frozen_by_no_trade_band,
            "effective_no_trade_band": self.effective_no_trade_band,
            "target_turnover_l1": self.target_turnover_l1,
            "effective_asset_count": self.effective_asset_count,
            "maximum_weight": self.maximum_weight,
        }


def normalize_structural_features(features: pd.DataFrame) -> pd.DataFrame:
    """Normalize a PIT long feature table to formation_date/asset columns."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("structural features must be a pandas DataFrame.")
    frame = features.copy()
    date_candidates = ("formation_date", "available_date", "date")
    asset_candidates = ("asset", "permno", "ticker")
    date_column = next((name for name in date_candidates if name in frame), None)
    asset_column = next((name for name in asset_candidates if name in frame), None)
    if date_column is None or asset_column is None:
        raise ValueError(
            "Structural features require a formation_date/date and asset/permno column."
        )
    frame = frame.rename(columns={date_column: "formation_date", asset_column: "asset"})
    frame["formation_date"] = pd.to_datetime(frame["formation_date"], errors="coerce")
    frame["asset"] = frame["asset"].astype(str).str.strip()
    frame = frame.loc[frame["formation_date"].notna() & frame["asset"].ne("")]
    ignored = {
        "formation_date",
        "asset",
        "gvkey",
        "datadate",
        "available_date",
        "age_days",
        "rdq_fallback",
    }
    for column in frame.columns.difference(list(ignored)):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["formation_date", "asset"]).reset_index(drop=True)


def combine_structural_feature_tables(
    tables: list[pd.DataFrame],
) -> pd.DataFrame | None:
    """Combine market, fundamental, and event tables without losing sparse fields."""
    if not tables:
        return None
    normalized = [normalize_structural_features(table) for table in tables]
    combined = pd.concat(normalized, ignore_index=True, sort=False)
    value_columns = [
        column
        for column in combined.columns
        if column not in {"formation_date", "asset"}
    ]
    return (
        combined.groupby(["formation_date", "asset"], as_index=False)[value_columns]
        .last()
        .sort_values(["formation_date", "asset"])
        .reset_index(drop=True)
    )


def structural_snapshot(
    features: pd.DataFrame | None,
    as_of: pd.Timestamp,
    assets: pd.Index,
    max_age_days: int,
) -> pd.DataFrame:
    """Select each asset's latest feature row known by the formation date."""
    if features is None or features.empty:
        return pd.DataFrame(index=assets)
    history = features.loc[features["formation_date"] <= as_of].copy()
    if history.empty:
        return pd.DataFrame(index=assets)
    latest = history.sort_values("formation_date").groupby("asset").tail(1)
    age = (as_of - latest["formation_date"]).dt.days
    latest = latest.loc[age <= max_age_days]
    latest = latest.set_index("asset").reindex(assets)
    return latest.drop(columns=["formation_date"], errors="ignore")


def _cross_sectional_signal_features(
    returns: pd.DataFrame,
    position: int,
) -> pd.DataFrame:
    """Return-only predictors available at one historical formation position."""
    history = returns.iloc[: position + 1]
    mom_26 = _compounded_return(history, 26, 4)
    mom_52 = _compounded_return(history, 52, 4)
    reversal = -_compounded_return(history, 4, 0)
    low_volatility = -history.iloc[-26:].std(axis=0, ddof=1)
    wealth = (1.0 + history.iloc[-26:]).cumprod()
    drawdown = wealth.iloc[-1] / wealth.cummax().max(axis=0) - 1.0
    return pd.DataFrame(
        {
            "momentum_26_4": _rank_zscore(mom_26),
            "momentum_52_4": _rank_zscore(mom_52),
            "reversal_4": _rank_zscore(reversal),
            "low_volatility_26": _rank_zscore(low_volatility),
            "drawdown_26": _rank_zscore(drawdown),
        },
        index=returns.columns,
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def causal_ridge_score(
    returns: pd.DataFrame,
    config: HRPAlphaV2Config,
) -> tuple[pd.Series, int, float, float, tuple[float, ...]]:
    """Fit a purged chronological ridge model to forward cross-sectional ranks."""
    horizon = config.ml_forward_weeks
    start = max(52 + config.momentum_skip_recent_weeks - 1, 25)
    first = max(start, len(returns) - config.ml_training_weeks)
    positions = list(range(first, len(returns) - horizon, horizon))
    zero_coefficients = tuple(0.0 for _ in _ML_FEATURE_NAMES)
    zero_score = pd.Series(0.0, index=returns.columns)
    if len(positions) < config.ml_min_training_dates:
        return zero_score, len(positions), 0.0, 0.0, zero_coefficients

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    date_ids: list[np.ndarray] = []
    for date_id, position in enumerate(positions):
        predictors = _cross_sectional_signal_features(returns, position)
        forward = (
            (1.0 + returns.iloc[position + 1 : position + 1 + horizon]).prod(axis=0)
            - 1.0
        )
        target = _rank_zscore(forward).reindex(predictors.index).fillna(0.0)
        x_parts.append(predictors.loc[:, _ML_FEATURE_NAMES].to_numpy(dtype=float))
        y_parts.append(target.to_numpy(dtype=float))
        date_ids.append(np.full(len(target), date_id, dtype=int))
    x = np.vstack(x_parts)
    y = np.concatenate(y_parts)
    ids = np.concatenate(date_ids)

    validation_dates = max(1, int(np.ceil(len(positions) * config.ml_validation_fraction)))
    split_date = len(positions) - validation_dates
    train_mask = ids < split_date
    validation_mask = ~train_mask
    if not train_mask.any() or not validation_mask.any():
        return zero_score, len(positions), 0.0, 0.0, zero_coefficients

    validation_model = Ridge(alpha=config.ml_ridge_alpha, fit_intercept=True)
    validation_model.fit(x[train_mask], y[train_mask])
    prediction = validation_model.predict(x[validation_mask])
    validation_ics = []
    for date_id in range(split_date, len(positions)):
        mask = ids[validation_mask] == date_id
        if int(mask.sum()) >= 3:
            correlation = pd.Series(prediction[mask]).corr(
                pd.Series(y[validation_mask][mask]),
                method="spearman",
            )
            if np.isfinite(correlation):
                validation_ics.append(float(correlation))
    validation_ic = float(np.mean(validation_ics)) if validation_ics else 0.0
    confidence = float(
        np.clip(validation_ic / max(config.ml_ic_full_confidence, _EPS), 0.0, 1.0)
    )

    final_model = Ridge(alpha=config.ml_ridge_alpha, fit_intercept=True)
    final_model.fit(x, y)
    current = _cross_sectional_signal_features(returns, len(returns) - 1)
    current_prediction = final_model.predict(
        current.loc[:, _ML_FEATURE_NAMES].to_numpy(dtype=float)
    )
    score = _rank_zscore(pd.Series(current_prediction, index=returns.columns))
    coefficients = tuple(float(value) for value in final_model.coef_)
    return score, len(positions), validation_ic, confidence, coefficients


def _weighted_available_scores(
    components: list[tuple[pd.Series, float]],
    assets: pd.Index,
) -> pd.Series:
    numerator = pd.Series(0.0, index=assets)
    denominator = pd.Series(0.0, index=assets)
    for score, weight in components:
        aligned = score.reindex(assets)
        available = aligned.notna()
        numerator.loc[available] += float(weight) * aligned.loc[available]
        denominator.loc[available] += abs(float(weight))
    result = numerator / denominator.replace(0.0, np.nan)
    return result


def structural_anomaly_score(snapshot: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Combine only structural fields actually present at the formation date."""
    assets = snapshot.index
    components: list[tuple[pd.Series, float]] = []

    def add(column: str, direction: float, weight: float, log: bool = False) -> None:
        if column not in snapshot:
            return
        values = pd.to_numeric(snapshot[column], errors="coerce")
        if log:
            values = np.log(values.where(values > 0.0))
        score = _rank_zscore(values).where(values.notna())
        components.append((direction * score, weight))

    add("market_cap_usd", -1.0, 0.20, log=True)
    add("analyst_count", -1.0, 0.08, log=True)
    add("earnings_yield", 1.0, 0.10)
    add("book_to_market", 1.0, 0.08)
    add("book_to_assets", 1.0, 0.04)
    add("gross_profitability", 1.0, 0.10)
    add("return_on_assets", 1.0, 0.08)
    add("cash_return_on_assets", 1.0, 0.04)
    add("leverage", -1.0, 0.05)
    add("asset_growth", -1.0, 0.04)
    add("dollar_volume_20d", 1.0, 0.05, log=True)
    add("amihud_20d", -1.0, 0.04, log=True)

    event_values: dict[str, pd.Series] = {}
    for column in (
        "merger_spread",
        "cef_discount",
        "warrant_discount",
        "option_mispricing_score",
    ):
        if column in snapshot:
            event_values[column] = pd.to_numeric(snapshot[column], errors="coerce")

    current_price = None
    for price_column in ("current_price", "price"):
        if price_column in snapshot:
            current_price = pd.to_numeric(snapshot[price_column], errors="coerce")
            break
    if current_price is not None:
        safe_price = current_price.where(current_price > 0.0)
        if "offer_price" in snapshot:
            offer = pd.to_numeric(snapshot["offer_price"], errors="coerce")
            event_values.setdefault("merger_spread", offer / safe_price - 1.0)
            required = {"deal_success_probability", "deal_break_price"}
            if required.issubset(snapshot.columns):
                probability = pd.to_numeric(
                    snapshot["deal_success_probability"], errors="coerce"
                ).clip(0.0, 1.0)
                break_price = pd.to_numeric(
                    snapshot["deal_break_price"], errors="coerce"
                )
                event_values["merger_expected_return"] = (
                    probability * (offer / safe_price - 1.0)
                    + (1.0 - probability) * (break_price / safe_price - 1.0)
                )
        if "cef_nav" in snapshot:
            nav = pd.to_numeric(snapshot["cef_nav"], errors="coerce")
            event_values.setdefault("cef_discount", (nav - safe_price) / nav)
        if "warrant_fair_value" in snapshot:
            fair_value = pd.to_numeric(
                snapshot["warrant_fair_value"], errors="coerce"
            )
            event_values.setdefault(
                "warrant_discount",
                (fair_value - safe_price) / fair_value,
            )
        if "option_fair_value" in snapshot:
            fair_value = pd.to_numeric(
                snapshot["option_fair_value"], errors="coerce"
            )
            event_values.setdefault(
                "option_mispricing_score",
                (fair_value - safe_price) / fair_value,
            )

    event_signal = pd.Series(np.nan, index=assets, dtype=float)
    event_components = []
    for values in event_values.values():
        score = _rank_zscore(values).where(values.notna())
        event_components.append(score)
        components.append((score, 0.10))
    if event_components:
        event_signal = pd.concat(event_components, axis=1).mean(axis=1, skipna=True)
    return _weighted_available_scores(components, assets), event_signal


def structural_weight_caps(
    snapshot: pd.DataFrame,
    event_signal: pd.Series,
    config: HRPAlphaV2Config,
) -> tuple[pd.Series, int, int, int, int]:
    """Translate liquidity, capacity, micro-cap, and event data into hard caps."""
    upper = pd.Series(config.max_weight, index=snapshot.index, dtype=float)
    microcap = pd.Series(False, index=snapshot.index)
    excluded = pd.Series(False, index=snapshot.index)
    capacity_capped = pd.Series(False, index=snapshot.index)

    if "market_cap_usd" in snapshot:
        market_cap = pd.to_numeric(snapshot["market_cap_usd"], errors="coerce")
        microcap = market_cap.notna() & (market_cap < config.microcap_market_cap_usd)
        upper.loc[microcap] = np.minimum(
            upper.loc[microcap],
            config.microcap_max_weight,
        )
    if "dollar_volume_20d" in snapshot:
        dollar_volume = pd.to_numeric(snapshot["dollar_volume_20d"], errors="coerce")
        illiquid = dollar_volume.notna() & (dollar_volume < config.minimum_dollar_volume)
        upper.loc[illiquid] = 0.0
        excluded |= illiquid
        capacity = (
            config.maximum_adv_participation
            * dollar_volume
            / max(config.portfolio_value, _EPS)
        )
        valid_capacity = capacity.notna() & (capacity >= 0.0)
        capacity_capped = valid_capacity & (capacity < upper)
        upper.loc[valid_capacity] = np.minimum(
            upper.loc[valid_capacity],
            capacity.loc[valid_capacity],
        )
    if "bid_ask_spread" in snapshot:
        spread = pd.to_numeric(snapshot["bid_ask_spread"], errors="coerce")
        too_wide = spread.notna() & (spread > config.maximum_bid_ask_spread)
        upper.loc[too_wide] = 0.0
        excluded |= too_wide

    # Presence of a live event field, not its cross-sectional z-score, marks an
    # event-risk position.  A single live deal has a zero rank dispersion but
    # still needs the dedicated concentration cap.
    event_assets = event_signal.notna()
    upper.loc[event_assets] = np.minimum(
        upper.loc[event_assets],
        config.event_max_weight,
    )
    return (
        upper,
        int(microcap.sum()),
        int(excluded.sum()),
        int(capacity_capped.sum()),
        int(event_assets.sum()),
    )


class HRPAlphaV2Allocator:
    """Causal systematic/ML and capacity-aware structural alpha around HRP."""

    def __init__(
        self,
        config: HRPAlphaV2Config | None = None,
        structural_features: pd.DataFrame | None = None,
    ) -> None:
        self.config = config or HRPAlphaV2Config()
        self.structural_features = (
            normalize_structural_features(structural_features)
            if structural_features is not None
            else None
        )
        self.last_diagnostics: HRPAlphaV2Diagnostics | None = None
        self.last_composite_score: pd.Series | None = None
        self._previous_weights: pd.Series | None = None
        self._validate_config()

    def _validate_config(self) -> None:
        weights = (
            self.config.momentum_component_weight,
            self.config.microstructure_component_weight,
            self.config.machine_learning_component_weight,
            self.config.structural_component_weight,
        )
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("Alpha component weights must be non-negative and non-zero.")
        if self.config.ml_forward_weeks < 1:
            raise ValueError("ml_forward_weeks must be positive.")
        if not 0.0 < self.config.ml_validation_fraction < 1.0:
            raise ValueError("ml_validation_fraction must be in (0, 1).")
        if self.config.portfolio_value <= 0.0:
            raise ValueError("portfolio_value must be positive.")
        if not 0.0 <= self.config.maximum_adv_participation <= 1.0:
            raise ValueError("maximum_adv_participation must be in [0, 1].")

    def reset_state(self) -> None:
        self.last_diagnostics = None
        self.last_composite_score = None
        self._previous_weights = None

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        window = _clean_window(returns)
        covariance, long_shrinkage, short_shrinkage, _, _ = (
            blended_ledoit_wolf_covariance(window, self.config)
        )
        base_weights, tree_count = ensemble_hrp_weights(covariance, self.config)
        assets = window.columns

        current_features = _cross_sectional_signal_features(window, len(window) - 1)
        momentum = 0.5 * current_features["momentum_26_4"] + 0.5 * current_features[
            "momentum_52_4"
        ]
        microstructure = 0.60 * current_features["reversal_4"] + 0.40 * current_features[
            "low_volatility_26"
        ]
        ml_score, training_dates, validation_ic, ml_confidence, coefficients = (
            causal_ridge_score(window, self.config)
        )
        snapshot = structural_snapshot(
            self.structural_features,
            pd.Timestamp(window.index[-1]),
            assets,
            self.config.structural_max_age_days,
        )
        structural_score, event_signal = structural_anomaly_score(snapshot)

        components = [
            (momentum, self.config.momentum_component_weight),
            (microstructure, self.config.microstructure_component_weight),
        ]
        if ml_confidence > 0.0:
            components.append(
                (
                    ml_score,
                    self.config.machine_learning_component_weight * ml_confidence,
                )
            )
        if structural_score.notna().any():
            components.append(
                (structural_score, self.config.structural_component_weight)
            )
        composite = _weighted_available_scores(components, assets).fillna(0.0)
        composite = _rank_zscore(composite)

        tilted = base_weights * np.exp(
            self.config.momentum_tilt_strength * composite
        )
        absolute_trend = _compounded_return(window, self.config.momentum_slow_weeks, 0)
        tilted.loc[absolute_trend < 0.0] *= self.config.negative_trend_multiplier
        tilted = tilted.clip(lower=0.0)
        tilted /= tilted.sum()
        momentum_active_share = 0.5 * float((tilted - base_weights).abs().sum())

        upper, microcaps, exclusions, capacity_caps, event_assets = (
            structural_weight_caps(snapshot, event_signal, self.config)
        )
        lower = pd.Series(self.config.min_weight, index=assets, dtype=float)
        if float(upper.sum()) < 1.0 - 1e-10:
            raise ValueError(
                "Liquidity/capacity constraints are infeasible: aggregate upper "
                f"weight is {float(upper.sum()):.4f}."
            )
        candidate = _project_with_bounds(tilted, lower, upper)
        effective_band = min(
            self.config.no_trade_band,
            self.config.no_trade_band_equal_weight_fraction / len(candidate),
        )
        final, frozen, turnover = _apply_no_trade_band(
            candidate,
            self._previous_weights,
            lower,
            upper,
            effective_band,
        )

        structural_assets = int(snapshot.notna().any(axis=1).sum()) if not snapshot.empty else 0
        self._previous_weights = final.copy()
        self.last_composite_score = composite.copy()
        self.last_diagnostics = HRPAlphaV2Diagnostics(
            eligible_asset_count=len(assets),
            tree_count=tree_count,
            covariance_condition_number=float(np.linalg.cond(covariance.to_numpy())),
            long_ledoit_wolf_shrinkage=long_shrinkage,
            short_ledoit_wolf_shrinkage=short_shrinkage,
            momentum_active_share=momentum_active_share,
            microstructure_score_dispersion=float(microstructure.std(ddof=0)),
            ml_training_dates=training_dates,
            ml_validation_ic=validation_ic,
            ml_confidence=ml_confidence,
            ml_coefficients=coefficients,
            structural_feature_assets=structural_assets,
            structural_feature_coverage=float(structural_assets / len(assets)),
            structural_score_dispersion=float(structural_score.std(ddof=0))
            if structural_score.notna().any()
            else 0.0,
            microcap_asset_count=microcaps,
            liquidity_exclusion_count=exclusions,
            capacity_capped_asset_count=capacity_caps,
            event_signal_asset_count=event_assets,
            frozen_by_no_trade_band=frozen,
            effective_no_trade_band=effective_band,
            target_turnover_l1=turnover,
            effective_asset_count=float(1.0 / np.sum(final.to_numpy() ** 2)),
            maximum_weight=float(final.max()),
        )
        return final
