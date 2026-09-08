from __future__ import annotations

"""Dynamic balanced-core + top-2500 Barra-style alpha allocator.

This is a research implementation of the public factor-risk architecture, not
a licensed MSCI Barra model.  Every admission decision is made from the
balanced PIT membership, broad PIT mask, returns, and feature snapshot known at
the formation date supplied by the walk-forward engine.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from scipy.stats import norm, spearmanr

from akm_hrp.cov.ensemble import covariance_to_correlation
from akm_hrp.hrp.allocation import hrp_allocate
from akm_hrp.overlay.bounds import apply_bounds

_EPS = 1e-12


@dataclass(frozen=True)
class DynamicBarraAlphaConfig:
    """Fixed weekly equivalents of the daily production specification."""

    minimum_history_weeks: int = 104
    requires_pairwise_finite_correlation: bool = False
    risk_lookback_weeks: int = 156
    factor_covariance_halflife_weeks: float = 12.0
    specific_risk_halflife_weeks: float = 8.0
    specific_variance_shrinkage: float = 0.40
    dcc_a: float = 0.02
    dcc_b: float = 0.95
    maximum_added_assets: int = 50
    minimum_candidate_score: float = 0.0
    minimum_price: float = 5.0
    minimum_dollar_volume: float = 1_000_000.0
    max_added_per_sector: int = 8
    max_weight: float = 0.03
    max_sector_weight: float = 0.25
    max_absolute_style_exposure: float = 0.25
    portfolio_value: float = 1_000_000.0
    maximum_adv_participation: float = 0.10
    execution_days: float = 5.0
    risk_aversion: float = 4.0
    alpha_strength: float = 0.10
    hrp_anchor_strength: float = 0.25
    turnover_penalty: float = 0.02
    weekly_cvar_95_limit: float = 0.06
    optimizer_max_iterations: int = 300


class CovarianceBackend:
    """Abstract covariance backend."""

    def estimate(
        self,
        returns: pd.DataFrame,
        exposures: pd.DataFrame,
        sectors: pd.Series,
        market_caps: pd.Series,
        config: DynamicBarraAlphaConfig,
    ) -> tuple[pd.DataFrame, pd.Series]:
        raise NotImplementedError


class FactorRiskCovarianceBackend(CovarianceBackend):
    """Baseline backend: original Barra-style factor risk model."""

    def estimate(
        self,
        returns: pd.DataFrame,
        exposures: pd.DataFrame,
        sectors: pd.Series,
        market_caps: pd.Series,
        config: DynamicBarraAlphaConfig,
    ) -> tuple[pd.DataFrame, pd.Series]:
        return _factor_risk_model(returns, exposures, sectors, market_caps, config)


@dataclass(frozen=True)
class DynamicBarraAlphaDiagnostics:
    core_asset_count: int
    added_asset_count: int
    selected_asset_count: int
    factor_count: int
    industry_count: int
    bayesian_rank_ic: float
    covariance_condition_number: float
    predicted_weekly_volatility: float
    parametric_weekly_cvar_95: float
    effective_asset_count: float
    maximum_weight: float
    maximum_sector_weight: float
    maximum_absolute_style_exposure: float
    target_turnover_l1: float
    maximum_participation_utilization: float
    optimizer_success: bool

    def as_dict(self) -> dict[str, float | int | str]:
        return self.__dict__.copy()


def _asset_ids(values: pd.Index | pd.Series) -> pd.Index:
    return pd.Index(
        pd.Series(values, dtype="string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def _robust_zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = float(values.median())
    mad = 1.4826 * float((values - median).abs().median())
    if not np.isfinite(mad) or mad <= _EPS:
        mad = float(values.std(ddof=0))
    if not np.isfinite(mad) or mad <= _EPS:
        return pd.Series(0.0, index=values.index)
    return ((values.fillna(median) - median) / mad).clip(-3.0, 3.0)


def _rank_normal(values: pd.Series) -> pd.Series:
    """Winsorize, rank, and map the cross-section through the normal probit."""

    clean = pd.to_numeric(values, errors="coerce")
    if clean.notna().sum() < 3:
        return pd.Series(0.0, index=values.index)
    lower, upper = clean.quantile([0.01, 0.99])
    ranks = clean.clip(lower, upper).rank(method="average", pct=True)
    probability = ranks.clip(1.0 / (2.0 * len(ranks)), 1.0 - 1.0 / (2.0 * len(ranks)))
    result = pd.Series(norm.ppf(probability), index=values.index)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _ewma_covariance(values: np.ndarray, halflife: float) -> np.ndarray:
    age = np.arange(len(values) - 1, -1, -1, dtype=float)
    weights = np.exp2(-age / halflife)
    weights /= weights.sum()
    mean = weights @ values
    centred = values - mean
    denominator = max(1.0 - float(weights @ weights), _EPS)
    covariance = (centred * weights[:, None]).T @ centred / denominator
    return 0.5 * (covariance + covariance.T)


class _FeatureStore:
    """Incremental as-of store for already-causal feature observations."""

    def __init__(self, features: pd.DataFrame | None) -> None:
        if features is None or features.empty:
            self.frame = None
            self.date_values = np.array([], dtype="datetime64[ns]")
            self.reset()
            return
        frame = features.copy()
        date_column = next(
            (column for column in ("formation_date", "date") if column in frame),
            None,
        )
        asset_column = next(
            (column for column in ("asset", "permno") if column in frame),
            None,
        )
        if date_column is None or asset_column is None:
            raise ValueError(
                "Dynamic features require formation_date and asset/permno."
            )
        frame = frame.rename(
            columns={date_column: "formation_date", asset_column: "asset"}
        )
        frame["formation_date"] = pd.to_datetime(
            frame["formation_date"], errors="raise"
        )
        frame["asset"] = _asset_ids(frame["asset"])
        frame = frame.sort_values(["formation_date", "asset"])
        frame = frame.drop_duplicates(["formation_date", "asset"], keep="last")
        self.date_values = frame["formation_date"].to_numpy(dtype="datetime64[ns]")
        self.frame = frame.reset_index(drop=True)
        self.value_columns = [
            column
            for column in frame.columns
            if column not in {"formation_date", "asset"}
        ]
        self.reset()

    def reset(self) -> None:
        self.cursor = 0
        self.last_as_of: pd.Timestamp | None = None
        self.current = pd.DataFrame()

    def snapshot(self, as_of: pd.Timestamp, assets: pd.Index) -> pd.DataFrame:
        if self.frame is None or len(self.date_values) == 0:
            return pd.DataFrame(index=assets)
        if self.last_as_of is not None and as_of < self.last_as_of:
            self.reset()
        right = int(
            np.searchsorted(self.date_values, np.datetime64(as_of), side="right")
        )
        if right <= self.cursor:
            return self.current.reindex(assets)
        batch = self.frame.iloc[self.cursor : right]
        if batch.empty:
            return pd.DataFrame(index=assets)
        latest = batch.groupby("asset", observed=True)[self.value_columns].last()
        if self.current.empty:
            self.current = latest
        else:
            self.current = (
                pd.concat([self.current, latest]).groupby(level=0, observed=True).last()
            )
        self.cursor = right
        self.last_as_of = as_of
        return self.current.reindex(assets).apply(pd.to_numeric, errors="coerce")


class _SectorStore:
    """Compact point-in-time sector interval lookup."""

    def __init__(self, history: pd.DataFrame | None) -> None:
        self.intervals: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if history is None or history.empty:
            return
        frame = history.copy()
        required = {"permno", "sec_info_start", "sec_info_end", "sector"}
        if missing := required.difference(frame.columns):
            raise ValueError(f"Sector history is missing columns: {sorted(missing)}")
        frame["asset"] = _asset_ids(frame["permno"])
        frame["start"] = pd.to_datetime(frame["sec_info_start"], errors="raise")
        frame["end"] = pd.to_datetime(frame["sec_info_end"], errors="coerce").fillna(
            pd.Timestamp.max.normalize()
        )
        labels = frame["sector"].astype("string").str.strip()
        unusable = labels.str.upper().isin({"", "NOAVAIL", "N/A", "NA"})
        if "sic_code" in frame:
            sic = (
                frame["sic_code"].astype("string").str.extract(r"(\d{2})", expand=False)
            )
            labels = labels.mask(unusable, "SIC_" + sic)
        frame["label"] = labels.fillna("UNKNOWN")
        for asset, rows in frame.sort_values("start").groupby("asset", observed=True):
            self.intervals[str(asset)] = (
                rows["start"].to_numpy(dtype="datetime64[ns]"),
                rows["end"].to_numpy(dtype="datetime64[ns]"),
                rows["label"].astype(str).to_numpy(),
            )

    def lookup(self, as_of: pd.Timestamp, assets: pd.Index) -> pd.Series:
        when = np.datetime64(as_of.to_datetime64())
        result: dict[str, str] = {}
        for asset in assets.astype(str):
            interval = self.intervals.get(asset)
            if interval is None:
                result[asset] = "UNKNOWN"
                continue
            starts, ends, labels = interval
            position = int(np.searchsorted(starts, when, side="right") - 1)
            result[asset] = (
                str(labels[position])
                if position >= 0 and when <= ends[position]
                else "UNKNOWN"
            )
        return pd.Series(result).reindex(assets)


def _latest_pit_assets(
    pit: pd.DataFrame,
    as_of: pd.Timestamp,
    available: pd.Index,
) -> pd.Index:
    position = int(pit.index.searchsorted(as_of, side="right") - 1)
    if position < 0:
        return pd.Index([])
    row = pit.iloc[position].reindex(available).fillna(0).astype(bool)
    return available[row.to_numpy()]


def _return_signals(returns: pd.DataFrame) -> pd.DataFrame:
    momentum_26 = (1.0 + returns.iloc[-26:]).prod() - 1.0
    momentum_52 = (1.0 + returns.iloc[-52:]).prod() - 1.0
    low_volatility = -returns.iloc[-26:].std(ddof=1)
    return pd.DataFrame(
        {
            "momentum": _rank_normal(0.5 * (momentum_26 + momentum_52)),
            "low_volatility": _rank_normal(low_volatility),
        }
    )


def _current_exposures(
    returns: pd.DataFrame,
    snapshot: pd.DataFrame,
    sectors: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    signals = _return_signals(returns)
    size = _robust_zscore(
        np.log(
            snapshot.get(
                "market_cap_usd", pd.Series(index=returns.columns, dtype=float)
            ).clip(lower=1.0)
        )
    )
    value = _robust_zscore(
        snapshot.get("book_to_market", pd.Series(index=returns.columns, dtype=float))
    )
    quality_parts = []
    for column, direction in (
        ("gross_profitability", 1.0),
        ("return_on_assets", 1.0),
        ("cash_return_on_assets", 1.0),
        ("accruals_to_assets", -1.0),
        ("leverage", -1.0),
    ):
        if column in snapshot:
            quality_parts.append(direction * _robust_zscore(snapshot[column]))
    quality = (
        pd.concat(quality_parts, axis=1).mean(axis=1)
        if quality_parts
        else pd.Series(0.0, index=returns.columns)
    )
    styles = pd.DataFrame(
        {
            "SIZE": size,
            "VALUE": value,
            "MOMENTUM": signals["momentum"],
            "QUALITY": quality,
            "LOW_VOL": signals["low_volatility"],
        },
        index=returns.columns,
    ).fillna(0.0)
    styles = styles.apply(_robust_zscore, axis=0)
    industry = pd.get_dummies(
        sectors.reindex(returns.columns).fillna("UNKNOWN"), prefix="IND", dtype=float
    )
    if industry.shape[1] > 1:
        industry = industry.iloc[:, 1:]
    exposures = pd.concat(
        [pd.DataFrame({"MARKET": 1.0}, index=returns.columns), styles, industry],
        axis=1,
    )
    raw_alpha = pd.concat(
        [styles["VALUE"], styles["MOMENTUM"], styles["QUALITY"], styles["LOW_VOL"]],
        axis=1,
    ).mean(axis=1)
    return exposures, _rank_normal(raw_alpha)


def _neutralize(
    score: pd.Series, exposures: pd.DataFrame, weights: pd.Series
) -> pd.Series:
    x = exposures.to_numpy(dtype=float)
    y = score.reindex(exposures.index).fillna(0.0).to_numpy(dtype=float)
    root_weight = np.sqrt(
        weights.reindex(exposures.index)
        .fillna(weights.median())
        .clip(lower=_EPS)
        .to_numpy()
    )
    xw, yw = x * root_weight[:, None], y * root_weight
    coefficient = np.linalg.lstsq(xw, yw, rcond=1e-8)[0]
    return _rank_normal(pd.Series(y - x @ coefficient, index=exposures.index))


def _bayesian_rank_ic(returns: pd.DataFrame) -> float:
    correlations: list[float] = []
    start = max(52, len(returns) - 104)
    for position in range(start, len(returns) - 4, 4):
        history = returns.iloc[: position + 1]
        signal = 0.5 * (
            (1.0 + history.iloc[-26:]).prod()
            - 1.0
            + (1.0 + history.iloc[-52:]).prod()
            - 1.0
        )
        forward = (1.0 + returns.iloc[position + 1 : position + 5]).prod() - 1.0
        correlation = spearmanr(signal, forward, nan_policy="omit").statistic
        if np.isfinite(correlation):
            correlations.append(float(correlation))
    prior_ic, prior_strength = 0.02, 12.0
    posterior = (prior_strength * prior_ic + float(np.sum(correlations))) / (
        prior_strength + len(correlations)
    )
    return float(np.clip(posterior, 0.0, 0.05))


def _factor_risk_model(
    returns: pd.DataFrame,
    exposures: pd.DataFrame,
    sectors: pd.Series,
    market_caps: pd.Series,
    config: DynamicBarraAlphaConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    r = returns.iloc[-config.risk_lookback_weeks :].to_numpy(dtype=float)
    b = exposures.to_numpy(dtype=float)
    cap_weight = np.sqrt(
        market_caps.reindex(returns.columns)
        .fillna(market_caps.median())
        .clip(lower=1.0)
        .to_numpy()
    )
    cap_weight /= max(float(np.mean(cap_weight)), _EPS)
    gram = b.T @ (cap_weight[:, None] * b)
    ridge = 1e-6 * max(float(np.trace(gram) / len(gram)), _EPS)
    projection = np.linalg.solve(
        gram + ridge * np.eye(len(gram)),
        b.T * cap_weight[None, :],
    )
    factor_returns = (projection @ r.T).T
    factor_covariance = _ewma_covariance(
        factor_returns,
        config.factor_covariance_halflife_weeks,
    )
    if len(factor_returns) > 1:
        centred = factor_returns - factor_returns.mean(axis=0)
        lag = centred[1:].T @ centred[:-1] / max(len(centred) - 1, 1)
        factor_covariance += 0.5 * (lag + lag.T)
    residuals = r - factor_returns @ b.T
    age = np.arange(len(residuals) - 1, -1, -1, dtype=float)
    ewma_weight = np.exp2(-age / config.specific_risk_halflife_weeks)
    ewma_weight /= ewma_weight.sum()
    raw_specific = ewma_weight @ (residuals**2)
    sector_target = (
        pd.Series(raw_specific, index=returns.columns)
        .groupby(sectors)
        .transform("median")
        .to_numpy()
    )
    specific = (
        (1.0 - config.specific_variance_shrinkage) * raw_specific
        + config.specific_variance_shrinkage * sector_target
    ).clip(min=_EPS)

    residual_correlation = np.eye(len(returns.columns))
    standardized = residuals / np.sqrt(specific)[None, :]
    for members in sectors.groupby(sectors, observed=True).groups.values():
        positions = returns.columns.get_indexer(pd.Index(members))
        positions = positions[positions >= 0]
        if len(positions) < 2:
            continue
        u = standardized[:, positions]
        q_bar = np.corrcoef(u, rowvar=False)
        q_bar = np.nan_to_num(q_bar, nan=0.0)
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
        + np.sqrt(specific)[:, None] * residual_correlation * np.sqrt(specific)[None, :]
    )
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(np.median(np.diag(covariance))) * 1e-6, _EPS)
    covariance = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    return pd.DataFrame(
        covariance, index=returns.columns, columns=returns.columns
    ), pd.Series(np.sqrt(specific), index=returns.columns)


def _project_box_simplex(
    target: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> pd.Series:
    """Project a target onto fully-invested asset-specific trade bounds."""

    index = target.index
    lo = lower.reindex(index).to_numpy(dtype=float)
    hi = upper.reindex(index).to_numpy(dtype=float)
    if float(lo.sum()) > 1.0 + 1e-10 or float(hi.sum()) < 1.0 - 1e-10:
        raise ValueError(
            "Liquidity/participation bounds are infeasible: "
            f"lower sum={lo.sum():.4f}, upper sum={hi.sum():.4f}."
        )
    values = target.reindex(index).fillna(0.0).to_numpy(dtype=float)
    left = float(np.min(values - hi) - 1.0)
    right = float(np.max(values - lo) + 1.0)
    for _ in range(200):
        middle = 0.5 * (left + right)
        projected = np.clip(values - middle, lo, hi)
        if float(projected.sum()) > 1.0:
            left = middle
        else:
            right = middle
    projected = np.clip(values - 0.5 * (left + right), lo, hi)
    return pd.Series(projected / projected.sum(), index=index)


class DynamicBarraAlphaAllocator:
    """Balanced core with PIT alpha-driven admissions from a broader universe."""

    def __init__(
        self,
        balanced_pit: pd.DataFrame,
        *,
        structural_features: pd.DataFrame | None = None,
        sector_history: pd.DataFrame | None = None,
        config: DynamicBarraAlphaConfig | None = None,
        covariance_backend: CovarianceBackend | None = None,
    ) -> None:
        self.config = config or DynamicBarraAlphaConfig()
        pit = balanced_pit.copy()
        pit.index = pd.to_datetime(pit.index, errors="raise")
        pit.columns = _asset_ids(pit.columns)
        self.balanced_pit = pit.sort_index().fillna(0).astype(bool)
        self.features = _FeatureStore(structural_features)
        self.sectors = _SectorStore(sector_history)
        self.last_diagnostics: DynamicBarraAlphaDiagnostics | None = None
        self.last_covariance: pd.DataFrame | None = None
        self.last_alpha: pd.Series | None = None
        self.last_selected_assets: pd.Index | None = None
        self._previous_weights: pd.Series | None = None
        self.covariance_backend = covariance_backend or FactorRiskCovarianceBackend()
        self._validate_config()

    def _validate_config(self) -> None:
        if self.config.minimum_history_weeks < 52:
            raise ValueError("minimum_history_weeks must be at least 52.")
        if self.config.maximum_added_assets < 0:
            raise ValueError("maximum_added_assets cannot be negative.")
        if self.config.max_added_per_sector < 1:
            raise ValueError("max_added_per_sector must be positive.")
        if not 0.0 < self.config.max_weight <= 1.0:
            raise ValueError("max_weight must be in (0, 1].")
        if not 0.0 <= self.config.maximum_adv_participation <= 1.0:
            raise ValueError("maximum_adv_participation must be in [0, 1].")
        if self.config.dcc_a < 0.0 or self.config.dcc_b < 0.0:
            raise ValueError("DCC parameters cannot be negative.")
        if self.config.dcc_a + self.config.dcc_b >= 1.0:
            raise ValueError("DCC parameters must satisfy a + b < 1.")

    def reset_state(self) -> None:
        self.last_diagnostics = None
        self.last_covariance = None
        self.last_alpha = None
        self.last_selected_assets = None
        self._previous_weights = None
        self.features.reset()

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        window = (
            returns.copy().astype(float).sort_index().replace([np.inf, -np.inf], np.nan)
        )
        valid = (window.notna().sum() >= self.config.minimum_history_weeks) & (
            window.std(skipna=True) > _EPS
        )
        window = window.loc[:, valid].fillna(window.mean())
        if window.shape[1] < 2:
            raise ValueError("Dynamic Barra Alpha has too few eligible assets.")
        as_of = pd.Timestamp(window.index[-1])
        snapshot = self.features.snapshot(as_of, window.columns)
        sectors = self.sectors.lookup(as_of, window.columns)
        exposures, raw_score = _current_exposures(window, snapshot, sectors)
        market_caps = snapshot.get(
            "market_cap_usd", pd.Series(1.0, index=window.columns)
        ).fillna(1.0)
        neutral_score = _neutralize(raw_score, exposures, market_caps)
        core = _latest_pit_assets(self.balanced_pit, as_of, window.columns)
        if len(core) < 2:
            raise ValueError(
                "Balanced PIT core contains fewer than two eligible assets."
            )

        price = snapshot.get("price", pd.Series(np.nan, index=window.columns))
        dollar_volume = snapshot.get(
            "dollar_volume_20d", pd.Series(np.nan, index=window.columns)
        )
        candidate_mask = (
            ~window.columns.isin(core)
            & neutral_score.ge(self.config.minimum_candidate_score).to_numpy()
            & price.ge(self.config.minimum_price).to_numpy()
            & dollar_volume.ge(self.config.minimum_dollar_volume).to_numpy()
        )
        ranked_candidates = neutral_score.loc[
            window.columns[candidate_mask]
        ].sort_values(ascending=False)
        additions: list[str] = []
        sector_counts: dict[str, int] = {}
        for asset in ranked_candidates.index:
            sector = str(sectors.get(asset, "UNKNOWN"))
            if sector_counts.get(sector, 0) >= self.config.max_added_per_sector:
                continue
            additions.append(str(asset))
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if len(additions) >= self.config.maximum_added_assets:
                break
        selected = core.append(pd.Index(additions)).drop_duplicates()
        selected_returns = window.loc[:, selected]
        selected_snapshot = snapshot.reindex(selected)
        selected_sectors = sectors.reindex(selected).fillna("UNKNOWN")
        selected_exposures, selected_raw_score = _current_exposures(
            selected_returns,
            selected_snapshot,
            selected_sectors,
        )
        selected_caps = selected_snapshot.get(
            "market_cap_usd", pd.Series(1.0, index=selected)
        ).fillna(1.0)
        selected_score = _neutralize(
            selected_raw_score, selected_exposures, selected_caps
        )
        covariance, specific_volatility = self.covariance_backend.estimate(
            selected_returns,
            selected_exposures,
            selected_sectors,
            selected_caps,
            self.config,
        )
        rank_ic = _bayesian_rank_ic(selected_returns)
        alpha = rank_ic * specific_volatility * selected_score

        correlation = covariance_to_correlation(covariance)
        distance = np.sqrt(np.maximum(0.5 * (1.0 - correlation.to_numpy()), 0.0))
        np.fill_diagonal(distance, 0.0)
        tree = linkage(squareform(distance, checks=False), method="average")
        hrp = hrp_allocate(correlation, covariance, tree, list(selected), risk_cap=0.15)
        hrp = apply_bounds(hrp, 0.0, self.config.max_weight)
        has_previous = self._previous_weights is not None
        previous_actual = (
            self._previous_weights.reindex(selected).fillna(0.0)
            if has_previous
            else pd.Series(0.0, index=selected)
        )
        optimizer_reference = previous_actual if has_previous else hrp.copy()

        adv = selected_snapshot.get(
            "dollar_volume_20d", pd.Series(np.nan, index=selected)
        )
        trade_capacity = (
            self.config.maximum_adv_participation
            * self.config.execution_days
            * adv
            / self.config.portfolio_value
        ).fillna(self.config.max_weight)
        lower = (previous_actual - trade_capacity).clip(lower=0.0)
        upper = np.minimum(
            self.config.max_weight,
            previous_actual + trade_capacity,
        ).clip(lower=0.0)
        if not has_previous:
            lower[:] = 0.0
        if float(upper.sum()) < 1.0 - 1e-9:
            raise ValueError(
                "Participation limits cannot fund a fully invested target; "
                "increase execution_days, participation, or reduce portfolio value."
            )

        covariance_values = covariance.to_numpy(dtype=float)
        alpha_scale = max(float(alpha.abs().median()), _EPS)
        alpha_score = alpha.to_numpy(dtype=float) / alpha_scale
        prior = hrp.reindex(selected).to_numpy(dtype=float)
        previous_values = optimizer_reference.to_numpy(dtype=float)
        exposure_columns = [
            column
            for column in ("SIZE", "VALUE", "MOMENTUM", "QUALITY", "LOW_VOL")
            if column in selected_exposures
        ]
        style_matrix = selected_exposures.loc[:, exposure_columns].to_numpy(dtype=float)
        sector_members = [
            selected.get_indexer(pd.Index(members))
            for _, members in selected_sectors.groupby(
                selected_sectors, observed=True
            ).groups.items()
        ]

        def objective(weights: np.ndarray) -> float:
            change = weights - previous_values
            return float(
                0.5
                * self.config.risk_aversion
                * (weights @ covariance_values @ weights)
                - self.config.alpha_strength * (alpha_score @ weights)
                + self.config.hrp_anchor_strength * np.sum((weights - prior) ** 2)
                + self.config.turnover_penalty * np.sum(np.sqrt(change**2 + 1e-8))
            )

        constraints: list[dict[str, object]] = [
            {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
            {
                "type": "ineq",
                "fun": lambda weights: float(
                    self.config.weekly_cvar_95_limit
                    - 2.0627 * np.sqrt(max(weights @ covariance_values @ weights, 0.0))
                ),
            },
        ]
        for column in range(style_matrix.shape[1]):
            vector = style_matrix[:, column].copy()
            constraints.extend(
                [
                    {
                        "type": "ineq",
                        "fun": lambda weights, v=vector: (
                            self.config.max_absolute_style_exposure - float(v @ weights)
                        ),
                    },
                    {
                        "type": "ineq",
                        "fun": lambda weights, v=vector: (
                            self.config.max_absolute_style_exposure + float(v @ weights)
                        ),
                    },
                ]
            )
        for positions in sector_members:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weights, p=positions: (
                        self.config.max_sector_weight - float(weights[p].sum())
                    ),
                }
            )
        result = minimize(
            objective,
            prior,
            method="SLSQP",
            bounds=list(zip(lower.to_numpy(), upper.to_numpy(), strict=True)),
            constraints=constraints,
            options={
                "maxiter": self.config.optimizer_max_iterations,
                "ftol": 1e-9,
                "disp": False,
            },
        )
        optimizer_success = bool(result.success and np.isfinite(result.x).all())
        candidate = (
            pd.Series(result.x, index=selected)
            if optimizer_success
            else optimizer_reference
        )
        final = _project_box_simplex(
            candidate.clip(lower=0.0),
            lower,
            upper,
        )
        self._previous_weights = final.copy()
        self.last_covariance = covariance
        self.last_alpha = alpha
        self.last_selected_assets = selected

        values = final.to_numpy(dtype=float)
        volatility = float(np.sqrt(max(values @ covariance_values @ values, 0.0)))
        sector_weight = final.groupby(selected_sectors).sum()
        style_exposure = selected_exposures.loc[:, exposure_columns].T @ final
        change = (final - previous_actual).abs()
        utilization = (
            change
            * self.config.portfolio_value
            / (
                self.config.maximum_adv_participation
                * self.config.execution_days
                * adv.reindex(selected).replace(0.0, np.nan)
            )
        )

        covariance_condition_number = float(
            np.linalg.cond(covariance_values)
        )
        parametric_cvar = float(2.0627 * volatility)
        effective_n = float(1.0 / max(np.sum(values**2), _EPS))
        max_weight = float(final.max())
        max_sector_weight = float(sector_weight.max())
        max_style_exposure = float(style_exposure.abs().max())
        target_turnover_l1 = float(change.sum())
        max_participation_utilization = float(utilization.max())

        self.last_diagnostics = DynamicBarraAlphaDiagnostics(
            core_asset_count=len(core),
            added_asset_count=len(additions),
            selected_asset_count=len(selected),
            factor_count=selected_exposures.shape[1],
            industry_count=int(selected_sectors.nunique()),
            bayesian_rank_ic=rank_ic,
            covariance_condition_number=covariance_condition_number,
            predicted_weekly_volatility=volatility,
            parametric_weekly_cvar_95=parametric_cvar,
            effective_asset_count=effective_n,
            maximum_weight=max_weight,
            maximum_sector_weight=max_sector_weight,
            maximum_absolute_style_exposure=max_style_exposure,
            target_turnover_l1=target_turnover_l1,
            maximum_participation_utilization=max_participation_utilization,
            optimizer_success=optimizer_success,
        )

        return final

