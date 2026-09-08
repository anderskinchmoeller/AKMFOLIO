#!/usr/bin/env python3
"""Standalone adaptive consensus HRP research model.

This script needs only a wide weekly-return CSV and, for confirmatory research,
a matching point-in-time universe CSV. It deliberately imports no project-local
modules.

Architecture
------------
1. Causal PIT eligibility and rolling estimation windows.
2. Ledoit-Wolf, EWMA, PCA-factor, and downside covariance scenarios.
3. Causal validation-block covariance blending.
4. Consensus clustering across covariance scenarios and linkage methods.
5. Worst-case relative-variance-regret HRP splits with turnover anchoring.
6. Multi-resolution hierarchical alpha: cluster leadership, within-cluster
   relative value, graph diffusion, and sparse causal leader-lagger forecasts.
7. Every alpha expert is weighted only from returns observed after publication.
8. A zero-net active sleeve capped by scenario tracking error, active share,
   regime confidence, long-only bounds, no-trade bands, and turnover limits.

The model is experimental research software. It does not guarantee alpha and
must be evaluated on frozen, untouched data after all choices are finalized.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf


EPS = 1e-12
WEEKS_PER_YEAR = 52.0


@dataclass(frozen=True)
class ModelConfig:
    lookback_weeks: int = 260
    minimum_observations: int = 104
    maximum_interior_missing_fraction: float = 0.05
    rebalance_weeks: int = 4

    min_weight: float = 0.0
    max_weight: float = 0.10
    risk_contribution_cap: float = 0.08

    ewma_halflife: float = 26.0
    covariance_validation_weeks: int = 26
    covariance_prior_strength: float = 0.65
    covariance_temperature: float = 0.75
    pca_explained_variance: float = 0.90
    downside_quantile: float = 0.25

    tree_methods: tuple[str, ...] = ("average", "complete")
    consensus_linkage: str = "average"
    minimum_split: float = 0.10
    maximum_split: float = 0.90
    split_grid_size: int = 81
    split_turnover_penalty: float = 0.05

    signal_evaluation_weeks: int = 78
    signal_halflife: float = 26.0
    signal_prior_strength: float = 26.0
    signal_learning_rate: float = 0.50
    signal_weight_floor: float = 0.025

    hierarchy_resolutions: tuple[int, ...] = (4, 8, 16)
    lead_lag_lookback_weeks: int = 104
    lead_lag_max_parents: int = 5
    lead_lag_min_correlation: float = 0.05
    lead_lag_prior_strength: float = 104.0

    maximum_active_share: float = 0.05
    maximum_tracking_error: float = 0.04
    precision_loading: float = 0.10
    minimum_regime_scale: float = 0.15

    no_trade_l1: float = 0.02
    maximum_rebalance_turnover_l1: float = 0.20
    transaction_cost_bps: float = 10.0


@dataclass
class AllocationDiagnostics:
    covariance_weights: dict[str, float]
    cluster_stability: float
    regime_scale: float
    signal_confidence: float
    signal_weights: dict[str, float]
    hierarchical_signal_weight: float
    active_share: float
    worst_case_tracking_error: float
    mean_split_regret: float


def _nearest_psd(matrix: np.ndarray, floor_fraction: float = 1e-6) -> np.ndarray:
    x = np.asarray(matrix, dtype=float)
    x = 0.5 * (x + x.T)
    values, vectors = np.linalg.eigh(x)
    scale = max(float(np.mean(np.maximum(np.diag(x), 0.0))), EPS)
    values = np.maximum(values, floor_fraction * scale)
    out = (vectors * values) @ vectors.T
    return 0.5 * (out + out.T)


def _bounded_simplex(
    values: pd.Series,
    minimum: float,
    maximum: float,
) -> pd.Series:
    """Euclidean projection onto a long-only bounded simplex."""
    index = pd.Index(values.index)
    if len(index) == 0:
        return pd.Series(dtype=float)
    v = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    v = np.where(np.isfinite(v), v, 0.0)
    lower = np.full(len(index), max(float(minimum), 0.0), dtype=float)
    upper = np.full(len(index), min(float(maximum), 1.0), dtype=float)
    if lower.sum() > 1.0 + 1e-10 or upper.sum() < 1.0 - 1e-10:
        raise ValueError(
            f"Infeasible bounds for {len(index)} assets: "
            f"sum(lower)={lower.sum():.6f}, sum(upper)={upper.sum():.6f}."
        )
    lo = float(np.min(v - upper)) - 1.0
    hi = float(np.max(v - lower)) + 1.0
    for _ in range(160):
        level = 0.5 * (lo + hi)
        projected = np.clip(v - level, lower, upper)
        if projected.sum() > 1.0:
            lo = level
        else:
            hi = level
    out = np.clip(v - 0.5 * (lo + hi), lower, upper)
    residual = 1.0 - float(out.sum())
    if abs(residual) > 1e-10:
        slack = upper - out if residual > 0 else out - lower
        room = float(slack.sum())
        if room <= EPS:
            raise RuntimeError("Bounded-simplex projection has no residual capacity.")
        out += residual * slack / room
    result = pd.Series(out, index=index, dtype=float)
    if not np.isclose(result.sum(), 1.0, atol=1e-8):
        raise RuntimeError("Bounded-simplex projection failed sum-to-one.")
    return result


def _robust_z(values: pd.Series, index: pd.Index | None = None) -> pd.Series:
    target = pd.Index(values.index if index is None else index)
    x = pd.to_numeric(values, errors="coerce").reindex(target)
    valid = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3:
        return pd.Series(0.0, index=target)
    median = float(valid.median())
    mad = float((valid - median).abs().median())
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(valid.std(ddof=1))
    if not np.isfinite(scale) or scale <= EPS:
        return pd.Series(0.0, index=target)
    return ((x - median) / scale).clip(-3.0, 3.0).fillna(0.0)


def _compound(frame: pd.DataFrame, lookback: int, skip: int = 0) -> pd.Series:
    stop = len(frame) - int(skip)
    if stop <= 0:
        return pd.Series(0.0, index=frame.columns)
    start = max(0, stop - int(lookback))
    window = frame.iloc[start:stop]
    if window.empty:
        return pd.Series(0.0, index=frame.columns)
    return (1.0 + window).prod(axis=0) - 1.0


def _causal_fill(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan).fillna(frame.mean()).fillna(0.0)


def _prepare_window(
    returns: pd.DataFrame,
    active: pd.Series,
    config: ModelConfig,
) -> pd.DataFrame:
    x = returns.copy().astype(float).sort_index().replace([np.inf, -np.inf], np.nan)
    active = active.reindex(x.columns).fillna(False).astype(bool)
    observations = x.notna().sum()
    volatility = x.std(skipna=True)
    keep = active & (observations >= config.minimum_observations) & (volatility > EPS)
    x = x.loc[:, keep]
    while x.shape[1] >= 2:
        first_dates = x.apply(lambda column: column.first_valid_index()).dropna()
        if first_dates.empty:
            return x.iloc[:, 0:0]
        common = x.loc[first_dates.max():]
        missing = common.isna().mean()
        bad = missing[missing > config.maximum_interior_missing_fraction]
        if bad.empty:
            return common
        x = x.drop(columns=[bad.sort_values(ascending=False).index[0]])
    return x


def _ewma_covariance(frame: pd.DataFrame, halflife: float) -> np.ndarray:
    x = _causal_fill(frame).to_numpy(dtype=float)
    ages = np.arange(len(x) - 1, -1, -1, dtype=float)
    decay = np.exp(np.log(0.5) / max(float(halflife), 1.0))
    weights = decay ** ages
    weights /= weights.sum()
    mean = np.sum(x * weights[:, None], axis=0)
    centered = x - mean
    denominator = max(1.0 - float(np.sum(weights ** 2)), 1e-8)
    return _nearest_psd((centered * weights[:, None]).T @ centered / denominator)


def _pca_factor_covariance(frame: pd.DataFrame, explained: float) -> np.ndarray:
    x = _causal_fill(frame).to_numpy(dtype=float)
    lw = LedoitWolf().fit(x).covariance_
    values, vectors = np.linalg.eigh(lw)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], EPS)
    vectors = vectors[:, order]
    ratio = np.cumsum(values) / values.sum()
    k = int(np.searchsorted(ratio, np.clip(explained, 0.50, 0.99)) + 1)
    factor = (vectors[:, :k] * values[:k]) @ vectors[:, :k].T
    residual = np.maximum(np.diag(lw - factor), EPS)
    return _nearest_psd(factor + np.diag(residual))


def _downside_covariance(frame: pd.DataFrame, quantile: float) -> np.ndarray:
    x = _causal_fill(frame)
    market = x.mean(axis=1)
    threshold = float(market.quantile(np.clip(quantile, 0.05, 0.50)))
    downside = x.loc[market <= threshold]
    if len(downside) < max(12, x.shape[1] // 10):
        downside = x
    return _nearest_psd(LedoitWolf().fit(downside.to_numpy()).covariance_)


def _covariance_candidates(frame: pd.DataFrame, config: ModelConfig) -> dict[str, np.ndarray]:
    x = _causal_fill(frame)
    return {
        "ledoit_wolf": _nearest_psd(LedoitWolf().fit(x.to_numpy()).covariance_),
        "ewma": _ewma_covariance(x, config.ewma_halflife),
        "pca_factor": _pca_factor_covariance(x, config.pca_explained_variance),
        "downside": _downside_covariance(x, config.downside_quantile),
    }


def _gaussian_covariance_loss(forecast: np.ndarray, realized: np.ndarray) -> float:
    forecast = _nearest_psd(forecast)
    realized = _nearest_psd(realized)
    sign, logdet = np.linalg.slogdet(forecast)
    if sign <= 0 or not np.isfinite(logdet):
        return np.inf
    precision = np.linalg.pinv(forecast, rcond=1e-8)
    return float((logdet + np.trace(precision @ realized)) / len(forecast))


def covariance_scenarios(
    frame: pd.DataFrame,
    config: ModelConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    names = ("ledoit_wolf", "ewma", "pca_factor", "downside")
    prior = np.asarray([0.35, 0.30, 0.20, 0.15], dtype=float)
    validation = int(config.covariance_validation_weeks)
    weights = prior.copy()
    if len(frame) >= config.minimum_observations + validation:
        calibration = frame.iloc[:-validation]
        holdout = _causal_fill(frame.iloc[-validation:])
        forecasts = _covariance_candidates(calibration, config)
        realized = _nearest_psd(holdout.cov().to_numpy(dtype=float))
        losses = np.asarray(
            [_gaussian_covariance_loss(forecasts[name], realized) for name in names]
        )
        finite = np.isfinite(losses)
        if finite.sum() >= 2:
            replacement = float(np.max(losses[finite]) + 10.0)
            losses = np.where(finite, losses, replacement)
            scale = max(float(np.std(losses[finite])), 1e-6)
            logits = -(losses - losses.min()) / (
                scale * max(config.covariance_temperature, 1e-3)
            )
            empirical = np.exp(np.clip(logits, -30.0, 0.0))
            empirical /= empirical.sum()
            strength = np.clip(config.covariance_prior_strength, 0.0, 1.0)
            weights = strength * prior + (1.0 - strength) * empirical
            weights /= weights.sum()
    raw = _covariance_candidates(frame, config)
    scenarios = {
        name: pd.DataFrame(raw[name], index=frame.columns, columns=frame.columns)
        for name in names
    }
    blend = sum(weights[i] * raw[name] for i, name in enumerate(names))
    scenarios["blend"] = pd.DataFrame(
        _nearest_psd(blend), index=frame.columns, columns=frame.columns
    )
    return scenarios, {name: float(weights[i]) for i, name in enumerate(names)}


def _covariance_to_correlation(covariance: pd.DataFrame) -> pd.DataFrame:
    x = covariance.to_numpy(dtype=float)
    inverse_vol = 1.0 / np.sqrt(np.maximum(np.diag(x), EPS))
    corr = np.clip(inverse_vol[:, None] * x * inverse_vol[None, :], -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=covariance.index, columns=covariance.columns)


def _tree(correlation: pd.DataFrame, method: str) -> np.ndarray:
    distance = np.sqrt(np.maximum(0.5 * (1.0 - correlation.to_numpy()), 0.0))
    np.fill_diagonal(distance, 0.0)
    return linkage(squareform(distance, checks=False), method=method)


def _coclustering(trees: Iterable[np.ndarray], assets: list[str]) -> pd.DataFrame:
    trees = list(trees)
    n = len(assets)
    k = int(np.clip(round(np.sqrt(n)), 2, n))
    matrix = np.zeros((n, n), dtype=float)
    for candidate in trees:
        labels = fcluster(candidate, t=k, criterion="maxclust")
        matrix += (labels[:, None] == labels[None, :]).astype(float)
    matrix /= len(trees)
    np.fill_diagonal(matrix, 1.0)
    return pd.DataFrame(matrix, index=assets, columns=assets)


def _consensus_tree(coclustering: pd.DataFrame, method: str) -> np.ndarray:
    similarity = coclustering.to_numpy(dtype=float)
    distance = np.sqrt(np.maximum(1.0 - similarity, 0.0))
    np.fill_diagonal(distance, 0.0)
    return linkage(squareform(distance, checks=False), method=method)


def _ordered_assets(tree: np.ndarray, assets: list[str]) -> list[str]:
    n = len(assets)
    clusters = {i: [i] for i in range(n)}
    for row_index, row in enumerate(tree):
        left, right = int(row[0]), int(row[1])
        clusters[n + row_index] = clusters[left] + clusters[right]
    return [assets[position] for position in clusters[n + len(tree) - 1]]


def _inverse_variance(covariance: pd.DataFrame, assets: list[str]) -> np.ndarray:
    variance = np.diag(covariance.loc[assets, assets].to_numpy(dtype=float))
    raw = 1.0 / np.maximum(variance, EPS)
    return raw / raw.sum()


def _branch_moments(
    scenario: pd.DataFrame,
    reference: pd.DataFrame,
    left: list[str],
    right: list[str],
) -> tuple[float, float, float]:
    wl = _inverse_variance(reference, left)
    wr = _inverse_variance(reference, right)
    ll = scenario.loc[left, left].to_numpy(dtype=float)
    rr = scenario.loc[right, right].to_numpy(dtype=float)
    lr = scenario.loc[left, right].to_numpy(dtype=float)
    return (
        max(float(wl @ ll @ wl), EPS),
        max(float(wr @ rr @ wr), EPS),
        float(wl @ lr @ wr),
    )


def _robust_split(
    scenarios: dict[str, pd.DataFrame],
    reference: pd.DataFrame,
    left: list[str],
    right: list[str],
    previous_split: float | None,
    config: ModelConfig,
) -> tuple[float, float]:
    grid = np.linspace(
        config.minimum_split,
        config.maximum_split,
        max(21, int(config.split_grid_size)),
    )
    regrets = []
    for scenario in scenarios.values():
        vl, vr, cross = _branch_moments(scenario, reference, left, right)
        risk = (
            grid * grid * vl
            + (1.0 - grid) ** 2 * vr
            + 2.0 * grid * (1.0 - grid) * cross
        )
        risk = np.maximum(risk, EPS)
        regrets.append(risk / risk.min() - 1.0)
    worst = np.max(np.vstack(regrets), axis=0)
    objective = worst.copy()
    if previous_split is not None:
        objective += config.split_turnover_penalty * np.abs(grid - previous_split)
    objective += 1e-10 * np.abs(grid - 0.5)
    selected = int(np.argmin(objective))
    return float(grid[selected]), float(worst[selected])


def _risk_contribution_regularize(
    weights: pd.Series,
    covariance: pd.DataFrame,
    cap: float,
) -> pd.Series:
    if cap <= 0 or len(weights) < 2:
        return weights
    cov = covariance.loc[weights.index, weights.index].to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    inverse_vol = 1.0 / np.sqrt(np.maximum(np.diag(cov), EPS))
    inverse_vol /= inverse_vol.sum()
    effective_cap = max(float(cap), 1.5 / len(w))
    for _ in range(40):
        marginal = cov @ w
        variance = float(w @ marginal)
        if variance <= EPS:
            break
        contribution = np.maximum(w * marginal / variance, 0.0)
        if contribution.sum() > 0:
            contribution /= contribution.sum()
        if contribution.max() <= effective_cap + 1e-6:
            break
        w = 0.90 * w + 0.10 * inverse_vol
        w /= w.sum()
    return pd.Series(w, index=weights.index)


def _regret_aware_core(
    scenarios: dict[str, pd.DataFrame],
    consensus: np.ndarray,
    assets: list[str],
    previous: pd.Series | None,
    aggressiveness: float,
    config: ModelConfig,
) -> tuple[pd.Series, float]:
    reference = scenarios["blend"]
    order = _ordered_assets(consensus, assets)
    weights = pd.Series(1.0, index=order)
    previous = None if previous is None else previous.reindex(order).fillna(0.0)
    regrets: list[float] = []

    def recurse(cluster: list[str]) -> None:
        if len(cluster) <= 1:
            return
        midpoint = len(cluster) // 2
        left, right = cluster[:midpoint], cluster[midpoint:]
        prior_split = None
        if previous is not None:
            mass = float(previous.loc[cluster].sum())
            if mass > EPS:
                prior_split = float(previous.loc[left].sum() / mass)
        raw, regret = _robust_split(
            scenarios, reference, left, right, prior_split, config
        )
        allocation = 0.5 + np.clip(aggressiveness, 0.0, 1.0) * (raw - 0.5)
        allocation = float(np.clip(allocation, config.minimum_split, config.maximum_split))
        weights.loc[left] *= allocation
        weights.loc[right] *= 1.0 - allocation
        regrets.append(regret)
        recurse(left)
        recurse(right)

    recurse(order)
    weights /= weights.sum()
    weights = _risk_contribution_regularize(
        weights, reference, config.risk_contribution_cap
    )
    weights = _bounded_simplex(weights, config.min_weight, config.max_weight)
    return weights, float(np.mean(regrets)) if regrets else 0.0


def _legacy_hrp(covariance: pd.DataFrame, config: ModelConfig) -> pd.Series:
    assets = list(covariance.columns)
    tree = _tree(_covariance_to_correlation(covariance), "average")
    order = _ordered_assets(tree, assets)
    weights = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        next_level: list[list[str]] = []
        for cluster in clusters:
            if len(cluster) > 1:
                midpoint = len(cluster) // 2
                next_level.extend([cluster[:midpoint], cluster[midpoint:]])
        clusters = next_level
        for position in range(0, len(clusters), 2):
            left, right = clusters[position], clusters[position + 1]
            wl = _inverse_variance(covariance, left)
            wr = _inverse_variance(covariance, right)
            vl = float(wl @ covariance.loc[left, left].to_numpy() @ wl)
            vr = float(wr @ covariance.loc[right, right].to_numpy() @ wr)
            allocation = vr / max(vl + vr, EPS)
            weights.loc[left] *= allocation
            weights.loc[right] *= 1.0 - allocation
    weights /= weights.sum()
    return _bounded_simplex(weights, config.min_weight, config.max_weight)


SIGNAL_PRIOR = pd.Series(
    {
        "momentum_12_1": 0.15,
        "momentum_6_1": 0.08,
        "residual_momentum": 0.10,
        "trend_quality": 0.06,
        "low_volatility": 0.06,
        "downside_resilience": 0.04,
        "drawdown_resilience": 0.03,
        "short_reversal": 0.03,
        "cluster_momentum": 0.12,
        "cluster_breadth": 0.07,
        "within_cluster_continuation": 0.07,
        "within_cluster_reversal": 0.06,
        "hierarchy_diffusion": 0.06,
        "sparse_lead_lag": 0.07,
    },
    dtype=float,
)

HIERARCHICAL_SIGNALS = frozenset(
    {
        "cluster_momentum",
        "cluster_breadth",
        "within_cluster_continuation",
        "within_cluster_reversal",
        "hierarchy_diffusion",
        "sparse_lead_lag",
    }
)


def _multiresolution_hierarchy_signals(
    returns: pd.DataFrame,
    consensus_tree: np.ndarray,
    coclustering: pd.DataFrame,
    config: ModelConfig,
) -> dict[str, pd.Series]:
    """Create forecasts from the hierarchy rather than merely allocating on it.

    The tree is cut at several resolutions.  This prevents a single arbitrary
    cut from defining the economic groups and makes each forecast an average
    over coarse and fine neighborhoods.
    """
    assets = returns.columns
    n_assets = len(assets)
    medium = _compound(
        returns, min(22, len(returns) - 4), skip=min(4, len(returns) - 1)
    )
    short = _compound(returns, min(4, len(returns)))
    twelve_week = _compound(returns, min(12, len(returns)))
    cluster_momentum_parts: list[pd.Series] = []
    breadth_parts: list[pd.Series] = []
    continuation_parts: list[pd.Series] = []
    reversal_parts: list[pd.Series] = []
    valid_resolutions = sorted(
        {
            int(np.clip(resolution, 2, max(2, n_assets - 1)))
            for resolution in config.hierarchy_resolutions
            if n_assets >= 3
        }
    )
    if not valid_resolutions:
        valid_resolutions = [1]
    for resolution in valid_resolutions:
        if resolution == 1:
            labels = pd.Series(1, index=assets)
        else:
            labels = pd.Series(
                fcluster(consensus_tree, t=resolution, criterion="maxclust"),
                index=assets,
            )
        group_momentum = medium.groupby(labels).transform("mean")
        group_short = short.groupby(labels).transform("mean")
        positive = twelve_week.gt(0.0).astype(float)
        group_breadth = positive.groupby(labels).transform("mean")
        cluster_momentum_parts.append(_robust_z(group_momentum, assets))
        breadth_parts.append(_robust_z(group_breadth, assets))
        continuation_parts.append(_robust_z(medium - group_momentum, assets))
        reversal_parts.append(_robust_z(-(short - group_short), assets))

    def average(parts: list[pd.Series]) -> pd.Series:
        return pd.concat(parts, axis=1).mean(axis=1).reindex(assets).fillna(0.0)

    affinity = coclustering.reindex(index=assets, columns=assets).to_numpy(dtype=float)
    np.fill_diagonal(affinity, 0.0)
    row_sum = affinity.sum(axis=1)
    normalized_affinity = np.divide(
        affinity,
        row_sum[:, None],
        out=np.zeros_like(affinity),
        where=row_sum[:, None] > EPS,
    )
    peer_impulse = pd.Series(
        normalized_affinity @ _robust_z(short, assets).to_numpy(dtype=float),
        index=assets,
    )

    lag_window = returns.iloc[-min(config.lead_lag_lookback_weeks, len(returns)):]
    lead_lag = pd.Series(0.0, index=assets)
    if len(lag_window) >= 26:
        x = lag_window.iloc[:-1].to_numpy(dtype=float)
        y = lag_window.iloc[1:].to_numpy(dtype=float)
        x -= x.mean(axis=0, keepdims=True)
        y -= y.mean(axis=0, keepdims=True)
        scale_x = np.sqrt(np.maximum(np.sum(x * x, axis=0), EPS))
        scale_y = np.sqrt(np.maximum(np.sum(y * y, axis=0), EPS))
        lag_correlation = (x.T @ y) / np.maximum(
            scale_x[:, None] * scale_y[None, :], EPS
        )
        lag_correlation *= affinity
        np.fill_diagonal(lag_correlation, 0.0)
        shrink = (len(lag_window) - 1) / (
            len(lag_window) - 1 + config.lead_lag_prior_strength
        )
        lag_correlation *= shrink
        latest = lag_window.iloc[-1]
        latest_z = _robust_z(latest, assets).to_numpy(dtype=float)
        prediction = np.zeros(n_assets, dtype=float)
        for target in range(n_assets):
            coefficients = lag_correlation[:, target].copy()
            eligible = np.flatnonzero(
                np.abs(coefficients) >= config.lead_lag_min_correlation * shrink
            )
            if eligible.size == 0:
                continue
            order = eligible[np.argsort(np.abs(coefficients[eligible]))[::-1]]
            chosen = order[: config.lead_lag_max_parents]
            denominator = float(np.abs(coefficients[chosen]).sum())
            if denominator > EPS:
                prediction[target] = float(
                    coefficients[chosen] @ latest_z[chosen] / denominator
                )
        lead_lag = pd.Series(prediction, index=assets)

    return {
        "cluster_momentum": average(cluster_momentum_parts),
        "cluster_breadth": average(breadth_parts),
        "within_cluster_continuation": average(continuation_parts),
        "within_cluster_reversal": average(reversal_parts),
        "hierarchy_diffusion": _robust_z(peer_impulse, assets),
        "sparse_lead_lag": _robust_z(lead_lag, assets),
    }


def _signal_components(
    frame: pd.DataFrame,
    consensus_tree: np.ndarray,
    coclustering: pd.DataFrame,
    config: ModelConfig,
) -> pd.DataFrame:
    returns = _causal_fill(frame)
    assets = returns.columns
    if len(returns) < 20:
        return pd.DataFrame(0.0, index=assets, columns=SIGNAL_PRIOR.index)
    momentum_12_1 = _compound(returns, min(48, len(returns) - 4), skip=min(4, len(returns) - 1))
    momentum_6_1 = _compound(returns, min(22, len(returns) - 4), skip=min(4, len(returns) - 1))
    short_reversal = -_compound(returns, min(4, len(returns)))
    recent = returns.iloc[-min(26, len(returns)):]
    volatility = recent.std(ddof=1)
    low_volatility = -volatility
    downside_resilience = -np.sqrt((recent.clip(upper=0.0) ** 2).mean())
    trend_quality = recent.mean() / volatility.replace(0.0, np.nan)
    wealth = (1.0 + returns.iloc[-min(52, len(returns)):]).cumprod()
    drawdown_resilience = wealth.iloc[-1] / wealth.cummax().max(axis=0) - 1.0

    residual_window = returns.iloc[-min(52, len(returns)):]
    market = residual_window.mean(axis=1)
    market_centered = market - market.mean()
    denominator = float(market_centered @ market_centered)
    centered = residual_window - residual_window.mean()
    if denominator > EPS:
        beta = (market_centered.to_numpy()[:, None] * centered.to_numpy()).sum(axis=0) / denominator
        residual = centered.to_numpy() - market_centered.to_numpy()[:, None] * beta
        residual = pd.DataFrame(residual, index=residual_window.index, columns=assets)
        residual_momentum = _compound(residual.clip(lower=-0.95), len(residual))
    else:
        residual_momentum = pd.Series(0.0, index=assets)

    raw = {
        "momentum_12_1": momentum_12_1,
        "momentum_6_1": momentum_6_1,
        "residual_momentum": residual_momentum,
        "trend_quality": trend_quality,
        "low_volatility": low_volatility,
        "downside_resilience": downside_resilience,
        "drawdown_resilience": drawdown_resilience,
        "short_reversal": short_reversal,
    }
    raw.update(
        _multiresolution_hierarchy_signals(
            returns, consensus_tree, coclustering, config
        )
    )
    return pd.DataFrame(
        {name: _robust_z(values, assets) for name, values in raw.items()},
        index=assets,
    )


class OnlineSignalEnsemble:
    """Causally score signals after publication, across the live PIT universe.

    The learner never reconstructs a historical signal using today's surviving
    securities. At each allocation it publishes component positions. Subsequent
    weekly returns update expert payoffs, and only those realized payoffs can
    change the next allocation's expert weights.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.positions: pd.DataFrame | None = None
        self.payoff_history = pd.DataFrame(columns=SIGNAL_PRIOR.index, dtype=float)

    def observe(self, realized_returns: pd.Series) -> None:
        if self.positions is None or self.positions.empty:
            return
        realized = pd.to_numeric(realized_returns, errors="coerce")
        payoffs: dict[str, float] = {}
        for name in self.positions.columns:
            signal = self.positions[name]
            valid = signal.notna() & realized.reindex(signal.index).notna()
            aligned = realized.reindex(signal.index)
            gross = float(signal.loc[valid].abs().sum())
            payoffs[name] = (
                float(signal.loc[valid] @ aligned.loc[valid]) / gross
                if gross > EPS else 0.0
            )
        row = pd.DataFrame([payoffs], index=[realized_returns.name])
        if self.payoff_history.empty:
            self.payoff_history = row.reindex(columns=SIGNAL_PRIOR.index)
        else:
            self.payoff_history = pd.concat(
                [self.payoff_history, row], axis=0
            )
        self.payoff_history = self.payoff_history[~self.payoff_history.index.duplicated(keep="last")]
        self.payoff_history = self.payoff_history.tail(
            max(16, int(self.config.signal_evaluation_weeks))
        )

    def _weights(self) -> tuple[pd.Series, pd.Series]:
        prior = SIGNAL_PRIOR / SIGNAL_PRIOR.sum()
        payoffs = self.payoff_history.reindex(columns=prior.index)
        if len(payoffs) < 8:
            return prior.copy(), pd.Series(0.0, index=prior.index)
        ages = np.arange(len(payoffs) - 1, -1, -1, dtype=float)
        decay = np.exp(np.log(0.5) / max(self.config.signal_halflife, 1.0))
        observation_weights = decay ** ages
        observation_weights /= observation_weights.sum()
        values = payoffs.fillna(0.0).to_numpy(dtype=float)
        mean = np.sum(values * observation_weights[:, None], axis=0)
        centered = values - mean
        variance = np.sum(centered ** 2 * observation_weights[:, None], axis=0)
        information_ratio = pd.Series(
            np.sqrt(WEEKS_PER_YEAR) * mean / np.sqrt(np.maximum(variance, EPS)),
            index=payoffs.columns,
        ).clip(-2.0, 2.0)
        effective_n = 1.0 / float(np.sum(observation_weights ** 2))
        evidence = effective_n / (
            effective_n + self.config.signal_prior_strength
        )
        logits = self.config.signal_learning_rate * evidence * information_ratio
        raw = prior * np.exp(logits.clip(-4.0, 4.0))
        weights = raw / raw.sum()
        floor = np.clip(
            self.config.signal_weight_floor, 0.0, 1.0 / len(weights)
        )
        weights = floor + (1.0 - floor * len(weights)) * weights
        return weights / weights.sum(), information_ratio

    def publish(
        self,
        frame: pd.DataFrame,
        consensus_tree: np.ndarray,
        coclustering: pd.DataFrame,
    ) -> tuple[pd.Series, dict[str, float], float]:
        weights, information_ratio = self._weights()
        current = _signal_components(
            frame, consensus_tree, coclustering, self.config
        )
        score = current.mul(weights, axis=1).sum(axis=1)
        agreement = current.apply(np.sign).mul(weights, axis=1).sum(axis=1).abs()
        score = _robust_z(
            score * np.sqrt(agreement.clip(0.0, 1.0)), current.index
        )
        positive_ir = float((weights * information_ratio.clip(lower=0.0)).sum())
        ir_confidence = float(np.clip(positive_ir / 1.5, 0.0, 1.0))
        agreement_confidence = float(np.clip(agreement.mean(), 0.0, 1.0))
        confidence = float(np.sqrt(ir_confidence * agreement_confidence))
        self.positions = current.copy()
        return (
            score,
            {key: float(value) for key, value in weights.items()},
            confidence,
        )


def _regime_scale(frame: pd.DataFrame, minimum: float) -> float:
    returns = _causal_fill(frame)
    market = returns.mean(axis=1)
    if len(market) < 30:
        return 0.50
    window = min(26, len(market))
    rolling_vol = market.rolling(window).std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)
    history = rolling_vol.dropna()
    current_vol = float(history.iloc[-1])
    volatility_percentile = float((history <= current_vol).mean())
    trend = float(_compound(market.to_frame("market"), window).iloc[0])
    breadth = float((_compound(returns, window) > 0.0).mean())
    wealth = (1.0 + market.iloc[-min(52, len(market)):]).cumprod()
    drawdown = float(wealth.iloc[-1] / wealth.cummax().iloc[-1] - 1.0)
    vol_scale = np.clip(1.10 - 0.75 * volatility_percentile, 0.30, 1.0)
    trend_scale = 1.0 if trend >= 0.0 else 0.65
    breadth_scale = np.clip(0.55 + 0.60 * breadth, 0.55, 1.0)
    drawdown_scale = np.clip(1.0 + 2.5 * drawdown, 0.45, 1.0)
    return float(np.clip(
        vol_scale * trend_scale * breadth_scale * drawdown_scale,
        minimum,
        1.0,
    ))


def _scenario_active_overlay(
    core: pd.Series,
    alpha: pd.Series,
    scenarios: dict[str, pd.DataFrame],
    regime_scale: float,
    signal_confidence: float,
    config: ModelConfig,
) -> tuple[pd.Series, float, float]:
    assets = core.index
    covariance = scenarios["blend"].loc[assets, assets].to_numpy(dtype=float)
    mean_variance = max(float(np.trace(covariance) / len(assets)), EPS)
    loaded = _nearest_psd(
        covariance + config.precision_loading * mean_variance * np.eye(len(assets))
    )
    precision = np.linalg.pinv(loaded, rcond=1e-8)
    forecast = alpha.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    ones = np.ones(len(assets))
    denominator = float(ones @ precision @ ones)
    intercept = float(ones @ precision @ forecast) / max(denominator, EPS)
    direction = precision @ (forecast - intercept * ones)
    direction = np.where(np.isfinite(direction), direction, 0.0)
    direction -= direction.mean()
    gross = float(np.abs(direction).sum())
    if gross <= EPS or signal_confidence <= 0.0:
        return core, 0.0, 0.0
    direction *= 2.0 / gross

    active_budget = config.maximum_active_share * regime_scale * signal_confidence
    worst_variance = max(
        float(direction @ scenario.loc[assets, assets].to_numpy() @ direction)
        for scenario in scenarios.values()
    )
    if worst_variance > EPS:
        tracking_budget = (
            config.maximum_tracking_error / np.sqrt(WEEKS_PER_YEAR * worst_variance)
        )
        active_budget = min(active_budget, tracking_budget)
    proposed = core + active_budget * pd.Series(direction, index=assets)
    target = _bounded_simplex(proposed, config.min_weight, config.max_weight)
    active_share = 0.5 * float((target - core).abs().sum())
    permitted = config.maximum_active_share * regime_scale * signal_confidence
    if active_share > permitted + 1e-12:
        target = core + (target - core) * permitted / active_share
        target = _bounded_simplex(target, config.min_weight, config.max_weight)
        active_share = 0.5 * float((target - core).abs().sum())
    delta = (target - core).to_numpy(dtype=float)
    worst_te = np.sqrt(
        WEEKS_PER_YEAR
        * max(
            float(delta @ scenario.loc[assets, assets].to_numpy() @ delta)
            for scenario in scenarios.values()
        )
    )
    return target, active_share, float(worst_te)


class AdaptiveConsensusHRP:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.previous_coclustering: pd.DataFrame | None = None
        self.previous_core: pd.Series | None = None
        self.signal_ensemble = OnlineSignalEnsemble(self.config)
        self.last_core: pd.Series | None = None
        self.last_legacy: pd.Series | None = None
        self.last_diagnostics: AllocationDiagnostics | None = None

    def allocate(self, frame: pd.DataFrame) -> pd.Series:
        if frame.shape[1] < 2:
            raise ValueError("AdaptiveConsensusHRP requires at least two assets.")
        scenarios, covariance_weights = covariance_scenarios(frame, self.config)
        assets = list(frame.columns)
        trees = []
        for scenario in scenarios.values():
            correlation = _covariance_to_correlation(scenario)
            trees.extend(_tree(correlation, method) for method in self.config.tree_methods)
        coclustering = _coclustering(trees, assets)
        stability = 1.0
        if self.previous_coclustering is not None:
            common = self.previous_coclustering.index.intersection(coclustering.index)
            if len(common) >= 4:
                previous = self.previous_coclustering.loc[common, common].to_numpy()
                current = coclustering.loc[common, common].to_numpy()
                mask = ~np.eye(len(common), dtype=bool)
                stability = 1.0 - float(np.abs(current - previous)[mask].mean())
                stability = float(np.clip(stability, 0.0, 1.0))
        aggressiveness = float(np.clip(stability / 0.70, 0.35, 1.0))
        consensus = _consensus_tree(coclustering, self.config.consensus_linkage)
        core, mean_regret = _regret_aware_core(
            scenarios,
            consensus,
            assets,
            self.previous_core,
            aggressiveness,
            self.config,
        )
        alpha, signal_weights, signal_confidence = self.signal_ensemble.publish(
            frame, consensus, coclustering
        )
        regime = _regime_scale(frame, self.config.minimum_regime_scale)
        final, active_share, worst_te = _scenario_active_overlay(
            core,
            alpha,
            scenarios,
            regime,
            signal_confidence,
            self.config,
        )
        self.last_core = core.copy()
        self.last_legacy = _legacy_hrp(scenarios["blend"], self.config)
        self.last_diagnostics = AllocationDiagnostics(
            covariance_weights=covariance_weights,
            cluster_stability=stability,
            regime_scale=regime,
            signal_confidence=signal_confidence,
            signal_weights=signal_weights,
            hierarchical_signal_weight=float(
                sum(signal_weights.get(name, 0.0) for name in HIERARCHICAL_SIGNALS)
            ),
            active_share=active_share,
            worst_case_tracking_error=worst_te,
            mean_split_regret=mean_regret,
        )
        self.previous_coclustering = coclustering
        self.previous_core = core.copy()
        return final

    def observe(self, realized_returns: pd.Series) -> None:
        self.signal_ensemble.observe(realized_returns)


def _load_wide_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(frame.index)
    frame.columns = frame.columns.astype(str)
    frame = frame.apply(pd.to_numeric, errors="coerce").sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"{path} contains duplicate dates.")
    return frame


def _load_pit(
    path: str | Path | None,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(True, index=returns.index, columns=returns.columns)
    pit = _load_wide_csv(path)
    pit = pit.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    return pit.astype(bool)


def _turnover_limited_target(
    current: pd.Series,
    target: pd.Series,
    config: ModelConfig,
) -> tuple[pd.Series, float]:
    universe = current.index.union(target.index)
    current = current.reindex(universe).fillna(0.0)
    target = target.reindex(universe).fillna(0.0)
    if current.sum() <= EPS:
        return target, 0.0

    # Exit securities that have left the PIT universe immediately. Turnover
    # controls apply only to the discretionary move after those mandatory exits.
    eligible = target > EPS
    mandatory = current.copy()
    lost_mass = float(mandatory.loc[~eligible].sum())
    mandatory.loc[~eligible] = 0.0
    if lost_mass > EPS:
        destination = target.loc[eligible]
        mandatory.loc[eligible] += lost_mass * destination / destination.sum()
    mandatory /= mandatory.sum()
    mandatory_l1 = float((mandatory - current).abs().sum())
    discretionary_l1 = float((target - mandatory).abs().sum())
    if discretionary_l1 < config.no_trade_l1 and mandatory_l1 <= EPS:
        return current / current.sum(), 0.0
    discretionary_budget = max(
        0.0,
        config.maximum_rebalance_turnover_l1 - mandatory_l1,
    )
    if discretionary_l1 > discretionary_budget + EPS:
        fraction = discretionary_budget / discretionary_l1
        target = mandatory + fraction * (target - mandatory)
    target = target.clip(lower=0.0)
    target /= target.sum()
    return target, float((target - current).abs().sum())


def _metrics(returns: pd.Series, turnover: pd.Series) -> dict[str, float]:
    values = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {key: 0.0 for key in (
            "cagr", "sharpe", "volatility", "max_drawdown",
            "annual_turnover", "total_return", "n_obs"
        )}
    wealth = (1.0 + values).cumprod()
    years = len(values) / WEEKS_PER_YEAR
    cagr = float(wealth.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    volatility = float(values.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR))
    sharpe = (
        float(values.mean() / values.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR))
        if values.std(ddof=1) > EPS else 0.0
    )
    drawdown = wealth / wealth.cummax() - 1.0
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "volatility": volatility,
        "max_drawdown": float(drawdown.min()),
        "annual_turnover": float(turnover.reindex(values.index).fillna(0.0).mean() * WEEKS_PER_YEAR),
        "total_return": float(wealth.iloc[-1] - 1.0),
        "n_obs": float(len(values)),
    }


def run_backtest(
    returns: pd.DataFrame,
    pit: pd.DataFrame,
    config: ModelConfig,
    evaluation_start: pd.Timestamp | None,
    missing_held_return: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[dict[str, object]],
    pd.DataFrame,
]:
    strategies = (
        "equal_weight",
        "inverse_volatility",
        "legacy_hrp",
        "robust_consensus_core",
        "hierarchical_alpha_hrp",
    )
    allocator = AdaptiveConsensusHRP(config)
    current = {
        strategy: pd.Series(0.0, index=returns.columns) for strategy in strategies
    }
    portfolio_returns = pd.DataFrame(np.nan, index=returns.index, columns=strategies)
    turnover = pd.DataFrame(0.0, index=returns.index, columns=strategies)
    integrated_weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    diagnostics: list[dict[str, object]] = []

    for position in range(config.minimum_observations, len(returns)):
        date = returns.index[position]
        prior_membership = pit.loc[returns.index[position - 1]]
        forced_exit = any(
            bool(
                (
                    (weights > EPS)
                    & ~prior_membership.reindex(weights.index).fillna(False).astype(bool)
                ).any()
            )
            for weights in current.values()
        )
        should_rebalance = (
            (position - config.minimum_observations) % config.rebalance_weeks == 0
            or all(weights.sum() <= EPS for weights in current.values())
            or forced_exit
        )
        if should_rebalance:
            start = max(0, position - config.lookback_weeks)
            raw_window = returns.iloc[start:position]
            membership_date = raw_window.index[-1]
            active = pit.loc[membership_date]
            window = _prepare_window(raw_window, active, config)
            if window.shape[1] >= max(2, int(np.ceil(1.0 / config.max_weight))):
                integrated = allocator.allocate(window)
                core = allocator.last_core
                legacy = allocator.last_legacy
                inverse_volatility = 1.0 / window.std(ddof=1).replace(0.0, np.nan)
                inverse_volatility = inverse_volatility.fillna(0.0)
                inverse_volatility /= inverse_volatility.sum()
                targets = {
                    "equal_weight": pd.Series(1.0 / window.shape[1], index=window.columns),
                    "inverse_volatility": _bounded_simplex(
                        inverse_volatility, config.min_weight, config.max_weight
                    ),
                    "legacy_hrp": legacy,
                    "robust_consensus_core": core,
                    "hierarchical_alpha_hrp": integrated,
                }
                for strategy, target in targets.items():
                    target = target.reindex(returns.columns).fillna(0.0)
                    updated, traded = _turnover_limited_target(
                        current[strategy], target, config
                    )
                    current[strategy] = updated
                    turnover.loc[date, strategy] = traded
                if allocator.last_diagnostics is not None:
                    diagnostics.append(
                        {"date": str(date.date()), **asdict(allocator.last_diagnostics)}
                    )

        realized = returns.iloc[position]
        for strategy in strategies:
            weights = current[strategy]
            held = weights > EPS
            missing = held & realized.isna()
            if missing.any() and missing_held_return == "raise":
                examples = realized.index[missing].tolist()[:10]
                raise RuntimeError(
                    f"Missing held return on {date.date()} for {strategy}: {examples}. "
                    "Use --missing-held-return zero only after auditing these rows."
                )
            clean_return = realized.fillna(0.0)
            gross = float(weights @ clean_return)
            cost = turnover.loc[date, strategy] * config.transaction_cost_bps / 10_000.0
            portfolio_returns.loc[date, strategy] = gross - cost
            grown = weights * (1.0 + clean_return)
            current[strategy] = grown / grown.sum() if grown.sum() > EPS else weights
        allocator.observe(realized.rename(date))
        integrated_weights.loc[date] = current["hierarchical_alpha_hrp"]

    score_returns = portfolio_returns
    score_turnover = turnover
    if evaluation_start is not None:
        score_returns = score_returns.loc[score_returns.index >= evaluation_start]
        score_turnover = score_turnover.reindex(score_returns.index)
    metrics = pd.DataFrame(
        {
            strategy: _metrics(score_returns[strategy], score_turnover[strategy])
            for strategy in strategies
        }
    ).T
    metrics.index.name = "model"
    return portfolio_returns, turnover, integrated_weights, diagnostics, metrics


def _self_test() -> None:
    rng = np.random.default_rng(7)
    weeks, assets = 360, 30
    market = rng.normal(0.001, 0.018, weeks)
    values = np.column_stack(
        [0.45 * market + rng.normal(0.0002, 0.020, weeks) for _ in range(assets)]
    )
    returns = pd.DataFrame(
        values,
        index=pd.date_range("2015-01-02", periods=weeks, freq="W-FRI"),
        columns=[str(10_000 + i) for i in range(assets)],
    )
    pit = pd.DataFrame(True, index=returns.index, columns=returns.columns)
    config = ModelConfig(
        lookback_weeks=156,
        minimum_observations=78,
        max_weight=0.10,
        signal_evaluation_weeks=30,
    )
    outputs = run_backtest(returns, pit, config, None, "raise")
    portfolio_returns, _, weights, diagnostics, metrics = outputs
    assert np.isfinite(metrics.to_numpy()).all()
    invested = weights.sum(axis=1) > EPS
    assert np.allclose(weights.loc[invested].sum(axis=1), 1.0)
    assert (weights >= -1e-12).all().all()
    assert weights.max().max() <= config.max_weight + 1e-8
    assert diagnostics
    assert portfolio_returns.notna().sum().min() > 0
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone hierarchy-generated-alpha HRP research backtest."
    )
    parser.add_argument("--returns", help="Wide weekly returns CSV.")
    parser.add_argument("--pit", default=None, help="Matching PIT universe CSV.")
    parser.add_argument("--output-dir", default="hierarchical_alpha_outputs")
    parser.add_argument("--evaluation-start", type=pd.Timestamp, default=None)
    parser.add_argument("--as-of", type=pd.Timestamp, default=None)
    parser.add_argument("--lookback-weeks", type=int, default=260)
    parser.add_argument("--minimum-observations", type=int, default=104)
    parser.add_argument("--rebalance-weeks", type=int, default=4)
    parser.add_argument("--max-weight", type=float, default=0.10)
    parser.add_argument("--max-active-share", type=float, default=0.05)
    parser.add_argument("--max-tracking-error", type=float, default=0.04)
    parser.add_argument("--max-turnover-l1", type=float, default=0.20)
    parser.add_argument("--tc-bps", type=float, default=10.0)
    parser.add_argument(
        "--missing-held-return", choices=("raise", "zero"), default="raise"
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        _self_test()
        return
    if not args.returns:
        raise ValueError("--returns is required unless --self-test is used.")
    config = ModelConfig(
        lookback_weeks=args.lookback_weeks,
        minimum_observations=args.minimum_observations,
        rebalance_weeks=args.rebalance_weeks,
        max_weight=args.max_weight,
        maximum_active_share=args.max_active_share,
        maximum_tracking_error=args.max_tracking_error,
        maximum_rebalance_turnover_l1=args.max_turnover_l1,
        transaction_cost_bps=args.tc_bps,
    )
    returns = _load_wide_csv(args.returns)
    if args.as_of is not None:
        returns = returns.loc[returns.index <= args.as_of]
    pit = _load_pit(args.pit, returns)
    if args.pit is None:
        print("WARNING: no PIT universe supplied; results are exploratory only.")
    portfolio_returns, turnover, weights, diagnostics, metrics = run_backtest(
        returns,
        pit,
        config,
        args.evaluation_start,
        args.missing_held_return,
    )
    destination = Path(args.output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    portfolio_returns.to_csv(destination / "model_returns.csv")
    turnover.to_csv(destination / "turnover.csv")
    weights.to_csv(destination / "hierarchical_alpha_weights.csv")
    metrics.to_csv(destination / "model_comparison.csv")
    (destination / "allocation_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "run_config.json").write_text(
        json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.to_string())
    print(f"saved: {destination}")


if __name__ == "__main__":
    main()
