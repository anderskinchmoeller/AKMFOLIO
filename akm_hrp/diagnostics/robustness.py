"""Offline uncertainty and stress reports; never called by an allocator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from akm_hrp.diagnostics.significance import newey_west_mean_test


def path_metrics(values: np.ndarray) -> dict[str, float]:
    r = np.asarray(values, dtype=float)
    if r.ndim != 1 or len(r) < 2 or not np.isfinite(r).all() or (r < -1).any():
        raise ValueError("Expected at least two finite decimal returns >= -1.")
    wealth = np.r_[1.0, np.cumprod(1.0 + r)]
    sd = r.std(ddof=1)
    return {
        "sharpe": float(np.sqrt(52) * r.mean() / sd) if sd > 0 else np.nan,
        "cagr": float(wealth[-1] ** (52 / len(r)) - 1),
        "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
    }


def block_bootstrap(
    paired_returns: pd.DataFrame, *, samples: int = 1000, block_weeks: int = 8,
    seed: int = 0,
) -> pd.DataFrame:
    """Moving blocks, joint sampling preserves strategy/benchmark dependence.

    Percentile intervals describe historical sampling uncertainty, not the
    probability of future outperformance. Input rows must be consecutive;
    missing interior observations are rejected rather than stitched together.
    """
    frame = paired_returns.dropna(how="all")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Bootstrap requires aligned finite returns without gaps.")
    if isinstance(frame.index, pd.DatetimeIndex) and len(frame) > 1:
        if (frame.index.to_series().diff().dropna() > pd.Timedelta(days=10)).any():
            raise ValueError("Bootstrap cannot concatenate disconnected test blocks.")
    n = len(frame)
    if samples < 1 or not 1 <= block_weeks <= n or n < 2:
        raise ValueError("Need samples >= 1 and 1 <= block_weeks <= observations.")
    rng = np.random.default_rng(seed)
    values = frame.to_numpy()
    rows = []
    for draw in range(samples):
        starts = rng.integers(0, n - block_weeks + 1, size=int(np.ceil(n / block_weeks)))
        positions = (starts[:, None] + np.arange(block_weeks)).ravel()[:n]
        sample = values[positions]
        for j, name in enumerate(frame.columns):
            rows.append({"draw": draw, "model": name, **path_metrics(sample[:, j])})
        if sample.shape[1] == 2:
            rows.append({"draw": draw, "model": "active",
                         "annualized_mean": float(52 * np.mean(sample[:, 0] - sample[:, 1]))})
    return pd.DataFrame(rows)


def stratified_block_bootstrap(
    paired_returns: pd.DataFrame, regimes: pd.Series, *, samples: int = 1000,
    block_weeks: int = 8, seed: int = 0,
) -> pd.DataFrame:
    """Block-bootstrap within each regime separately, then pool each draw.

    A pooled `block_bootstrap` over the whole sample lets whichever regime
    happened to hold the most weeks dominate every draw's confidence
    interval. This instead draws blocks from each regime's own weeks only,
    so a short but eventful regime (a crash, a crowding unwind) still gets
    its own share of every resampled path instead of being diluted or
    dropped by chance alignment of the block grid.

    Each regime's own weeks are treated as a standalone ordered sequence for
    block purposes (blocks are drawn from within-regime row order, not
    calendar adjacency), since regime membership is rarely calendar-
    contiguous. Rows labeled "warmup" (or unlabeled) are excluded, matching
    `regime_hac`'s treatment of the same placeholder.
    """
    frame = paired_returns.dropna(how="all")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Bootstrap requires aligned finite returns without gaps.")
    labels = pd.Series(regimes).reindex(frame.index)
    keep = labels.notna() & (labels != "warmup")
    frame = frame.loc[keep]
    labels = labels.loc[keep]
    if samples < 1 or block_weeks < 1 or frame.empty:
        raise ValueError("Need samples >= 1, block_weeks >= 1, and labeled rows.")
    rng = np.random.default_rng(seed)
    groups = {
        name: frame.loc[labels == name].to_numpy()
        for name in labels.unique()
        if (labels == name).sum() > 0
    }
    rows = []
    for draw in range(samples):
        pieces = []
        for values in groups.values():
            n = len(values)
            width = min(block_weeks, n)
            starts = rng.integers(0, n - width + 1, size=int(np.ceil(n / width)))
            positions = (starts[:, None] + np.arange(width)).ravel()[:n]
            pieces.append(values[positions])
        sample = np.concatenate(pieces, axis=0)
        for j, name in enumerate(frame.columns):
            rows.append({"draw": draw, "model": name, **path_metrics(sample[:, j])})
        if sample.shape[1] == 2:
            rows.append({"draw": draw, "model": "active",
                         "annualized_mean": float(52 * np.mean(sample[:, 0] - sample[:, 1]))})
    return pd.DataFrame(rows)


def causal_volatility_regimes(benchmark: pd.Series, window: int = 26,
                              reference: int = 104) -> pd.Series:
    """Classify each period using only volatility observed before that period."""
    vol = benchmark.rolling(window, min_periods=window).std().shift(1)
    cutoff = vol.rolling(reference, min_periods=window).median().shift(1)
    labels = pd.Series("warmup", index=benchmark.index, name="regime")
    known = vol.notna() & cutoff.notna()
    labels.loc[known & vol.le(cutoff)] = "low_vol"
    labels.loc[known & vol.gt(cutoff)] = "high_vol"
    return labels


def structural_break_regimes(
    series: pd.Series, *, window: int = 26, warmup: int = 52,
    k: float = 0.5, h: float = 5.0, min_segment: int = 26,
) -> pd.Series:
    """Causal CUSUM changepoint segmentation (Page, 1954) on standardized returns.

    Picking regime boundaries by eyeballing a finished backtest ("here's
    where the bear market was") is itself hindsight bias: it uses
    information that was not available in real time to draw the very
    boundaries the strategy is then judged against. This instead only ever
    uses information available up to and including week t -- the reference
    mean/vol come from a trailing rolling window shifted by one week (the
    same convention as `causal_volatility_regimes`), and the two-sided CUSUM
    statistic is accumulated strictly left to right -- so a detected break
    can never depend on data after it.

    A full Bai-Perron multiple-breakpoint estimator is a heavier batch
    method that (in its usual form) needs the whole sample and an assumed
    break count up front, which makes it awkward to compute causally.
    Sequential CUSUM is the standard lighter-weight alternative for exactly
    this "would I have known it at the time" requirement.

    `k` is the per-week drift allowance (in standardized-return units) below
    which small wobbles are ignored; `h` is the alarm threshold, traded off
    against `min_segment` weeks of cooldown after each break so the detector
    can't immediately re-fire on the tail of the same shift. Returns a
    Series of segment labels ("segment_0", "segment_1", ...), one new label
    starting at each detected break; weeks before `warmup` observations have
    accumulated are labeled "warmup", consistent with
    `causal_volatility_regimes`.
    """
    values = series.astype(float)
    mu = values.rolling(window, min_periods=window).mean().shift(1)
    sigma = values.rolling(window, min_periods=window).std().shift(1)
    z = (values - mu) / sigma.replace(0.0, np.nan)

    labels = pd.Series("warmup", index=values.index, name="regime")
    segment = 0
    pos = neg = 0.0
    last_break = -min_segment
    for i, (idx, zi) in enumerate(z.items()):
        if i < warmup or not np.isfinite(zi):
            continue
        pos = max(0.0, pos + zi - k)
        neg = min(0.0, neg + zi + k)
        labels.loc[idx] = f"segment_{segment}"
        if (pos > h or neg < -h) and (i - last_break) >= min_segment:
            segment += 1
            last_break = i
            pos = neg = 0.0
            labels.loc[idx] = f"segment_{segment}"
    return labels


def regime_hac(active: pd.Series, labels: pd.Series, lags: int = 4,
               exclude_labels: tuple[str, ...] = ("warmup",)) -> pd.DataFrame:
    """HAC dummy regression on the original calendar, never compressed regimes.

    Coefficients are regime conditional mean active returns. HAC scores are
    lagged by actual rows, so two distant high-vol weeks aren't neighbors.
    Works with any categorical regime labeling (volatility-based, CUSUM
    structural-break segments, etc.) rather than only the two-state
    low/high-vol scheme -- one dummy per distinct label found, after
    dropping `exclude_labels` placeholders such as "warmup".
    """
    frame = pd.concat([active.rename("active"), pd.Series(labels).rename("regime")], axis=1).dropna()
    names = sorted(
        name for name in frame["regime"].unique()
        if name not in exclude_labels and (frame["regime"] == name).any()
    )
    if not names or len(frame) < 3:
        return pd.DataFrame()
    x = np.column_stack([(frame["regime"] == name).astype(float) for name in names])
    y = frame["active"].to_numpy()
    bread = np.linalg.pinv(x.T @ x)
    beta = bread @ x.T @ y
    scores = x * (y - x @ beta)[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(lags, len(y) - 1) + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (lags + 1)) * (cross + cross.T)
    covariance = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    from scipy.stats import norm
    rows = []
    for j, name in enumerate(names):
        t = beta[j] / se[j] if se[j] > 0 else np.nan
        rows.append({"regime": name, "observations": int(x[:, j].sum()),
                     "annualized_active_mean": 52 * beta[j], "hac_t_stat": t,
                     "hac_p_value": 2 * norm.sf(abs(t))})
    if len(names) == 2:
        contrast = np.array([-1., 1.])
        variance = float(contrast @ covariance @ contrast)
        difference = float(contrast @ beta)
        t = difference / np.sqrt(variance) if variance > 0 else np.nan
        # Preserve the original label for the two-state vol regime; use a
        # generic "b_minus_a" label for any other pair of regime names.
        contrast_name = (
            "high_minus_low" if set(names) == {"low_vol", "high_vol"}
            else f"{names[1]}_minus_{names[0]}"
        )
        rows.append({"regime": contrast_name, "observations": len(y),
                     "annualized_active_mean": 52 * difference, "hac_t_stat": t,
                     "hac_p_value": 2 * norm.sf(abs(t))})
    return pd.DataFrame(rows)


def write_robustness_report(results: dict, destination: Path, *, benchmark: str,
                            evaluation_start=None, samples: int = 1000,
                            block_weeks: int = 8, seed: int = 0) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if benchmark not in results:
        raise ValueError("Robustness reports require the benchmark in --models.")
    benchmark_returns = results[benchmark].portfolio_returns
    # Build regimes on the full available causal history before scoring.
    regimes = causal_volatility_regimes(benchmark_returns)
    break_regimes = structural_break_regimes(benchmark_returns)
    summary = []
    for name, result in results.items():
        history = pd.DataFrame({"net_return": result.portfolio_returns,
                                "turnover_l1": result.turnover,
                                "transaction_cost": result.transaction_costs})
        history.to_csv(destination / f"{name}_history.csv")
        if result.diagnostics is not None:
            result.diagnostics.to_csv(destination / f"{name}_diagnostics.csv", index=False)
        paired = pd.concat([result.portfolio_returns.rename("strategy"),
                            benchmark_returns.rename("benchmark")], axis=1)
        if evaluation_start is not None:
            paired = paired.loc[paired.index >= pd.Timestamp(evaluation_start)]
        # Trim unequal warmups only; reject any missing rows inside the sample.
        valid = paired.notna().all(axis=1)
        if valid.sum() < max(block_weeks, 2):
            summary.append({"model": name, "status": "insufficient_observations"})
            continue
        paired = paired.loc[valid[valid].index[0]:valid[valid].index[-1]]
        draws = block_bootstrap(paired, samples=samples, block_weeks=block_weeks, seed=seed)
        ci = draws.groupby("model").quantile([0.025, 0.5, 0.975], numeric_only=True).drop(columns="draw")
        ci.to_csv(destination / f"{name}_bootstrap_intervals.csv")
        active = paired.strategy - paired.benchmark
        regime_hac(active, regimes.reindex(active.index)).to_csv(
            destination / f"{name}_regimes.csv", index=False)
        # A pooled block bootstrap lets whichever vol regime holds the most
        # weeks in this sample dominate the interval; this reweights so a
        # short but eventful regime still gets a proportional say.
        stratified_draws = stratified_block_bootstrap(
            paired, regimes.reindex(paired.index),
            samples=samples, block_weeks=block_weeks, seed=seed,
        )
        if not stratified_draws.empty:
            stratified_ci = stratified_draws.groupby("model").quantile(
                [0.025, 0.5, 0.975], numeric_only=True
            ).drop(columns="draw")
            stratified_ci.to_csv(destination / f"{name}_stratified_bootstrap_intervals.csv")
        # Algorithmic (CUSUM) break segmentation as a second, independent
        # regime definition alongside the vol-threshold one above -- neither
        # was drawn by eyeballing the finished backtest.
        regime_hac(active, break_regimes.reindex(active.index)).to_csv(
            destination / f"{name}_regimes_cusum.csv", index=False)
        # Reprice the SAME trades; full execution stress requires a rerun with
        # --execution-cost-multiplier because the optimizer responds to costs.
        costs = result.transaction_costs.reindex(paired.index).fillna(0)
        stresses = []
        for multiplier in (1., 1.5, 2.):
            stressed = paired.strategy - (multiplier - 1) * costs
            stresses.append({"cost_multiplier": multiplier,
                             **path_metrics(stressed.to_numpy()),
                             **newey_west_mean_test(stressed - paired.benchmark)})
        pd.DataFrame(stresses).to_csv(destination / f"{name}_fixed_trade_cost_stress.csv", index=False)
        row = {"model": name, "status": "reported", "observations": len(paired),
               **newey_west_mean_test(active)}
        if result.diagnostics is not None and not result.diagnostics.empty:
            diagnostics = result.diagnostics
            if evaluation_start is not None:
                diagnostics = diagnostics.loc[pd.to_datetime(diagnostics.date) >= pd.Timestamp(evaluation_start)]
            for column in ("optimizer_repaired", "risk_limit_relaxed", "ml_consensus_fraction"):
                if column in diagnostics:
                    row[f"mean_{column}"] = diagnostics[column].mean()
        summary.append(row)
    pd.DataFrame(summary).to_csv(destination / "summary.csv", index=False)
    (destination / "methodology.json").write_text(json.dumps({
        "samples": samples, "block_weeks": block_weeks, "seed": seed,
        "bootstrap": "paired moving block percentile intervals",
        "stratified_bootstrap": "same, but blocks drawn within each vol regime and pooled per draw",
        "regimes": "lagged 26-week volatility versus prior rolling median",
        "regimes_cusum": "causal two-sided CUSUM (Page 1954) changepoint segments, independent of the vol-threshold regime",
        "cost_stress": "fixed-trade repricing; use CLI multiplier for optimizer reruns",
        "placebo": "compare incremental ML results; equity/other signals need not have zero Sharpe. "
                   "For a leakage check instead of a marginal-contribution check, rerun with "
                   "ml_shuffle_labels=True (config on RetailAlphaMLMPCAllocator) and confirm Sharpe collapses to ~0.",
        "limitations": "Historical uncertainty only; no certification of alpha or revision-safe data."
    }, indent=2) + "\n")
