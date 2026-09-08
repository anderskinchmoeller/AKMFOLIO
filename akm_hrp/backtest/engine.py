from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from akm_hrp.backtest.cpcv import build_cpcv_splits
from akm_hrp.data.pit_universe import load_pit_universe_mask
from akm_hrp.diagnostics.significance import sharpe_significance

WEEKS_PER_YEAR = 52.0
_EPS = 1e-12


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(float(seconds)))
    hours, remainder = divmod(seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


@dataclass(frozen=True)
class BacktestResult:
    """Container returned by walk-forward and CPCV backtests."""

    weights: pd.DataFrame
    portfolio_returns: pd.Series
    turnover: pd.Series
    metrics: dict[str, float]
    transaction_costs: pd.Series | None = None
    ending_weights: pd.Series | None = None

    # Compatibility aliases for older callers.
    @property
    def port_ret(self) -> pd.Series:
        return self.portfolio_returns

    @property
    def returns(self) -> pd.Series:
        return self.portfolio_returns


def _empty_result(index: pd.Index, assets: pd.Index) -> BacktestResult:
    weights = pd.DataFrame(0.0, index=index, columns=assets, dtype=float)
    port_ret = pd.Series(np.nan, index=index, dtype=float, name="portfolio_return")
    turnover = pd.Series(0.0, index=index, dtype=float, name="turnover")
    transaction_costs = pd.Series(
        0.0, index=index, dtype=float, name="transaction_cost"
    )
    return BacktestResult(
        weights=weights,
        portfolio_returns=port_ret,
        turnover=turnover,
        metrics=_compute_metrics(port_ret, turnover, risk_free_rate=0.0),
        transaction_costs=transaction_costs,
    )


def _compute_metrics(
    port_ret: pd.Series,
    turnover: pd.Series,
    risk_free_rate: float = 0.0,
    number_of_trials: int = 1,
) -> dict[str, float]:
    """Compute the metrics expected by the Rustuna objective."""

    r = pd.Series(port_ret, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    t = pd.Series(turnover, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()

    if r.empty:
        empty_metrics = {
            "sharpe": 0.0,
            "sortino": 0.0,
            "mean_return": 0.0,
            "cagr": 0.0,
            "vol": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "weekly_var_95": 0.0,
            "weekly_cvar_95": 0.0,
            "positive_week_fraction": 0.0,
            "ann_turnover": 0.0,
            "total_return": 0.0,
            "n_obs": 0.0,
        }
        empty_metrics.update(
            sharpe_significance(r, number_of_trials=number_of_trials)
        )
        return empty_metrics

    weekly_rf = float(risk_free_rate) / WEEKS_PER_YEAR
    excess = r - weekly_rf
    vol_weekly = float(r.std(ddof=1)) if len(r) > 1 else 0.0

    if np.isfinite(vol_weekly) and vol_weekly > _EPS:
        sharpe = float(np.sqrt(WEEKS_PER_YEAR) * excess.mean() / vol_weekly)
    else:
        sharpe = 0.0

    downside_deviation = float(
        np.sqrt(np.mean(np.square(np.minimum(excess.to_numpy(dtype=float), 0.0))))
    )
    if np.isfinite(downside_deviation) and downside_deviation > _EPS:
        sortino = float(np.sqrt(WEEKS_PER_YEAR) * excess.mean() / downside_deviation)
    else:
        sortino = 0.0

    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    maximum_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    years = len(r) / WEEKS_PER_YEAR
    terminal_wealth = float(equity.iloc[-1])
    cagr = (
        float(terminal_wealth ** (1.0 / years) - 1.0)
        if years > 0.0 and terminal_wealth > 0.0
        else -1.0
    )
    calmar = float(cagr / abs(maximum_drawdown)) if maximum_drawdown < -_EPS else 0.0
    weekly_var_95 = float(r.quantile(0.05))
    tail = r.loc[r <= weekly_var_95]
    weekly_cvar_95 = float(tail.mean()) if not tail.empty else weekly_var_95

    metrics = {
        "sharpe": sharpe,
        "sortino": sortino,
        "mean_return": float(r.mean() * WEEKS_PER_YEAR),
        "cagr": cagr,
        "vol": float(vol_weekly * np.sqrt(WEEKS_PER_YEAR)),
        "max_drawdown": maximum_drawdown,
        "calmar": calmar,
        "weekly_var_95": weekly_var_95,
        "weekly_cvar_95": weekly_cvar_95,
        "positive_week_fraction": float((r > 0.0).mean()),
        "ann_turnover": float(t.mean() * WEEKS_PER_YEAR) if not t.empty else 0.0,
        "total_return": float(equity.iloc[-1] - 1.0),
        "n_obs": float(len(r)),
    }
    metrics.update(
        sharpe_significance(excess, number_of_trials=number_of_trials)
    )
    return metrics


def _load_pit_mask(
    returns: pd.DataFrame,
    cfg: Any,
) -> pd.DataFrame | None:
    """Load the PIT universe if one is configured."""

    path = getattr(cfg, "pit_universe_file", None)
    if not path:
        return None

    try:
        return load_pit_universe_mask(
            path,
            dates=returns.index,
            assets=list(returns.columns),
        )
    except Exception:
        if bool(getattr(cfg, "strict_pit_universe", True)):
            raise
        return None


def _finite_correlation_subset(
    window: pd.DataFrame,
    min_obs: int,
) -> pd.DataFrame:
    """
    Remove assets responsible for non-finite pairwise correlations.

    This deliberately does NOT fill NaN correlations with zero. Assets without
    sufficient usable overlap are excluded from that rebalance instead.
    """

    x = window.copy()

    while x.shape[1] >= 2:
        corr = x.corr(min_periods=min_obs)
        values = corr.to_numpy(dtype=float)
        finite = np.isfinite(values)

        if finite.all():
            return x

        # Drop the column participating in the largest number of bad pairs.
        bad_counts = (~finite).sum(axis=0)
        drop_pos = int(np.argmax(bad_counts))
        x = x.drop(columns=[x.columns[drop_pos]])

    return x


def _eligible_window(
    window: pd.DataFrame,
    cfg: Any,
    pit_mask: pd.DataFrame | None,
    *,
    require_pairwise_finite_correlation: bool = True,
) -> pd.DataFrame:
    """Construct the point-in-time eligible matrix for one rebalance."""

    if window.empty:
        return window

    x = window.copy().replace([np.inf, -np.inf], np.nan)
    rebalance_date = x.index[-1]

    min_obs = int(getattr(cfg, "min_window_obs", 52))
    max_missing = float(getattr(cfg, "cov_max_interior_missing_fraction", 1.0))

    obs = x.notna().sum(axis=0)
    std = x.std(axis=0, skipna=True)
    missing_fraction = x.isna().mean(axis=0)

    eligible = (
        (obs >= min_obs)
        & std.notna()
        & (std > _EPS)
        & (missing_fraction <= max_missing)
    )

    if pit_mask is not None:
        if rebalance_date not in pit_mask.index:
            if bool(getattr(cfg, "strict_pit_universe", True)):
                raise RuntimeError(
                    f"PIT universe has no row for rebalance date {rebalance_date}."
                )
            active = pd.Series(True, index=x.columns)
        else:
            active = (
                pit_mask.loc[rebalance_date].reindex(x.columns).fillna(0).astype(bool)
            )
        eligible &= active

    x = x.loc[:, eligible]

    if x.shape[1] < 2:
        return x

    if not require_pairwise_finite_correlation:
        return x

    return _finite_correlation_subset(x, min_obs=min_obs)


def _sanitize_target_weights(
    target: pd.Series,
    all_assets: pd.Index,
) -> pd.Series:
    """Validate allocator output and align it to the full asset universe."""

    if not isinstance(target, pd.Series):
        target = pd.Series(target)

    target = target.astype(float).replace([np.inf, -np.inf], np.nan)

    if target.isna().any():
        bad = target.index[target.isna()].tolist()
        raise RuntimeError(f"Allocator returned non-finite weights for: {bad}")

    if (target < -_EPS).any():
        bad = target[target < -_EPS]
        raise RuntimeError(f"Allocator returned negative weights: {bad.to_dict()}")

    target = target.clip(lower=0.0)
    total = float(target.sum())

    if not np.isfinite(total) or total <= _EPS:
        raise RuntimeError("Allocator returned zero/invalid total portfolio weight.")

    target = target / total
    return target.reindex(all_assets).fillna(0.0)


def _cap_l1_turnover(
    current_w: pd.Series,
    target_w: pd.Series,
    max_l1: float | None,
) -> pd.Series:
    """Optionally cap a rebalance by interpolating toward its target."""

    if max_l1 is None:
        return target_w

    max_l1 = float(max_l1)
    if max_l1 <= 0:
        return current_w.copy()

    delta = target_w - current_w
    l1 = float(delta.abs().sum())

    if l1 <= max_l1 + _EPS:
        return target_w

    scaled = current_w + delta * (max_l1 / l1)
    scaled = scaled.clip(lower=0.0)

    if scaled.sum() > _EPS:
        scaled /= scaled.sum()

    return scaled


def _portfolio_bounds_feasible(
    allocator: Any,
    cfg: Any,
    n_assets: int,
) -> bool:
    """
    Return True when a fully-invested long-only portfolio can satisfy the
    configured global min/max weight bounds for the current universe.

    For N assets, feasibility requires:

        N * min_weight <= 1 <= N * max_weight

    AllocatorConfig is authoritative when present because Rustuna may tune
    min_weight trial-by-trial. HRPConfig is used as a fallback.
    """
    if n_assets <= 0:
        return False

    alloc_cfg = getattr(allocator, "config", None)

    min_weight = float(
        getattr(
            alloc_cfg,
            "min_weight",
            getattr(cfg, "min_weight", 0.0),
        )
    )

    max_weight = float(
        getattr(
            alloc_cfg,
            "max_weight",
            getattr(cfg, "max_weight", 1.0),
        )
    )

    lower_sum = n_assets * min_weight
    upper_sum = n_assets * max_weight

    return lower_sum <= 1.0 + _EPS and upper_sum >= 1.0 - _EPS


def _prepare_allocator_window(
    allocator: Any,
    full_window: pd.DataFrame,
    eligible_window: pd.DataFrame,
    current_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Allow dynamic allocators to restore causal core and exit-only assets."""

    preparer = getattr(allocator, "prepare_allocation_window", None)
    if not callable(preparer):
        return eligible_window
    prepared = preparer(full_window, eligible_window, current_weights)
    if not isinstance(prepared, pd.DataFrame):
        raise TypeError("prepare_allocation_window must return a DataFrame.")
    if not prepared.index.equals(full_window.index):
        raise ValueError("Prepared allocation window changed the return dates.")
    unknown = prepared.columns.difference(full_window.columns)
    if len(unknown):
        raise ValueError(
            "Prepared allocation window introduced unknown assets: "
            f"{unknown[:5].tolist()}"
        )
    return prepared


def latest_target_weights(
    returns: pd.DataFrame,
    allocator: Any,
    cfg: Any,
    *,
    current_weights: pd.Series | None = None,
) -> pd.Series:
    """Calculate the model target using information through the final row.

    This is a post-close target, not the beginning-of-period weights stored by
    ``run_walk_forward``. Portfolio drift, no-trade thresholds, and execution
    constraints should be applied separately when converting it into orders.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")

    returns = returns.copy().astype(float).sort_index()
    if returns.empty:
        raise ValueError("returns is empty")
    if returns.index.has_duplicates:
        raise ValueError("returns index contains duplicate dates")

    lookback = int(getattr(cfg, "lookback_weeks", 260))
    allocator_config = getattr(allocator, "config", None)
    allocator_minimum_history = int(
        getattr(allocator_config, "minimum_history_weeks", 0)
    )
    min_obs = max(
        int(getattr(cfg, "min_window_obs", 52)),
        allocator_minimum_history,
    )
    window = returns.iloc[-lookback:].copy()
    if len(window) < min_obs:
        raise RuntimeError(
            f"Latest target requires {min_obs} observations; received {len(window)}."
        )

    pit_mask = _load_pit_mask(returns, cfg)
    require_pairwise = bool(
        getattr(
            getattr(allocator, "config", None),
            "requires_pairwise_finite_correlation",
            True,
        )
    )
    eligible_window = _eligible_window(
        window,
        cfg,
        pit_mask,
        require_pairwise_finite_correlation=require_pairwise,
    )
    eligible_window = _prepare_allocator_window(
        allocator, window, eligible_window, current_weights
    )
    n_eligible = int(eligible_window.shape[1])
    if n_eligible < 2:
        raise RuntimeError(f"Latest target has too few eligible assets: {n_eligible}.")
    if not _portfolio_bounds_feasible(allocator, cfg, n_eligible):
        raise RuntimeError(
            f"Latest target bounds are infeasible for {n_eligible} eligible assets."
        )

    holdings_setter = getattr(allocator, "set_current_weights", None)
    if callable(holdings_setter) and current_weights is not None:
        holdings_setter(pd.Series(current_weights, dtype=float).copy())
    raw_target = allocator.allocate(eligible_window)
    return _sanitize_target_weights(raw_target, pd.Index(returns.columns))


def run_walk_forward(
    returns: pd.DataFrame,
    allocator: Any,
    cfg: Any,
    *,
    progress_every_rebalances: int = 0,
    progress_label: str | None = None,
) -> BacktestResult:
    """
    Run a chronological walk-forward backtest.

    At week i, the allocator sees only returns strictly before week i. Assets
    are filtered for minimum history, non-zero variance, finite pairwise
    correlations, and (when configured) the PIT universe as of the previous
    observation/rebalance date.
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")

    returns = returns.copy().astype(float).sort_index()

    if returns.index.has_duplicates:
        raise ValueError("returns index contains duplicate dates")

    assets = pd.Index(returns.columns)
    if len(assets) == 0:
        return _empty_result(returns.index, assets)

    weights = pd.DataFrame(0.0, index=returns.index, columns=assets, dtype=float)
    port_ret = pd.Series(
        np.nan, index=returns.index, dtype=float, name="portfolio_return"
    )
    turnover = pd.Series(0.0, index=returns.index, dtype=float, name="turnover")
    transaction_costs = pd.Series(
        0.0, index=returns.index, dtype=float, name="transaction_cost"
    )

    pit_mask = _load_pit_mask(returns, cfg)

    lookback = int(getattr(cfg, "lookback_weeks", 260))
    allocator_config = getattr(allocator, "config", None)
    allocator_minimum_history = int(
        getattr(allocator_config, "minimum_history_weeks", 0)
    )
    min_obs = max(
        int(getattr(cfg, "min_window_obs", 52)),
        allocator_minimum_history,
    )
    risk_free_rate = float(getattr(cfg, "risk_free_rate", 0.0))

    # Optional legacy controls. They default to inactive for the new config.
    drift_threshold = float(getattr(cfg, "drift_threshold", 0.0))
    min_holding_weeks = int(getattr(cfg, "min_holding_weeks", 0))
    max_rebalance_turnover_l1 = getattr(cfg, "max_rebalance_turnover_l1", None)
    tc_bps = float(getattr(cfg, "tc_bps", 0.0))

    current_w = pd.Series(0.0, index=assets, dtype=float)
    last_rebalance_i: int | None = None
    rebalance_count = 0
    run_started = time.perf_counter()
    total_backtest_steps = max(len(returns) - min_obs, 1)
    label = progress_label or allocator.__class__.__name__
    if progress_every_rebalances > 0:
        print(
            f"{label} started: {total_backtest_steps:,} weekly steps; "
            "ETA will update after completed allocations",
            flush=True,
        )

    # i is the return period being earned. The estimation window ends at i-1.
    for i in range(1, len(returns)):
        start = max(0, i - lookback)
        window = returns.iloc[start:i].copy()

        # Not enough history yet. If already invested, carry the portfolio.
        if len(window) < min_obs:
            if current_w.sum() > _EPS:
                realised = (
                    returns.iloc[i].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                )
                port_ret.iloc[i] = float(current_w.dot(realised))
                weights.iloc[i] = current_w
            continue

        require_pairwise = bool(
            getattr(
                getattr(allocator, "config", None),
                "requires_pairwise_finite_correlation",
                True,
            )
        )
        eligible_window = _eligible_window(
            window,
            cfg,
            pit_mask,
            require_pairwise_finite_correlation=require_pairwise,
        )
        eligible_window = _prepare_allocator_window(
            allocator, window, eligible_window, current_w
        )

        n_eligible = int(eligible_window.shape[1])

        should_rebalance = n_eligible >= 2 and _portfolio_bounds_feasible(
            allocator,
            cfg,
            n_eligible,
        )

        if (
            should_rebalance
            and last_rebalance_i is not None
            and min_holding_weeks > 0
            and i - last_rebalance_i < min_holding_weeks
        ):
            should_rebalance = False

        if should_rebalance:
            holdings_setter = getattr(allocator, "set_current_weights", None)
            if callable(holdings_setter):
                # Cost-aware allocators need drifted live holdings rather than
                # merely their previous target when pricing the next trade.
                holdings_setter(current_w.copy())
            raw_target = allocator.allocate(eligible_window)
            rebalance_count += 1
            if (
                progress_every_rebalances > 0
                and rebalance_count % progress_every_rebalances == 0
            ):
                elapsed_seconds = time.perf_counter() - run_started
                completed_steps = max(i - min_obs + 1, 1)
                progress_fraction = float(
                    np.clip(completed_steps / total_backtest_steps, _EPS, 1.0)
                )
                eta_seconds = (
                    elapsed_seconds * (1.0 - progress_fraction) / progress_fraction
                )
                estimated_finish = datetime.now().astimezone() + timedelta(
                    seconds=eta_seconds
                )
                print(
                    f"{label} rebalance {rebalance_count:,}: "
                    f"formation={window.index[-1].date()}, "
                    f"assets={n_eligible:,}, "
                    f"progress={100.0 * progress_fraction:.1f}%, "
                    f"elapsed={_format_duration(elapsed_seconds)}, "
                    f"eta~{_format_duration(eta_seconds)}, "
                    f"finish~{estimated_finish.strftime('%Y-%m-%d %H:%M %Z')}",
                    flush=True,
                )
            target_w = _sanitize_target_weights(raw_target, assets)

            proposed_l1 = float((target_w - current_w).abs().sum())

            # A drift threshold of zero means "rebalance whenever a target exists".
            if (
                current_w.sum() > _EPS
                and drift_threshold > 0
                and proposed_l1 < drift_threshold
            ):
                target_w = current_w.copy()
                proposed_l1 = 0.0

            target_w = _cap_l1_turnover(
                current_w,
                target_w,
                max_rebalance_turnover_l1,
            )

            actual_l1 = float((target_w - current_w).abs().sum())

            # Initial funding is not counted as strategy turnover.
            if current_w.sum() <= _EPS:
                actual_l1 = 0.0

            cost_estimator = getattr(allocator, "estimate_execution_cost", None)
            if callable(cost_estimator) and current_w.sum() > _EPS:
                estimated_cost = float(cost_estimator(current_w, target_w))
                if not np.isfinite(estimated_cost) or estimated_cost < 0.0:
                    raise ValueError(
                        "Allocator estimate_execution_cost returned an invalid cost."
                    )
                transaction_costs.iloc[i] = estimated_cost
            else:
                transaction_costs.iloc[i] = actual_l1 * tc_bps / 10_000.0

            if float((target_w - current_w).abs().sum()) > _EPS:
                last_rebalance_i = i

            current_w = target_w
            turnover.iloc[i] = actual_l1

        # If no eligible target exists, preserve the existing portfolio. Before
        # first investment this leaves the return as NaN, so warm-up weeks do
        # not dilute Sharpe with artificial zero returns.
        if current_w.sum() > _EPS:
            realised = returns.iloc[i].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            period_weights = current_w.copy()
            gross_ret = float(period_weights.dot(realised))
            trading_cost = float(transaction_costs.iloc[i])
            port_ret.iloc[i] = gross_ret - trading_cost
            weights.iloc[i] = period_weights

            # Holdings drift with realised returns between rebalances.  Using
            # static target weights here understates both subsequent turnover
            # and concentration after large relative moves.
            gross_value = 1.0 + gross_ret
            if np.isfinite(gross_value) and gross_value > _EPS:
                drifted = period_weights * (1.0 + realised)
                drifted = drifted.clip(lower=0.0)
                if float(drifted.sum()) > _EPS:
                    current_w = drifted / drifted.sum()
                else:
                    current_w = pd.Series(0.0, index=assets, dtype=float)
            else:
                current_w = pd.Series(0.0, index=assets, dtype=float)

    if progress_every_rebalances > 0:
        elapsed_seconds = time.perf_counter() - run_started
        print(
            f"{label} finished: {rebalance_count:,} rebalances, "
            f"progress=100.0%, elapsed={_format_duration(elapsed_seconds)}",
            flush=True,
        )

    metrics = _compute_metrics(
        port_ret,
        turnover,
        risk_free_rate=risk_free_rate,
    )

    return BacktestResult(
        weights=weights,
        portfolio_returns=port_ret,
        turnover=turnover,
        metrics=metrics,
        transaction_costs=transaction_costs,
        ending_weights=current_w.copy(),
    )


def _selector_to_frame(
    frame: pd.DataFrame,
    selector: Any,
) -> pd.DataFrame:
    """
    Convert CPCV selector formats into a DataFrame slice.

    Supported forms include:
      - slice
      - boolean Series / ndarray
      - integer positions
      - date/index labels
      - lists/tuples of any of the above
      - lists/tuples of boolean Series (common for multiple CPCV groups)
    """

    if selector is None:
        return frame.iloc[0:0].copy()

    if isinstance(selector, slice):
        return frame.iloc[selector]

    # A collection of selectors may represent multiple CPCV groups.
    # Resolve each selector independently, then take the union of rows.
    if isinstance(selector, (list, tuple)):
        if len(selector) == 0:
            return frame.iloc[0:0].copy()

        # Plain scalar lists should still be treated as one selector.
        scalar_like = all(
            not isinstance(
                item,
                (
                    pd.Series,
                    pd.Index,
                    np.ndarray,
                    list,
                    tuple,
                    slice,
                ),
            )
            for item in selector
        )

        if scalar_like:
            values = np.asarray(selector)
        else:
            parts = [_selector_to_frame(frame, item) for item in selector]

            if not parts:
                return frame.iloc[0:0].copy()

            selected_index = frame.index[
                frame.index.isin(
                    pd.Index(
                        np.concatenate(
                            [part.index.to_numpy() for part in parts if not part.empty]
                        )
                    )
                    if any(not part.empty for part in parts)
                    else pd.Index([])
                )
            ]

            return frame.loc[selected_index]

    elif isinstance(selector, pd.Series):
        # Boolean Series can be index-aligned rather than position-aligned.
        if pd.api.types.is_bool_dtype(selector.dtype):
            mask = selector.reindex(frame.index, fill_value=False).to_numpy(dtype=bool)
            return frame.loc[mask]
        values = selector.to_numpy()

    elif isinstance(selector, pd.Index):
        values = selector.to_numpy()

    else:
        values = np.asarray(selector)

    if values.ndim == 0:
        values = np.asarray([values.item()])

    # Object arrays can themselves contain Series/masks.
    if values.dtype == object and any(
        isinstance(
            item,
            (
                pd.Series,
                pd.Index,
                np.ndarray,
                list,
                tuple,
                slice,
            ),
        )
        for item in values.tolist()
    ):
        return _selector_to_frame(frame, values.tolist())

    if pd.api.types.is_bool_dtype(values.dtype):
        if len(values) != len(frame):
            raise ValueError(
                f"Boolean CPCV selector has wrong length: {len(values)} != {len(frame)}"
            )
        return frame.iloc[values.astype(bool)]

    if np.issubdtype(values.dtype, np.integer):
        return frame.iloc[values.astype(int)]

    # Otherwise interpret as index labels/dates.
    labels = pd.Index(values)
    mask = frame.index.isin(labels)
    return frame.loc[mask]


def _split_components(split: Any) -> tuple[Any | None, Any | None]:
    """
    Extract train/test selectors from common CPCV split representations.
    """

    train_names = (
        "train_idx",
        "train_indices",
        "train_index",
        "train_dates",
        "train_mask",
        "train_masks",
        "train_groups",
        "train",
    )

    test_names = (
        "test_idx",
        "test_indices",
        "test_index",
        "test_dates",
        "test_mask",
        "test_masks",
        "test_groups",
        "test",
    )

    if isinstance(split, dict):
        train = next(
            (split[k] for k in train_names if k in split),
            None,
        )
        test = next(
            (split[k] for k in test_names if k in split),
            None,
        )
        return train, test

    train = next(
        (getattr(split, k) for k in train_names if hasattr(split, k)),
        None,
    )

    test = next(
        (getattr(split, k) for k in test_names if hasattr(split, k)),
        None,
    )

    if train is not None or test is not None:
        return train, test

    if isinstance(split, (tuple, list)) and len(split) >= 2:
        return split[0], split[1]

    return None, split


def run_cpcv_backtest(
    returns: pd.DataFrame,
    allocator_factory: Callable[[pd.DataFrame], Any],
    cfg: Any,
) -> dict[Any, BacktestResult]:
    """
    Run the configured CPCV splits.

    `cfg.cpcv` is intentionally required here because the current CPCV builder
    expects the nested CPCVConfig object. The tuning compatibility proxy can
    provide it while forwarding the remaining HRPConfig attributes.
    """

    if not hasattr(cfg, "cpcv"):
        raise AttributeError(
            "Backtest config must expose `.cpcv` (a CPCVConfig instance)."
        )

    returns = returns.copy().astype(float).sort_index()
    splits = build_cpcv_splits(returns.index, cfg.cpcv)

    if isinstance(splits, dict):
        split_items = list(splits.items())
    else:
        split_items = list(enumerate(splits))

    results: dict[Any, BacktestResult] = {}
    refit_per_split = bool(getattr(cfg, "cpcv_refit_per_split", False))
    shared_result: BacktestResult | None = None

    if not refit_per_split:
        # HRP has no globally fitted parameters: every estimate is formed from
        # the chronological prefix at the rebalance date. Reuse that one
        # expensive path and score it under every CPCV test selector.
        shared_allocator = allocator_factory(returns.iloc[0:0])
        shared_result = run_walk_forward(returns, shared_allocator, cfg)

    for split_name, split in split_items:
        train_selector, test_selector = _split_components(split)

        test_ret = _selector_to_frame(returns, test_selector)
        if test_ret.empty:
            continue

        if train_selector is None:
            train_ret = returns.drop(index=test_ret.index, errors="ignore")
        else:
            train_ret = _selector_to_frame(returns, train_selector)

        if shared_result is None:
            allocator = allocator_factory(train_ret)
            full_result = run_walk_forward(returns, allocator, cfg)
        else:
            full_result = shared_result
        test_index = test_ret.index
        test_returns = full_result.portfolio_returns.reindex(test_index)
        test_turnover = full_result.turnover.reindex(test_index).fillna(0.0)
        test_transaction_costs = (
            full_result.transaction_costs.reindex(test_index).fillna(0.0)
            if full_result.transaction_costs is not None
            else None
        )
        results[split_name] = BacktestResult(
            weights=full_result.weights.reindex(test_index),
            portfolio_returns=test_returns,
            turnover=test_turnover,
            metrics=_compute_metrics(
                test_returns,
                test_turnover,
                risk_free_rate=float(getattr(cfg, "risk_free_rate", 0.0)),
            ),
            transaction_costs=test_transaction_costs,
        )

    return results
