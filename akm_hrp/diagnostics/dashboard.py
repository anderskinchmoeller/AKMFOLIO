from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52
_EPS = 1e-12


_MODEL_LABELS = {
    "equal_weight": "Equal Weight",
    "inverse_volatility": "Inverse Volatility",
    "regularized_minimum_variance": "Regularized Minimum Variance",
    "ra_hrp": "RA-HRP",
    "ra_hrp_v2": "RA-HRP v2",
    "hrp_alpha_v1": "HRP Alpha v1",
    "hrp_alpha_v2": "HRP Alpha v2",
    "mapper_factor_nco": "Mapper Factor NCO",
    "barra_factor_hrp": "Barra-Style Factor HRP",
    "low_overfit_hrp": "Low-Overfit HRP + Momentum",
    "dynamic_barra_alpha": "Dynamic Barra-Style Alpha",
    "retail_alpha_mpc": "Retail Alpha MPC",
    "retail_alpha_ml_mpc": "Retail Alpha ML MPC",
    "retail_alpha_ml_mpc_equal_weight": "Retail Alpha ML MPC (Equal-Weight Sizing)",
    "retail_alpha_ml_mpc_crowding": "Retail Alpha ML MPC + Crowding Kappa",
    "retail_edge_mpc": "Retail Edge MPC",
    "retail_edge_ml_mpc": "Retail Edge ML MPC",
    "legacy_ensemble_hrp": "Legacy Ensemble HRP",
    "regret_aware_core": "Regret-Aware Core",
    "regret_aware_with_overlay": "Regret-Aware Overlay",
}


def model_display_name(name: str) -> str:
    """Return a readable plot label while preserving unknown model names."""

    return _MODEL_LABELS.get(name, name.replace("_", " ").title())


def _result_returns(result: Any) -> pd.Series:
    """Read the return series from current and legacy backtest result objects."""

    for attribute in ("portfolio_returns", "port_ret", "returns"):
        value = getattr(result, attribute, None)
        if value is not None:
            return pd.Series(value, dtype=float)
    raise AttributeError("Backtest result has no portfolio return series.")


def _result_turnover(result: Any) -> pd.Series:
    """Use realized engine turnover, or reconstruct it for legacy results."""

    value = getattr(result, "turnover", None)
    if value is not None:
        return pd.Series(value, dtype=float)

    weights = pd.DataFrame(result.weights, dtype=float).fillna(0.0)
    return weights.diff().abs().sum(axis=1).fillna(0.0)


def _result_transaction_costs(result: Any, tc_bps: float) -> pd.Series:
    """Prefer allocator-aware execution costs, with the legacy fallback."""

    value = getattr(result, "transaction_costs", None)
    if value is not None:
        return pd.Series(value, dtype=float)
    return _result_turnover(result) * float(tc_bps) / 10_000.0


def _after(series: pd.Series | pd.DataFrame, start: pd.Timestamp | None):
    if start is None:
        return series
    return series.loc[series.index >= start]


def _equity_curve(returns: pd.Series) -> pd.Series:
    clean = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (1.0 + clean).cumprod()


def _drawdown_pct(returns: pd.Series) -> pd.Series:
    equity = _equity_curve(returns)
    return 100.0 * (equity / equity.cummax() - 1.0)


def _rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
    clean = returns.replace([np.inf, -np.inf], np.nan)
    mean = clean.rolling(window, min_periods=window).mean()
    volatility = clean.rolling(window, min_periods=window).std(ddof=1)
    sharpe = np.sqrt(WEEKS_PER_YEAR) * mean / volatility
    return sharpe.replace([np.inf, -np.inf], np.nan)


def _ticker_labels(
    assets: pd.Index,
    ticker_map: Mapping[str, str] | None,
) -> list[str]:
    """Make ticker labels unique without losing the source identifier."""

    identifiers = [str(asset) for asset in assets]
    if not ticker_map:
        return identifiers

    raw_labels = [str(ticker_map.get(asset, asset)) for asset in identifiers]
    counts = pd.Series(raw_labels).value_counts()
    return [
        f"{label} ({asset})" if counts[label] > 1 else label
        for asset, label in zip(identifiers, raw_labels, strict=True)
    ]


def export_backtest_dashboard(
    results: Mapping[str, Any],
    *,
    focus_model: str,
    output_pdf: str | Path | None = None,
    output_png: str | Path | None = None,
    title: str | None = None,
    tc_bps: float = 10.0,
    lookback_weeks: int = 260,
    rolling_sharpe_years: int = 3,
    heatmap_assets: int = 15,
    evaluation_start: pd.Timestamp | str | None = None,
    ticker_map: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Export a model-agnostic research dashboard from walk-forward results.

    Every panel is computed from the supplied ``BacktestResult`` objects. This
    keeps the PDF consistent with the metrics produced by ``compare_models``
    and avoids re-running an allocator inside the reporting layer.
    """

    if not results:
        raise ValueError("At least one backtest result is required.")
    if focus_model not in results:
        raise ValueError(
            f"Dashboard focus model {focus_model!r} is not in {sorted(results)}."
        )
    if output_pdf is None and output_png is None:
        raise ValueError("Set output_pdf, output_png, or both.")
    if rolling_sharpe_years < 1:
        raise ValueError("rolling_sharpe_years must be at least 1.")
    if heatmap_assets < 1:
        raise ValueError("heatmap_assets must be at least 1.")
    if tc_bps < 0:
        raise ValueError("tc_bps cannot be negative.")

    start = pd.Timestamp(evaluation_start) if evaluation_start is not None else None
    returns_by_model = {
        name: _after(_result_returns(result).sort_index(), start)
        for name, result in results.items()
    }
    if any(series.empty for series in returns_by_model.values()):
        empty = [name for name, series in returns_by_model.items() if series.empty]
        raise ValueError(f"No dashboard observations remain for models: {empty}")

    focus_result = results[focus_model]
    focus_weights = _after(
        pd.DataFrame(focus_result.weights, dtype=float).sort_index(),
        start,
    )
    if focus_weights.empty or focus_weights.shape[1] == 0:
        raise ValueError(f"{focus_model!r} has no weight history to plot.")

    average_weight = focus_weights.abs().mean(axis=0).sort_values(ascending=False)
    top_assets = average_weight.loc[average_weight > _EPS].head(heatmap_assets).index
    if top_assets.empty:
        raise ValueError(f"{focus_model!r} has no non-zero weights to plot.")
    heatmap = focus_weights.loc[:, top_assets].T.mask(
        focus_weights.loc[:, top_assets].T.abs() <= _EPS
    )

    # Import plotting lazily so command-line and unit-test imports remain
    # headless-safe on servers without a graphical display.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(results)
    display_names = {name: model_display_name(name) for name in model_names}
    focus_label = display_names[focus_model]
    rolling_window = int(rolling_sharpe_years * WEEKS_PER_YEAR)

    with plt.rc_context(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
        }
    ):
        figure = plt.figure(figsize=(16, 13), constrained_layout=False)
        grid = figure.add_gridspec(
            3,
            2,
            height_ratios=[1.30, 1.0, 1.0],
            hspace=0.20,
            wspace=0.14,
        )
        equity_axis = figure.add_subplot(grid[0, :])
        drawdown_axis = figure.add_subplot(grid[1, 0])
        cost_axis = figure.add_subplot(grid[1, 1])
        sharpe_axis = figure.add_subplot(grid[2, 0])
        heatmap_axis = figure.add_subplot(grid[2, 1])

        for name in model_names:
            returns = returns_by_model[name]
            emphasized = name == focus_model
            style = "-" if emphasized else "--"
            width = 1.8 if emphasized else 1.25
            equity_axis.plot(
                returns.index,
                _equity_curve(returns),
                label=display_names[name],
                linewidth=width,
                linestyle=style,
            )
        equity_axis.set_title("Equity Curves")
        equity_axis.set_ylabel("Cumulative value ($1 start)")
        equity_axis.grid(True, alpha=0.25)
        equity_axis.legend(loc="upper left", ncols=min(3, len(model_names)))

        for name in model_names:
            drawdown = _drawdown_pct(returns_by_model[name])
            drawdown_axis.plot(
                drawdown.index,
                drawdown,
                label=display_names[name],
                linewidth=1.2 if name == focus_model else 0.9,
                alpha=0.95 if name == focus_model else 0.75,
            )
        drawdown_axis.fill_between(
            _drawdown_pct(returns_by_model[focus_model]).index,
            _drawdown_pct(returns_by_model[focus_model]).to_numpy(dtype=float),
            0.0,
            alpha=0.14,
        )
        drawdown_axis.set_title("Drawdown (%)")
        drawdown_axis.set_ylabel("Drawdown (%)")
        drawdown_axis.grid(True, alpha=0.25)
        drawdown_axis.legend(loc="lower left")

        transaction_cost = _after(
            _result_transaction_costs(focus_result, tc_bps).sort_index(), start
        )
        transaction_cost_pct = 100.0 * transaction_cost.fillna(0.0)
        cost_axis.vlines(
            transaction_cost_pct.index,
            0.0,
            transaction_cost_pct.to_numpy(dtype=float),
            linewidth=0.7,
        )
        cost_axis.set_title(f"Transaction Cost per Rebalance ({focus_label})")
        cost_axis.set_ylabel("Cost (% of capital)")
        cost_axis.grid(True, alpha=0.25)

        for name in model_names:
            rolling = _rolling_sharpe(returns_by_model[name], rolling_window)
            sharpe_axis.plot(
                rolling.index,
                rolling,
                label=display_names[name],
                linewidth=1.5 if name == focus_model else 1.1,
                linestyle="-" if name == focus_model else "--",
            )
        sharpe_axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
        sharpe_axis.set_title(f"Rolling {rolling_sharpe_years}-Year Sharpe Ratio")
        sharpe_axis.set_ylabel("Sharpe")
        sharpe_axis.grid(True, alpha=0.25)
        sharpe_axis.legend(loc="upper left")

        color_map = plt.get_cmap("viridis").with_extremes(bad="white")
        image = heatmap_axis.imshow(
            np.ma.masked_invalid(heatmap.to_numpy(dtype=float)),
            aspect="auto",
            interpolation="nearest",
            cmap=color_map,
            vmin=0.0,
        )
        heatmap_axis.set_title(f"{focus_label} Weights - Top Assets by Average")
        heatmap_axis.set_yticks(np.arange(len(top_assets)))
        heatmap_axis.set_yticklabels(_ticker_labels(top_assets, ticker_map))

        date_count = heatmap.shape[1]
        if date_count:
            tick_positions = np.linspace(
                0,
                date_count - 1,
                min(7, date_count),
            ).astype(int)
            heatmap_axis.set_xticks(tick_positions)
            heatmap_axis.set_xticklabels(
                [
                    pd.Timestamp(heatmap.columns[position]).strftime("%Y-%m")
                    for position in tick_positions
                ],
                rotation=45,
                ha="right",
            )
        color_bar = figure.colorbar(image, ax=heatmap_axis, fraction=0.046, pad=0.04)
        color_bar.set_label("Weight")

        dashboard_title = title or f"{focus_label} Research Dashboard"
        benchmark_labels = [
            display_names[name] for name in model_names if name != focus_model
        ]
        benchmark_text = ", ".join(benchmark_labels) if benchmark_labels else "None"
        cost_text = (
            "spread + temporary/permanent impact"
            if focus_model
            in {
                "retail_alpha_mpc",
                "retail_alpha_ml_mpc",
                "retail_alpha_ml_mpc_equal_weight",
                "retail_alpha_ml_mpc_crowding",
                "retail_edge_mpc",
                "retail_edge_ml_mpc",
            }
            else f"{tc_bps:g}bps transaction cost"
        )
        subtitle = (
            f"{lookback_weeks / WEEKS_PER_YEAR:g}yr rolling lookback, "
            f"{cost_text} | Comparisons: {benchmark_text}"
        )
        figure.suptitle(dashboard_title, fontsize=16, fontweight="bold", y=0.985)
        figure.text(0.5, 0.956, subtitle, ha="center", va="top", fontsize=10)
        figure.subplots_adjust(top=0.92, bottom=0.06, left=0.07, right=0.96)

        artifacts: dict[str, Path] = {}
        metadata = {"Title": dashboard_title, "Creator": "AKM-HRP"}
        if output_pdf is not None:
            pdf_path = Path(output_pdf)
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(pdf_path, bbox_inches="tight", metadata=metadata)
            artifacts["pdf"] = pdf_path
        if output_png is not None:
            png_path = Path(output_png)
            png_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(png_path, dpi=180, bbox_inches="tight", metadata=metadata)
            artifacts["png"] = png_path
        plt.close(figure)

    return artifacts
