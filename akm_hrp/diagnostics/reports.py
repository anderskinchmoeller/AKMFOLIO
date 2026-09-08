from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from akm_hrp.diagnostics.metrics import (
    compute_drawdowns,
    compute_risk_contributions,
    compute_stability,
    compute_turnover,
)
from akm_hrp.diagnostics.plots import (
    plot_alpha,
    plot_backtest_curve,
    plot_correlation,
    plot_covariance,
    plot_drawdowns,
    plot_latest_weights,
    plot_risk_contributions,
    plot_turnover,
    plot_weights,
)


WEEKS_PER_YEAR = 52.0
_EPS = 1e-12


@dataclass(frozen=True)
class DiagnosticsReport:
    sharpe: float
    annual_return: float
    annual_vol: float
    max_drawdown: float
    ann_turnover: float
    stability: float
    total_return: float
    n_weeks: int
    latest_weights: pd.Series
    risk_contributions: pd.Series
    extra_diagnostics: dict[str, Any]


def _jsonable(value: Any):
    if value is None:
        return None

    if is_dataclass(value):
        return {
            key: _jsonable(val)
            for key, val in asdict(value).items()
        }

    if isinstance(value, dict):
        return {
            str(key): _jsonable(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]

    if isinstance(value, pd.Series):
        return {
            str(key): _jsonable(val)
            for key, val in value.to_dict().items()
        }

    if isinstance(value, pd.DataFrame):
        return {
            str(index): {
                str(key): _jsonable(val)
                for key, val in row.items()
            }
            for index, row in value.to_dict(orient="index").items()
        }

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if np.isfinite(x) else None

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    return value


def generate_diagnostics(
    weights: pd.DataFrame,
    port_ret: pd.Series,
    alpha: pd.Series,
    cov: pd.DataFrame,
    corr: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    extra_diagnostics: dict[str, Any] | None = None,
) -> DiagnosticsReport:
    """
    Generate portfolio diagnostics from one backtest path.

    Returns the metrics/data object only. Use save_diagnostics_bundle()
    to persist JSON, CSV and plots.
    """
    if weights.empty:
        raise ValueError("weights is empty.")

    r = (
        pd.Series(port_ret, dtype=float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if r.empty:
        raise ValueError("port_ret contains no finite observations.")

    w = (
        weights.copy()
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    latest_weights = w.iloc[-1]
    latest_weights = latest_weights[latest_weights.abs() > _EPS]

    dd = compute_drawdowns(r)
    turnover = compute_turnover(w)
    stability = compute_stability(w)

    cov_aligned = cov.loc[
        latest_weights.index,
        latest_weights.index,
    ]
    rc = compute_risk_contributions(
        latest_weights,
        cov_aligned,
    )

    weekly_rf = float(risk_free_rate) / WEEKS_PER_YEAR
    excess = r - weekly_rf
    weekly_vol = float(r.std(ddof=1))

    if weekly_vol > _EPS:
        sharpe = float(
            np.sqrt(WEEKS_PER_YEAR)
            * excess.mean()
            / weekly_vol
        )
    else:
        sharpe = 0.0

    equity = (1.0 + r).cumprod()

    return DiagnosticsReport(
        sharpe=sharpe,
        annual_return=float(r.mean() * WEEKS_PER_YEAR),
        annual_vol=float(weekly_vol * np.sqrt(WEEKS_PER_YEAR)),
        max_drawdown=float(dd.min()),
        ann_turnover=float(turnover.mean() * WEEKS_PER_YEAR),
        stability=float(stability),
        total_return=float(equity.iloc[-1] - 1.0),
        n_weeks=int(len(r)),
        latest_weights=latest_weights,
        risk_contributions=rc,
        extra_diagnostics=extra_diagnostics or {},
    )


def save_diagnostics_bundle(
    report: DiagnosticsReport,
    *,
    weights: pd.DataFrame,
    port_ret: pd.Series,
    alpha: pd.Series,
    cov: pd.DataFrame,
    corr: pd.DataFrame,
    output_dir: str | Path = "reports",
    prefix: str = "research",
    close_figures: bool = True,
) -> dict[str, Path]:
    """
    Save a complete diagnostics bundle:

      - summary JSON
      - summary CSV
      - latest weights CSV
      - risk contributions CSV
      - weight history CSV
      - portfolio return CSV
      - PNG plots
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "sharpe": report.sharpe,
        "annual_return": report.annual_return,
        "annual_vol": report.annual_vol,
        "max_drawdown": report.max_drawdown,
        "ann_turnover": report.ann_turnover,
        "stability": report.stability,
        "total_return": report.total_return,
        "n_weeks": report.n_weeks,
        "extra_diagnostics": _jsonable(report.extra_diagnostics),
    }

    paths: dict[str, Path] = {}

    json_path = out / f"{prefix}_summary.json"
    json_path.write_text(
        json.dumps(
            _jsonable(summary),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    paths["summary_json"] = json_path

    summary_csv = out / f"{prefix}_summary.csv"
    pd.Series(summary).drop(labels=["extra_diagnostics"]).to_csv(
        summary_csv,
        header=["value"],
    )
    paths["summary_csv"] = summary_csv

    latest_weights_path = out / f"{prefix}_latest_weights.csv"
    report.latest_weights.sort_values(ascending=False).to_csv(
        latest_weights_path,
        header=["weight"],
    )
    paths["latest_weights_csv"] = latest_weights_path

    rc_path = out / f"{prefix}_risk_contributions.csv"
    report.risk_contributions.sort_values(ascending=False).to_csv(
        rc_path,
        header=["risk_contribution"],
    )
    paths["risk_contributions_csv"] = rc_path

    weight_history_path = out / f"{prefix}_weight_history.csv"
    weights.to_csv(weight_history_path)
    paths["weight_history_csv"] = weight_history_path

    return_path = out / f"{prefix}_portfolio_returns.csv"
    pd.Series(port_ret, name="portfolio_return").to_csv(return_path)
    paths["portfolio_returns_csv"] = return_path

    figures = {
        "performance": plot_backtest_curve(
            port_ret,
            out / f"{prefix}_performance.png",
        ),
        "drawdowns": plot_drawdowns(
            port_ret,
            out / f"{prefix}_drawdowns.png",
        ),
        "turnover": plot_turnover(
            compute_turnover(weights),
            out / f"{prefix}_turnover.png",
        ),
        "covariance": plot_covariance(
            cov,
            out / f"{prefix}_covariance.png",
        ),
        "correlation": plot_correlation(
            corr,
            out / f"{prefix}_correlation.png",
        ),
        "alpha": plot_alpha(
            alpha,
            out / f"{prefix}_alpha.png",
        ),
        "weights": plot_weights(
            weights,
            out / f"{prefix}_weights.png",
        ),
        "latest_weights": plot_latest_weights(
            report.latest_weights,
            out / f"{prefix}_latest_weights.png",
        ),
        "risk_contributions": plot_risk_contributions(
            report.risk_contributions,
            out / f"{prefix}_risk_contributions.png",
        ),
    }

    for name in figures:
        paths[f"plot_{name}"] = out / f"{prefix}_{name}.png"

    if close_figures:
        for fig in figures.values():
            plt.close(fig)

    return paths
