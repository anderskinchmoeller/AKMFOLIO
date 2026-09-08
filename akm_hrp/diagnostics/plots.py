from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _finish(fig, path: str | Path | None = None):
    fig.tight_layout()

    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=160, bbox_inches="tight")

    return fig


def plot_backtest_curve(
    port_ret: pd.Series,
    path: str | Path | None = None,
):
    """
    Plot cumulative portfolio growth from periodic returns.
    """
    r = pd.Series(port_ret, dtype=float).fillna(0.0)
    cumulative = (1.0 + r).cumprod()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(cumulative.index, cumulative.values, label="Portfolio")
    ax.set_title("Backtest performance")
    ax.set_ylabel("Growth of 1")
    ax.grid(True, alpha=0.25)
    ax.legend()

    return _finish(fig, path)


def plot_drawdowns(
    port_ret: pd.Series,
    path: str | Path | None = None,
):
    """
    Plot portfolio drawdown through time.
    """
    r = pd.Series(port_ret, dtype=float).fillna(0.0)
    cumulative = (1.0 + r).cumprod()
    peak = cumulative.cummax()
    drawdown = cumulative / peak - 1.0

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(
        drawdown.index,
        drawdown.values,
        0.0,
        alpha=0.35,
    )
    ax.set_title("Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.25)

    return _finish(fig, path)


def _heatmap(
    matrix: pd.DataFrame,
    title: str,
    path: str | Path | None = None,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
):
    values = matrix.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)

    n = len(matrix.columns)
    if n <= 30:
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels(matrix.index, fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    return _finish(fig, path)


def plot_covariance(
    cov: pd.DataFrame,
    path: str | Path | None = None,
):
    return _heatmap(
        cov,
        "Covariance matrix",
        path,
    )


def plot_correlation(
    corr: pd.DataFrame,
    path: str | Path | None = None,
):
    return _heatmap(
        corr,
        "Correlation matrix",
        path,
        vmin=-1.0,
        vmax=1.0,
    )


def plot_alpha(
    alpha: pd.Series,
    path: str | Path | None = None,
    top_n: int | None = 30,
):
    """
    Plot cross-sectional alpha scores.

    By default only the 30 largest absolute scores are shown.
    """
    s = pd.Series(alpha, dtype=float).dropna()

    if top_n is not None and len(s) > top_n:
        keep = s.abs().nlargest(top_n).index
        s = s.loc[keep]

    s = s.sort_values()

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.22 * len(s))))
    ax.barh(s.index.astype(str), s.values)
    ax.set_title("Alpha scores")
    ax.set_xlabel("Score")
    ax.grid(True, axis="x", alpha=0.25)

    return _finish(fig, path)


def plot_weights(
    weights: pd.DataFrame,
    path: str | Path | None = None,
    top_n: int = 20,
):
    """
    Plot the largest average portfolio weights through time.
    """
    w = weights.copy().astype(float).fillna(0.0)

    if w.empty:
        raise ValueError("weights is empty.")

    average_weight = w.abs().mean().sort_values(ascending=False)
    keep = average_weight.head(max(1, int(top_n))).index
    shown = w.loc[:, keep]

    fig, ax = plt.subplots(figsize=(11, 5))
    for column in shown.columns:
        ax.plot(
            shown.index,
            shown[column].values,
            label=str(column),
            linewidth=1.0,
        )

    ax.set_title(f"Portfolio weights over time — top {len(keep)}")
    ax.set_ylabel("Weight")
    ax.grid(True, alpha=0.25)

    if len(keep) <= 20:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=7,
        )

    return _finish(fig, path)


def plot_latest_weights(
    weights: pd.Series,
    path: str | Path | None = None,
    top_n: int = 30,
):
    """
    Plot the current portfolio allocation.
    """
    w = pd.Series(weights, dtype=float).dropna()
    w = w[w.abs() > 0]

    if len(w) > top_n:
        w = w.abs().nlargest(top_n).index.to_series().map(weights)

    w = w.sort_values()

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.22 * len(w))))
    ax.barh(w.index.astype(str), w.values)
    ax.set_title("Latest portfolio weights")
    ax.set_xlabel("Weight")
    ax.grid(True, axis="x", alpha=0.25)

    return _finish(fig, path)


def plot_risk_contributions(
    rc: pd.Series,
    path: str | Path | None = None,
    top_n: int = 30,
):
    """
    Plot current normalized risk contributions.
    """
    s = pd.Series(rc, dtype=float).dropna()

    if len(s) > top_n:
        s = s.abs().nlargest(top_n).index.to_series().map(rc)

    s = s.sort_values()

    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.22 * len(s))))
    ax.barh(s.index.astype(str), s.values)
    ax.set_title("Risk contributions")
    ax.set_xlabel("Fraction of portfolio risk")
    ax.grid(True, axis="x", alpha=0.25)

    return _finish(fig, path)


def plot_turnover(
    turnover: pd.Series,
    path: str | Path | None = None,
):
    t = pd.Series(turnover, dtype=float).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t.index, t.values)
    ax.set_title("Portfolio turnover")
    ax.set_ylabel("L1 turnover")
    ax.grid(True, alpha=0.25)

    return _finish(fig, path)


def plot_cluster_stability(
    stability: pd.Series,
    path: str | Path | None = None,
    threshold: float | None = None,
):
    s = pd.Series(stability, dtype=float).dropna()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(s.index, s.values, label="Cluster stability")

    if threshold is not None:
        ax.axhline(
            float(threshold),
            linestyle="--",
            linewidth=1.0,
            label="Gating threshold",
        )

    ax.set_ylim(0.0, 1.05)
    ax.set_title("Cluster stability")
    ax.set_ylabel("Stability")
    ax.grid(True, alpha=0.25)
    ax.legend()

    return _finish(fig, path)


def plot_covariance_blend_weights(
    diagnostics: pd.DataFrame,
    path: str | Path | None = None,
):
    """
    Plot adaptive LW/EWMA/PCA blend weights through time.

    diagnostics must contain:
        lw_weight, ewma_weight, pca_weight
    """
    required = ["lw_weight", "ewma_weight", "pca_weight"]

    missing = [c for c in required if c not in diagnostics.columns]
    if missing:
        raise ValueError(
            f"Missing covariance diagnostics columns: {missing}"
        )

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for column, label in [
        ("lw_weight", "LW"),
        ("ewma_weight", "EWMA"),
        ("pca_weight", "PCA"),
    ]:
        ax.plot(
            diagnostics.index,
            diagnostics[column],
            label=label,
        )

    ax.set_title("Adaptive covariance blend")
    ax.set_ylabel("Estimator weight")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend()

    return _finish(fig, path)
