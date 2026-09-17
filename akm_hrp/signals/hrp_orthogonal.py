"""Make alpha signals cross-sectionally orthogonal to HRP risk clusters.

HRP groups assets by hierarchical correlation. A signal that loads on those
same correlation clusters (or on broad market beta) does not add
idiosyncratic information: it simply over-weights a whole cluster that the
HRP recursive bisection has already sized by its variance. This module strips
those components out of a raw score before it reaches the optimizer.

Pipeline for one raw score on one formation date::

    raw --(rank-normal on available names)--> r
    r   --(least-squares residual on basis B)--> e
    e   --(exact de-mean, unit std)----------> clean

with basis ``B = [1, beta, cluster dummies, (industry), (styles)]``.

The clusters use the same correlation distance and average linkage as the
allocator's own HRP tree (``d_ij = sqrt((1 - rho_ij) / 2)``), computed on the
*broad* eligible universe from a random-matrix-denoised correlation matrix
(eigenvalues above the Marchenko-Pastur edge only), because the allocator's
own tree is built after selection on ~40 names and does not exist yet at
signal time.

Everything here only reads the trailing window passed in, so it is causal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm

_EPS = 1e-12
UNASSIGNED = "CL_UNASSIGNED"


@dataclass(frozen=True)
class OrthogonalizationConfig:
    cluster_lookback_weeks: int = 104
    cluster_min_observations: int = 52
    max_clusters: int = 24
    min_cluster_size: int = 8
    max_pca_components: int = 12
    # Recompute clusters every N calls (clusters are slow-moving; the
    # O(n^2) linkage is the only expensive step in this module).
    cluster_refresh_calls: int = 4
    beta_lookback_weeks: int = 104
    beta_min_observations: int = 26
    # "equal" makes the output exactly de-meaned and exactly orthogonal to
    # the basis in the plain cross-section. "sqrt_cap" matches the
    # weighting of the codebase's existing neutralisation.
    weighting: str = "equal"


def _rank_normal(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce")
    if clean.notna().sum() < 3:
        return pd.Series(0.0, index=values.index)
    lower, upper = clean.quantile([0.01, 0.99])
    ranks = clean.clip(lower, upper).rank(method="average", pct=True)
    n = int(clean.notna().sum())
    probability = ranks.clip(1.0 / (2.0 * n), 1.0 - 1.0 / (2.0 * n))
    out = pd.Series(norm.ppf(probability), index=values.index)
    return out.replace([np.inf, -np.inf], np.nan)


def market_proxy(returns: pd.DataFrame) -> pd.Series:
    """Equal-weighted cross-sectional mean return of the eligible universe."""

    return returns.mean(axis=1, skipna=True).fillna(0.0)


def shrunk_beta(
    returns: pd.DataFrame,
    lookback: int = 104,
    min_observations: int = 26,
) -> pd.Series:
    """Rolling OLS beta on the universe mean, Vasicek-shrunk toward the
    cross-sectional mean beta. Names with too little history get the prior."""

    window = returns.iloc[-lookback:]
    market = market_proxy(window)
    values = window.to_numpy(dtype=float)
    observed = np.isfinite(values)
    m = market.to_numpy(dtype=float)[:, None]
    count = observed.sum(axis=0).astype(float)
    m_obs = np.where(observed, m, 0.0)
    r_obs = np.where(observed, values, 0.0)
    safe = np.maximum(count, 1.0)
    m_mean = m_obs.sum(axis=0) / safe
    r_mean = r_obs.sum(axis=0) / safe
    m_c = np.where(observed, m - m_mean, 0.0)
    r_c = np.where(observed, values - r_mean, 0.0)
    var_m = (m_c**2).sum(axis=0)
    cov = (m_c * r_c).sum(axis=0)
    beta = np.where(var_m > _EPS, cov / np.maximum(var_m, _EPS), np.nan)
    resid = np.where(observed, r_c - beta[None, :] * m_c, 0.0)
    resid_var = (resid**2).sum(axis=0) / np.maximum(count - 2.0, 1.0)
    se2 = resid_var / np.maximum(var_m, _EPS)
    enough = count >= min_observations
    beta = np.where(enough, beta, np.nan)
    finite = np.isfinite(beta)
    if finite.sum() < 3:
        return pd.Series(1.0, index=returns.columns)
    prior_mean = float(np.nanmean(beta))
    prior_var = float(np.nanvar(beta))
    weight = prior_var / (prior_var + se2 + _EPS)
    shrunk = np.where(finite, weight * beta + (1.0 - weight) * prior_mean, prior_mean)
    return pd.Series(shrunk, index=returns.columns)


def _denoised_correlation(
    window: pd.DataFrame, max_components: int
) -> np.ndarray:
    values = window.to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std > _EPS, std, 1.0)
    z = np.nan_to_num((values - mean) / std, nan=0.0)
    t, n = z.shape
    _, s, vt = np.linalg.svd(z / np.sqrt(t), full_matrices=False)
    eigen = s**2
    # Marchenko-Pastur upper edge for a pure-noise correlation matrix.
    edge = (1.0 + np.sqrt(n / t)) ** 2
    k = int(np.clip((eigen > edge).sum(), 1, max_components))
    loadings = vt[:k].T * s[:k]
    corr = loadings @ loadings.T
    norms = np.sqrt(np.clip(np.diag(corr), _EPS, None))
    # Re-normalise on the systematic part so names with a large idiosyncratic
    # share are still placed by *which* factors they load on.
    corr = corr / norms[:, None] / norms[None, :]
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def correlation_clusters(
    returns: pd.DataFrame,
    config: OrthogonalizationConfig | None = None,
) -> pd.Series:
    """HRP-style hierarchical correlation clusters for every column.

    Small clusters are pooled into ``CL_UNASSIGNED`` so the dummy basis does
    not spend degrees of freedom on a handful of names.
    """

    cfg = config or OrthogonalizationConfig()
    window = returns.iloc[-cfg.cluster_lookback_weeks :]
    enough = window.notna().sum() >= cfg.cluster_min_observations
    names = window.columns[enough.to_numpy()]
    labels = pd.Series(UNASSIGNED, index=returns.columns, dtype=object)
    if len(names) < max(2 * cfg.min_cluster_size, 4):
        return labels
    corr = _denoised_correlation(window.loc[:, names], cfg.max_pca_components)
    distance = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    raw = fcluster(tree, t=cfg.max_clusters, criterion="maxclust")
    assigned = pd.Series(raw, index=names)
    sizes = assigned.value_counts()
    big = sizes.index[sizes >= cfg.min_cluster_size]
    for cluster_id in big:
        labels.loc[assigned.index[assigned.eq(cluster_id)]] = f"CL_{int(cluster_id):03d}"
    return labels


class ClusterCache:
    """Refreshes the broad-universe clusters every few calls.

    Between refreshes, names that entered the universe are left
    ``CL_UNASSIGNED`` (absorbed by the intercept), never forced into a cluster
    using data they did not have at the refresh date.
    """

    def __init__(self, config: OrthogonalizationConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._labels: pd.Series | None = None
        self._calls_since_refresh = 0
        self._last_date: pd.Timestamp | None = None

    def labels(self, returns: pd.DataFrame) -> pd.Series:
        as_of = pd.Timestamp(returns.index[-1])
        stale = (
            self._labels is None
            or self._calls_since_refresh >= self.config.cluster_refresh_calls
            or (self._last_date is not None and as_of < self._last_date)
        )
        if stale:
            self._labels = correlation_clusters(returns, self.config)
            self._calls_since_refresh = 0
        self._calls_since_refresh += 1
        self._last_date = as_of
        return self._labels.reindex(returns.columns).fillna(UNASSIGNED)


def build_basis(
    assets: pd.Index,
    beta: pd.Series,
    clusters: pd.Series,
    exposures: pd.DataFrame | None = None,
    *,
    include_industry: bool = True,
    include_styles: bool = False,
) -> pd.DataFrame:
    """Nuisance basis the signal is made orthogonal to."""

    parts = [pd.DataFrame({"INTERCEPT": 1.0}, index=assets)]
    b = beta.reindex(assets).astype(float)
    b = b.fillna(b.median() if b.notna().any() else 1.0)
    parts.append(pd.DataFrame({"BETA": b}, index=assets))
    dummies = pd.get_dummies(
        clusters.reindex(assets).fillna(UNASSIGNED), dtype=float
    )
    dummies = dummies.drop(columns=[UNASSIGNED], errors="ignore")
    if dummies.shape[1] and float(dummies.sum(axis=1).min()) >= 1.0:
        # Every name belongs to some cluster: drop one to avoid collinearity
        # with the intercept (lstsq would cope, but diagnostics are cleaner).
        dummies = dummies.iloc[:, 1:]
    parts.append(dummies)
    if exposures is not None and not exposures.empty:
        frame = exposures.reindex(assets)
        if include_industry:
            parts.append(frame.loc[:, [c for c in frame if str(c).startswith("IND_")]])
        if include_styles:
            styles = [
                c
                for c in ("SIZE", "VALUE", "MOMENTUM", "QUALITY", "LOW_VOL")
                if c in frame
            ]
            parts.append(frame.loc[:, styles])
    basis = pd.concat(parts, axis=1).fillna(0.0).astype(float)
    return basis


def _weights(
    names: pd.Index, market_caps: pd.Series | None, weighting: str
) -> np.ndarray:
    if weighting == "equal" or market_caps is None:
        return np.ones(len(names))
    if weighting != "sqrt_cap":
        raise ValueError(f"Unknown weighting {weighting!r}.")
    caps = market_caps.reindex(names).astype(float)
    caps = caps.fillna(caps.median() if caps.notna().any() else 1.0)
    return np.sqrt(caps.clip(lower=_EPS).to_numpy(dtype=float))


def orthogonalize(
    raw: pd.Series,
    available: pd.Series,
    basis: pd.DataFrame,
    *,
    market_caps: pd.Series | None = None,
    weighting: str = "equal",
    rank_transform: bool = True,
    min_names: int = 20,
) -> pd.Series:
    """Rank-normal -> residualise on ``basis`` -> exact de-mean -> unit std.

    Unavailable names receive 0 (neutral). Returns all zeros when fewer than
    ``min_names`` usable names exist.
    """

    assets = basis.index
    result = pd.Series(0.0, index=assets)
    values = pd.to_numeric(raw.reindex(assets), errors="coerce")
    mask = available.reindex(assets).fillna(False).astype(bool) & values.notna()
    names = assets[mask.to_numpy()]
    if len(names) < min_names:
        return result
    y = _rank_normal(values.loc[names]) if rank_transform else values.loc[names]
    y = y.fillna(0.0).to_numpy(dtype=float)
    x = basis.loc[names]
    # Drop columns with no variation among the available names (e.g. a
    # cluster none of them belongs to) -- the intercept is kept.
    keep = [c for c in x.columns if c == "INTERCEPT" or float(x[c].std(ddof=0)) > _EPS]
    x = x.loc[:, keep].to_numpy(dtype=float)
    if x.shape[1] >= len(names) - 5:
        return result
    root = np.sqrt(_weights(names, market_caps, weighting))
    coef = np.linalg.lstsq(x * root[:, None], y * root, rcond=1e-10)[0]
    residual = y - x @ coef
    w = root**2
    residual = residual - float(np.average(residual, weights=w))
    scale = float(np.sqrt(np.average(residual**2, weights=w)))
    if not np.isfinite(scale) or scale <= 1e-10:
        return result
    result.loc[names] = residual / scale
    return result


def exposure_diagnostics(
    score: pd.Series,
    available: pd.Series,
    clusters: pd.Series,
    beta: pd.Series,
) -> dict[str, float]:
    """How much of a score is explained by clusters and beta."""

    mask = available.reindex(score.index).fillna(False).astype(bool)
    s = pd.to_numeric(score[mask], errors="coerce").dropna()
    if len(s) < 10 or float(s.std(ddof=0)) <= _EPS:
        return {"n": float(len(s)), "cluster_r2": np.nan, "beta_corr": np.nan,
                "max_abs_cluster_mean": np.nan}
    labels = clusters.reindex(s.index).fillna(UNASSIGNED)
    means = s.groupby(labels).transform("mean")
    total = float(((s - s.mean()) ** 2).sum())
    between = float(((means - s.mean()) ** 2).sum())
    b = beta.reindex(s.index)
    beta_corr = float(np.corrcoef(s, b.fillna(b.median()))[0, 1]) if b.std() > _EPS else 0.0
    grouped = s.groupby(labels).mean()
    grouped = grouped.drop(UNASSIGNED, errors="ignore")
    return {
        "n": float(len(s)),
        "cluster_r2": between / max(total, _EPS),
        "beta_corr": beta_corr,
        "max_abs_cluster_mean": float(grouped.abs().max()) if len(grouped) else 0.0,
    }
