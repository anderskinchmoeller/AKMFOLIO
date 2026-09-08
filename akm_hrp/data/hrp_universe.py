from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass(frozen=True)
class UniverseAsset:
    """One investable exposure and its economic clustering role."""

    ticker: str
    cluster: str
    role: str


CRSP_HRP_ETF_TEMPLATE = (
    UniverseAsset("SPY", "equity", "US large-cap equity"),
    UniverseAsset("QQQ", "equity", "US growth and technology equity"),
    UniverseAsset("IWM", "equity", "US small-cap equity"),
    UniverseAsset("EFA", "equity", "developed-market equity"),
    UniverseAsset("EEM", "equity", "emerging-market equity"),
    UniverseAsset("SHY", "rates", "short US Treasury exposure"),
    UniverseAsset("IEF", "rates", "intermediate US Treasury exposure"),
    UniverseAsset("TLT", "rates", "long US Treasury exposure"),
    UniverseAsset("TIP", "rates", "US inflation-linked Treasury exposure"),
    UniverseAsset("LQD", "credit", "investment-grade corporate credit"),
    UniverseAsset("HYG", "credit", "high-yield corporate credit"),
    UniverseAsset("GLD", "real_asset", "gold exposure"),
    UniverseAsset("DBC", "real_asset", "broad commodity exposure"),
    UniverseAsset("VNQ", "real_asset", "US listed real estate exposure"),
    UniverseAsset("UUP", "currency", "US dollar exposure"),
)


@dataclass(frozen=True)
class HRPDistanceInputs:
    """Aligned return and dependence matrices used before HRP clustering."""

    returns: pd.DataFrame
    volatility_normalized_returns: pd.DataFrame
    spearman_correlation: pd.DataFrame
    denoised_correlation: pd.DataFrame
    angular_distance: pd.DataFrame
    shrinkage_intensity: float


def prepare_hrp_distance_inputs(
    weekly_returns: pd.DataFrame,
    *,
    pit_universe: pd.DataFrame | None = None,
    minimum_complete_weeks: int = 104,
) -> HRPDistanceInputs:
    """Create a complete, robust and positive-semidefinite HRP distance input.

    The denoised dependence estimator applies Ledoit-Wolf shrinkage to
    standardized within-asset ranks. Pearson correlation of ranks is Spearman
    correlation, so this retains rank robustness while shrinking noisy
    off-diagonal estimates toward a well-conditioned target.

    Volatility normalization is exported for diagnostics and algorithms that
    use Euclidean return distances. It does not change Pearson or Spearman
    correlation, which is already invariant to positive per-column scaling.
    """

    if minimum_complete_weeks < 2:
        raise ValueError("minimum_complete_weeks must be at least 2.")
    if not isinstance(weekly_returns, pd.DataFrame) or weekly_returns.empty:
        raise ValueError("weekly_returns must be a non-empty DataFrame.")

    clean = (
        weekly_returns.copy()
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .sort_index()
    )
    clean.columns = clean.columns.astype(str)
    if clean.index.has_duplicates:
        raise ValueError("weekly_returns contains duplicate dates.")
    if clean.columns.has_duplicates:
        raise ValueError("weekly_returns contains duplicate assets.")

    if pit_universe is not None:
        eligible = pit_universe.copy()
        eligible.columns = eligible.columns.astype(str)
        eligible = (
            eligible.reindex(index=clean.index, columns=clean.columns)
            .fillna(0)
            .astype(bool)
        )
        clean = clean.where(eligible)

    sufficient = clean.notna().sum(axis=0) >= int(minimum_complete_weeks)
    clean = clean.loc[:, sufficient]
    if clean.shape[1] < 2:
        raise ValueError(
            "Fewer than two assets have the required point-in-time history."
        )

    # A common observation window avoids pair-specific sample sizes that can
    # make a correlation matrix internally inconsistent for clustering.
    aligned = clean.dropna(axis=0, how="any")
    if len(aligned) < int(minimum_complete_weeks):
        raise ValueError(
            f"Only {len(aligned)} complete weeks remain; at least "
            f"{minimum_complete_weeks} are required."
        )

    volatility = aligned.std(axis=0, ddof=1)
    valid_volatility = volatility.replace([np.inf, -np.inf], np.nan).gt(0.0)
    aligned = aligned.loc[:, valid_volatility]
    volatility = volatility.loc[valid_volatility]
    if aligned.shape[1] < 2:
        raise ValueError("Fewer than two non-constant assets remain.")
    normalized = aligned.divide(volatility, axis=1)

    spearman = aligned.corr(method="spearman")

    ranks = aligned.rank(axis=0, method="average")
    rank_std = ranks.std(axis=0, ddof=1)
    standardized_ranks = (ranks - ranks.mean(axis=0)).divide(rank_std, axis=1)
    estimator = LedoitWolf(assume_centered=True).fit(
        standardized_ranks.to_numpy(dtype=float)
    )
    covariance = estimator.covariance_
    scale = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
    denoised_values = covariance / np.outer(scale, scale)
    denoised_values = np.clip(
        0.5 * (denoised_values + denoised_values.T),
        -1.0,
        1.0,
    )
    np.fill_diagonal(denoised_values, 1.0)
    denoised = pd.DataFrame(
        denoised_values,
        index=aligned.columns,
        columns=aligned.columns,
    )

    distance_values = np.sqrt(np.clip(0.5 * (1.0 - denoised_values), 0.0, 1.0))
    np.fill_diagonal(distance_values, 0.0)
    distance = pd.DataFrame(
        distance_values,
        index=aligned.columns,
        columns=aligned.columns,
    )

    return HRPDistanceInputs(
        returns=aligned,
        volatility_normalized_returns=normalized,
        spearman_correlation=spearman,
        denoised_correlation=denoised,
        angular_distance=distance,
        shrinkage_intensity=float(estimator.shrinkage_),
    )
