from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster


_EPS = 1e-12


@dataclass(frozen=True)
class ClusterStabilityDiagnostics:
    stability: float
    transition_rate: float
    transition_ema: float
    transition_spike: float
    bisection_aggressiveness: float
    risk_budget_weight: float
    n_common_assets: int
    n_clusters: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "stability": self.stability,
            "transition_rate": self.transition_rate,
            "transition_ema": self.transition_ema,
            "transition_spike": self.transition_spike,
            "bisection_aggressiveness": self.bisection_aggressiveness,
            "risk_budget_weight": self.risk_budget_weight,
            "n_common_assets": self.n_common_assets,
            "n_clusters": self.n_clusters,
        }


def choose_cluster_count(n_assets: int) -> int:
    """
    Stable, low-variance cut level for comparing cluster structure
    across adjacent rebalances.
    """
    if n_assets <= 2:
        return max(1, n_assets)

    return int(np.clip(round(np.sqrt(n_assets)), 2, n_assets))


def ensemble_coclustering(
    trees: list[np.ndarray],
    asset_names: list[str],
    n_clusters: int | None = None,
) -> pd.DataFrame:
    """
    Consensus co-clustering matrix.

    Entry (i, j) is the fraction of linkage methods placing assets i and j
    in the same cluster at the selected cut. This is invariant to arbitrary
    cluster label numbering.
    """
    if not trees:
        raise ValueError("At least one linkage tree is required.")

    n = len(asset_names)
    if n == 0:
        raise ValueError("asset_names is empty.")

    k = choose_cluster_count(n) if n_clusters is None else int(n_clusters)
    k = int(np.clip(k, 1, n))

    consensus = np.zeros((n, n), dtype=float)

    for tree in trees:
        labels = fcluster(tree, t=k, criterion="maxclust")
        consensus += (labels[:, None] == labels[None, :]).astype(float)

    consensus /= float(len(trees))
    np.fill_diagonal(consensus, 1.0)

    return pd.DataFrame(
        consensus,
        index=asset_names,
        columns=asset_names,
    )


def cluster_stability(
    previous: pd.DataFrame | None,
    current: pd.DataFrame,
    min_common_assets: int = 4,
) -> tuple[float, int]:
    """
    Compare two consensus co-clustering matrices.

    Stability is 1 minus the mean absolute change in pairwise co-clustering
    probability over the common asset universe.
    """
    if previous is None:
        return 1.0, 0

    common = previous.index.intersection(current.index)

    if len(common) < min_common_assets:
        # Not enough overlap for a meaningful transition estimate.
        return 1.0, int(len(common))

    prev = previous.loc[common, common].to_numpy(dtype=float)
    curr = current.loc[common, common].to_numpy(dtype=float)

    # Exclude diagonal: every asset is always co-clustered with itself.
    mask = ~np.eye(len(common), dtype=bool)

    delta = np.abs(curr - prev)[mask]
    if delta.size == 0:
        return 1.0, int(len(common))

    stability = 1.0 - float(np.mean(delta))
    return float(np.clip(stability, 0.0, 1.0)), int(len(common))


def transition_spike_score(
    transition_rate: float,
    previous_ema: float | None,
    ema_alpha: float = 0.25,
) -> tuple[float, float]:
    """
    Detect an increase in cluster transition intensity.

    Returns:
        spike score in [0, 1]
        updated EMA
    """
    transition = float(np.clip(transition_rate, 0.0, 1.0))
    alpha = float(np.clip(ema_alpha, _EPS, 1.0))

    if previous_ema is None:
        return 0.0, transition

    baseline = float(np.clip(previous_ema, 0.0, 1.0))
    excess = max(0.0, transition - baseline)

    # Normalize by the remaining headroom so a jump from an already-high
    # baseline is not understated.
    spike = excess / max(1.0 - baseline, _EPS)
    spike = float(np.clip(spike, 0.0, 1.0))

    updated = (1.0 - alpha) * baseline + alpha * transition
    return spike, float(np.clip(updated, 0.0, 1.0))


def gating_parameters(
    stability: float,
    transition_spike: float,
    *,
    stability_threshold: float = 0.70,
    min_bisection_aggressiveness: float = 0.35,
    base_risk_budget_weight: float = 0.00,
    max_risk_budget_weight: float = 0.45,
) -> tuple[float, float]:
    """
    Convert cluster diagnostics into HRP gating controls.

    - Below the stability threshold, recursive-bisection allocations are
      shrunk toward 50/50 splits.
    - Transition spikes increase the blend toward a risk-budget portfolio.
    """
    s = float(np.clip(stability, 0.0, 1.0))
    spike = float(np.clip(transition_spike, 0.0, 1.0))
    threshold = float(np.clip(stability_threshold, _EPS, 1.0))
    min_aggr = float(np.clip(min_bisection_aggressiveness, 0.0, 1.0))

    if s >= threshold:
        instability = 0.0
        aggressiveness = 1.0
    else:
        instability = (threshold - s) / threshold
        aggressiveness = 1.0 - instability * (1.0 - min_aggr)

    base_rb = float(np.clip(base_risk_budget_weight, 0.0, 1.0))
    max_rb = float(np.clip(max_risk_budget_weight, base_rb, 1.0))

    # Both broad instability and an abrupt transition can increase the
    # stabilizing risk-budget blend. Spike gets the larger coefficient.
    transition_pressure = float(
        np.clip(0.35 * instability + 0.65 * spike, 0.0, 1.0)
    )
    risk_budget_weight = base_rb + transition_pressure * (max_rb - base_rb)

    return (
        float(np.clip(aggressiveness, min_aggr, 1.0)),
        float(np.clip(risk_budget_weight, base_rb, max_rb)),
    )
