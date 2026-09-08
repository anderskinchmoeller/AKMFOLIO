import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

def correlation_distance(corr: pd.DataFrame) -> np.ndarray:
    """
    Convert correlation matrix to distance matrix:
        d_ij = sqrt(0.5 * (1 - corr_ij))
    """
    if corr.empty:
        raise ValueError("Correlation matrix is empty.")

    values = corr.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        bad = corr.columns[
            ~np.isfinite(values).all(axis=0)
        ].tolist()

        raise ValueError(
            "Correlation matrix contains non-finite values. "
            f"Problem assets: {bad}"
        )

    c = np.clip(values, -1.0, 1.0)

    # Protect the diagonal from floating-point noise.
    np.fill_diagonal(c, 1.0)

    d = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - c)))
    np.fill_diagonal(d, 0.0)

    return d

def _single_tree(dist: np.ndarray, method: str) -> np.ndarray:
    """
    Build a single linkage tree from a distance matrix.
    """
    condensed = squareform(dist, checks=False)
    return linkage(condensed, method=method)

def build_tree_ensemble(
    corr: pd.DataFrame,
    methods=("single", "average", "complete")
):
    """
    Build an ensemble of linkage trees using multiple methods.
    Returns list of SciPy linkage matrices.
    """
    dist = correlation_distance(corr)
    trees = []
    for m in methods:
        trees.append(_single_tree(dist, m))
    return trees


def build_consensus_tree(
    coclustering: pd.DataFrame,
    method: str = "average",
) -> np.ndarray:
    """Build one stable tree from pairwise co-clustering probabilities."""
    if coclustering.empty:
        raise ValueError("Co-clustering matrix is empty.")

    values = coclustering.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Co-clustering matrix contains non-finite values.")

    values = np.clip(0.5 * (values + values.T), 0.0, 1.0)
    np.fill_diagonal(values, 1.0)
    distance = np.sqrt(np.maximum(0.0, 1.0 - values))
    np.fill_diagonal(distance, 0.0)
    return _single_tree(distance, method)

def quasi_diagonalize(tree: np.ndarray, asset_names: list[str]) -> list[str]:
    """
    Convert a linkage tree into an ordered list of tickers.
    """
    n = len(asset_names)
    clusters = {i: [i] for i in range(n)}

    for idx, (c1, c2, _, _) in enumerate(tree):
        c1, c2 = int(c1), int(c2)
        new_cluster = clusters[c1] + clusters[c2]
        clusters[n + idx] = new_cluster

    final_cluster = clusters[max(clusters)]
    return [asset_names[i] for i in final_cluster]
