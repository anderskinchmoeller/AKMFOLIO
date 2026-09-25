"""Schur complementary HRP allocation on a fixed linkage tree.

Cotton (2024, arXiv:2411.05807) generalises Lopez de Prado's HRP bisection by
conditioning each cluster on its complement. At a node whose children split the
covariance as ``Sigma = [[A, B], [B', D]]``, the left child is allocated inside

    S_A = A - gamma * B D^-1 B'        (conditioned covariance)
    b_A = 1 - gamma * B D^-1 1         (hedge tilt)
    M_A = diag(b_A)^-1 S_A diag(b_A)^-1

and receives the un-normalised budget ``diag(b_A)^-1 u_A / (u_A' M_A u_A)``,
where ``u_A`` is the (recursively computed) allocation inside ``M_A``. The right
child is symmetric.

* ``gamma = 0`` leaves ``b = 1`` and ``M_A = A``: the classic HRP split, with the
  cluster budget equal to the inverse of its own recursive-weight variance.
* ``gamma = 1`` gives ``w proportional to Sigma^-1 1``, the unconstrained global
  minimum-variance portfolio, by induction over the tree.

Long-only guard: if any entry of ``b`` at a split falls to ``min_b`` or below,
gamma is halved at that split only until it is positive, so every weight stays
strictly positive without an optimiser.

Split rule (``split``):

* ``"midpoint"`` (default) quasi-diagonalises the tree and halves the ordered
  leaves at each step, as in Lopez de Prado's HRP, Cotton's paper and
  ``hrp/allocation.py``'s ``recursive_bisection``.
* ``"dendrogram"`` splits on the linkage tree's own children instead, as
  ``hrp/return_adjusted.py`` does. Under single linkage this chains: the tree is
  deeply unbalanced, so the conditioned-variance budget concentrates. Measured on
  the sector-balanced universe (2016-2025, quarterly), it gives an effective N of
  about 10 against 80 for the midpoint rule, a top weight near 40% against 7%,
  and double the turnover. Use it only with a balanced linkage such as Ward.

The traversal is iterative: a chain-like tree is as deep as the universe is wide.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from akm_hrp.hrp.trees import quasi_diagonalize

_EPS = 1e-12


@dataclass(frozen=True)
class SchurHRPDiagnostics:
    node_count: int
    gamma_reduced_count: int
    gamma: float
    split: str
    min_weight: float
    max_weight: float


def _conditioned_block(
    a: np.ndarray,
    b_blk: np.ndarray,
    d: np.ndarray,
    gamma: float,
    min_b: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Augmented matrix and hedge tilt for block ``a`` given its complement."""
    size = a.shape[0]
    if gamma <= 0.0:
        return a, np.ones(size), False

    try:
        d_inv_bt = np.linalg.solve(d, b_blk.T)
        d_inv_one = np.linalg.solve(d, np.ones(d.shape[0]))
    except np.linalg.LinAlgError:
        return a, np.ones(size), True

    correction = b_blk @ d_inv_bt
    tilt = b_blk @ d_inv_one

    g = float(gamma)
    while g > 1e-6:
        tilt_vector = 1.0 - g * tilt
        if tilt_vector.min() > min_b:
            conditioned = a - g * correction
            scaled = conditioned / np.outer(tilt_vector, tilt_vector)
            return (scaled + scaled.T) / 2.0, tilt_vector, g < gamma
        g *= 0.5
    return a, np.ones(size), True


def schur_hrp_allocate(
    covariance: pd.DataFrame,
    tree: np.ndarray,
    asset_names: list[str],
    *,
    gamma: float = 0.5,
    min_b: float = 1e-3,
    long_only: bool = True,
    split: str = "midpoint",
) -> tuple[pd.Series, SchurHRPDiagnostics]:
    """Allocate with fixed-tree Schur complementary HRP.

    ``long_only=False`` disables the tilt guard, which is what makes the
    gamma = 1 minimum-variance limit exact (and lets weights go negative).
    """
    gamma = float(gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between zero and one.")
    if min_b <= 0.0:
        raise ValueError("min_b must be positive.")
    if split not in ("midpoint", "dendrogram"):
        raise ValueError('split must be "midpoint" or "dendrogram".')
    guard = min_b if long_only else -np.inf

    assets = list(asset_names)
    if len(assets) < 2:
        raise ValueError("Schur HRP requires at least two assets.")
    tree = np.asarray(tree, dtype=float)
    expected_shape = (len(assets) - 1, 4)
    if tree.shape != expected_shape:
        raise ValueError(
            f"tree must have shape {expected_shape}; received {tree.shape}."
        )
    if set(covariance.index) != set(assets) or set(covariance.columns) != set(assets):
        raise ValueError("covariance labels must match asset_names.")

    covariance = covariance.loc[assets, assets].astype(float)
    if not np.isfinite(covariance.to_numpy()).all():
        raise ValueError("covariance contains non-finite values.")

    n_assets = len(assets)
    children: dict[int, tuple[int, int]] = {}
    positions: dict[int, list[int]] = {leaf: [leaf] for leaf in range(n_assets)}
    for row_number, row in enumerate(tree):
        node_id = n_assets + row_number
        left_id, right_id = int(row[0]), int(row[1])
        if left_id not in positions or right_id not in positions:
            raise ValueError("tree contains an invalid or forward child reference.")
        if set(positions[left_id]).intersection(positions[right_id]):
            raise ValueError("tree child clusters overlap.")
        children[node_id] = (left_id, right_id)
        positions[node_id] = positions[left_id] + positions[right_id]

    root = n_assets + len(tree) - 1
    if split == "midpoint":
        # Keep the tree's leaf order, but halve it at every step.
        order = [assets.index(name) for name in quasi_diagonalize(tree, assets)]
        children = {}
        positions = {leaf: [leaf] for leaf in range(n_assets)}
        next_id = n_assets
        root = next_id
        positions[root] = order
        next_id += 1
        pending = [root]
        while pending:
            node = pending.pop()
            members = positions[node]
            if len(members) <= 1:
                continue
            mid = len(members) // 2
            halves = []
            for part in (members[:mid], members[mid:]):
                if len(part) == 1:
                    halves.append(part[0])
                else:
                    child_id = next_id
                    next_id += 1
                    positions[child_id] = part
                    halves.append(child_id)
                    pending.append(child_id)
            children[node] = (halves[0], halves[1])
    values = covariance.to_numpy(dtype=float)

    # Downward pass: every node's augmented matrix depends on its parent's.
    matrices: dict[int, np.ndarray] = {root: values[np.ix_(positions[root], positions[root])]}
    tilts: dict[int, np.ndarray] = {}
    order: list[int] = []
    gamma_reduced = 0
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        if node not in children:
            continue
        left_id, right_id = children[node]
        matrix = matrices[node]
        size_left = len(positions[left_id])
        left_slice = slice(0, size_left)
        right_slice = slice(size_left, matrix.shape[0])
        a = matrix[left_slice, left_slice]
        b_blk = matrix[left_slice, right_slice]
        d = matrix[right_slice, right_slice]

        m_left, b_left, cut_left = _conditioned_block(a, b_blk, d, gamma, guard)
        m_right, b_right, cut_right = _conditioned_block(d, b_blk.T, a, gamma, guard)
        gamma_reduced += int(cut_left) + int(cut_right)

        matrices[left_id], tilts[left_id] = m_left, b_left
        matrices[right_id], tilts[right_id] = m_right, b_right
        stack.extend((left_id, right_id))

    # Upward pass: children are always discovered after their parent.
    allocations: dict[int, np.ndarray] = {}
    for node in reversed(order):
        if node not in children:
            allocations[node] = np.ones(1)
            continue
        parts = []
        for child in children[node]:
            u = allocations.pop(child)
            m = matrices[child]
            variance = max(float(u @ m @ u), _EPS)
            parts.append((u / tilts[child]) / variance)
        combined = np.concatenate(parts)
        total = float(combined.sum())
        allocations[node] = combined / total if abs(total) > _EPS else combined

    weights = pd.Series(0.0, index=assets, dtype=float)
    ordered_assets = [assets[i] for i in positions[root]]
    weights.loc[ordered_assets] = allocations[root]

    diagnostics = SchurHRPDiagnostics(
        node_count=len(children),
        gamma_reduced_count=gamma_reduced,
        gamma=gamma,
        split=split,
        min_weight=float(weights.min()),
        max_weight=float(weights.max()),
    )
    return weights, diagnostics
