from __future__ import annotations

import numpy as np
import pandas as pd

from akm_hrp.hrp.trees import quasi_diagonalize


_EPS = 1e-12


def _cluster_variance(cov: pd.DataFrame, cluster: list[str]) -> float:
    """
    Compute equal-weight cluster variance.
    """
    sub = cov.loc[cluster, cluster].to_numpy(dtype=float)
    w = np.ones(len(cluster), dtype=float) / len(cluster)
    return float(w @ sub @ w)


def recursive_bisection(
    order: list[str],
    cov: pd.DataFrame,
    aggressiveness: float = 1.0,
) -> pd.Series:
    """
    HRP recursive bisection.

    aggressiveness=1.0:
        standard HRP variance-driven split.

    aggressiveness<1.0:
        shrink each split toward 50/50, reducing sensitivity to an unstable
        cluster tree during correlation-regime transitions.
    """
    a = float(np.clip(aggressiveness, 0.0, 1.0))
    weights = pd.Series(1.0, index=order, dtype=float)

    def split_and_assign(cluster: list[str]) -> None:
        if len(cluster) <= 1:
            return

        mid = len(cluster) // 2
        left = cluster[:mid]
        right = cluster[mid:]

        var_left = max(_cluster_variance(cov, left), _EPS)
        var_right = max(_cluster_variance(cov, right), _EPS)

        raw_left = var_right / (var_left + var_right)

        # Gating: low aggressiveness moves a variance-driven split toward 50/50.
        alloc_left = 0.5 + a * (raw_left - 0.5)
        alloc_left = float(np.clip(alloc_left, 0.0, 1.0))
        alloc_right = 1.0 - alloc_left

        weights.loc[left] *= alloc_left
        weights.loc[right] *= alloc_right

        split_and_assign(left)
        split_and_assign(right)

    split_and_assign(order)

    total = float(weights.sum())
    if total > _EPS:
        weights /= total

    return weights


def inverse_vol_risk_budget(cov: pd.DataFrame) -> pd.Series:
    """
    Stable equal-risk proxy used as the transition risk-budget anchor.
    """
    variance = pd.Series(
        np.diag(cov.to_numpy(dtype=float)),
        index=cov.index,
        dtype=float,
    ).clip(lower=_EPS)

    inv_vol = 1.0 / np.sqrt(variance)
    total = float(inv_vol.sum())

    if total <= _EPS or not np.isfinite(total):
        return pd.Series(1.0 / len(cov), index=cov.index, dtype=float)

    return inv_vol / total


def risk_contribution_regularize(
    weights: pd.Series,
    cov: pd.DataFrame,
    cap: float = 0.20,
) -> pd.Series:
    """
    Cap total risk-contribution fractions at a fixed threshold.
    """
    w = weights.to_numpy(dtype=float)
    C = cov.loc[weights.index, weights.index].to_numpy(dtype=float)

    port_var = float(w @ C @ w)
    if not np.isfinite(port_var) or port_var <= _EPS:
        return weights / weights.sum()

    port_vol = np.sqrt(port_var)
    mrc = (C @ w) / port_vol
    rc = w * mrc
    total = float(rc.sum())

    if not np.isfinite(total) or abs(total) <= _EPS:
        return weights / weights.sum()

    rc_frac = rc / total
    excess = rc_frac > cap

    if not excess.any():
        return weights / weights.sum()

    shrink = np.where(
        excess,
        cap / np.maximum(rc_frac, _EPS),
        1.0,
    )

    w_new = np.clip(w * shrink, 0.0, None)
    total_new = float(w_new.sum())

    if total_new <= _EPS:
        return weights / weights.sum()

    return pd.Series(w_new / total_new, index=weights.index)


def hrp_allocate(
    corr: pd.DataFrame,
    cov: pd.DataFrame,
    tree,
    asset_names: list[str],
    risk_cap: float = 0.20,
    bisection_aggressiveness: float = 1.0,
    risk_budget_weight: float = 0.0,
) -> pd.Series:
    """
    Full gated HRP pipeline:
      1. quasi-diagonalize tree
      2. recursive bisection with stability-controlled aggressiveness
      3. blend toward inverse-vol risk budget during cluster transitions
      4. risk-contribution regularization
    """
    order = quasi_diagonalize(tree, asset_names)

    raw = recursive_bisection(
        order,
        cov,
        aggressiveness=bisection_aggressiveness,
    )

    rb_weight = float(np.clip(risk_budget_weight, 0.0, 1.0))

    if rb_weight > 0.0:
        risk_budget = inverse_vol_risk_budget(cov).reindex(order)
        raw = (1.0 - rb_weight) * raw + rb_weight * risk_budget
        raw /= raw.sum()

    reg = risk_contribution_regularize(
        raw,
        cov,
        cap=risk_cap,
    )

    return reg
