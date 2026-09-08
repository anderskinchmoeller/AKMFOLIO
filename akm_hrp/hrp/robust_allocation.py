from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from akm_hrp.hrp.allocation import (
    inverse_vol_risk_budget,
    risk_contribution_regularize,
)
from akm_hrp.hrp.trees import quasi_diagonalize


_EPS = 1e-12


@dataclass(frozen=True)
class RobustAllocationDiagnostics:
    scenario_count: int
    split_count: int
    mean_worst_regret: float
    max_worst_regret: float
    mean_split_turnover: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "scenario_count": self.scenario_count,
            "split_count": self.split_count,
            "mean_worst_regret": self.mean_worst_regret,
            "max_worst_regret": self.max_worst_regret,
            "mean_split_turnover": self.mean_split_turnover,
        }


def _inverse_variance_weights(
    covariance: pd.DataFrame,
    assets: list[str],
) -> np.ndarray:
    variances = np.diag(
        covariance.loc[assets, assets].to_numpy(dtype=float)
    )
    inv = 1.0 / np.clip(variances, _EPS, None)
    total = float(inv.sum())
    if not np.isfinite(total) or total <= _EPS:
        return np.full(len(assets), 1.0 / len(assets), dtype=float)
    return inv / total


def _aggregate_moments(
    scenario: pd.DataFrame,
    reference: pd.DataFrame,
    left: list[str],
    right: list[str],
) -> tuple[float, float, float]:
    wl = _inverse_variance_weights(reference, left)
    wr = _inverse_variance_weights(reference, right)

    ll = scenario.loc[left, left].to_numpy(dtype=float)
    rr = scenario.loc[right, right].to_numpy(dtype=float)
    lr = scenario.loc[left, right].to_numpy(dtype=float)

    var_left = max(float(wl @ ll @ wl), _EPS)
    var_right = max(float(wr @ rr @ wr), _EPS)
    cross = float(wl @ lr @ wr)
    return var_left, var_right, cross


def _scenario_variance(
    allocation_left: np.ndarray,
    var_left: float,
    var_right: float,
    cross: float,
) -> np.ndarray:
    a = allocation_left
    return (
        a * a * var_left
        + (1.0 - a) * (1.0 - a) * var_right
        + 2.0 * a * (1.0 - a) * cross
    )


def robust_split(
    scenarios: dict[str, pd.DataFrame],
    reference: pd.DataFrame,
    left: list[str],
    right: list[str],
    *,
    previous_split: float | None = None,
    minimum: float = 0.10,
    maximum: float = 0.90,
    grid_size: int = 101,
    turnover_penalty: float = 0.02,
) -> tuple[float, float, float]:
    """Choose the branch weight with the smallest worst-case risk regret."""
    lo = float(np.clip(minimum, 0.0, 0.5))
    hi = float(np.clip(maximum, 0.5, 1.0))
    if lo > hi:
        raise ValueError("minimum split exceeds maximum split")

    grid = np.linspace(lo, hi, max(11, int(grid_size)), dtype=float)
    regrets = []

    for scenario in scenarios.values():
        vl, vr, cross = _aggregate_moments(
            scenario,
            reference,
            left,
            right,
        )
        risk = np.maximum(
            _scenario_variance(grid, vl, vr, cross),
            _EPS,
        )
        regrets.append(risk / float(risk.min()) - 1.0)

    worst_regret = np.max(np.vstack(regrets), axis=0)

    if previous_split is not None and np.isfinite(previous_split):
        prior = float(np.clip(previous_split, lo, hi))
        split_turnover = np.abs(grid - prior)
        objective = worst_regret + float(max(turnover_penalty, 0.0)) * split_turnover
    else:
        prior = 0.5
        split_turnover = np.zeros_like(grid)
        objective = worst_regret

    # A deterministic centre preference resolves numerically equal solutions.
    objective = objective + 1e-10 * np.abs(grid - 0.5)
    selected = int(np.argmin(objective))
    return (
        float(grid[selected]),
        float(worst_regret[selected]),
        float(split_turnover[selected]),
    )


def regret_aware_hrp_allocate(
    reference_covariance: pd.DataFrame,
    covariance_scenarios: dict[str, pd.DataFrame],
    tree: np.ndarray,
    asset_names: list[str],
    *,
    previous_weights: pd.Series | None = None,
    risk_cap: float = 0.20,
    bisection_aggressiveness: float = 1.0,
    risk_budget_weight: float = 0.0,
    minimum_split: float = 0.10,
    maximum_split: float = 0.90,
    grid_size: int = 101,
    turnover_penalty: float = 0.02,
) -> tuple[pd.Series, RobustAllocationDiagnostics]:
    """Allocate down a consensus tree using worst-case scenario regret."""
    order = quasi_diagonalize(tree, asset_names)
    weights = pd.Series(1.0, index=order, dtype=float)
    previous = (
        previous_weights.reindex(order).fillna(0.0).clip(lower=0.0)
        if previous_weights is not None
        else None
    )
    aggressiveness = float(np.clip(bisection_aggressiveness, 0.0, 1.0))
    regrets: list[float] = []
    split_turnovers: list[float] = []

    def split_and_assign(cluster: list[str]) -> None:
        if len(cluster) <= 1:
            return

        mid = len(cluster) // 2
        left = cluster[:mid]
        right = cluster[mid:]

        previous_split = None
        if previous is not None:
            node_mass = float(previous.loc[cluster].sum())
            if node_mass > _EPS:
                previous_split = float(previous.loc[left].sum() / node_mass)

        raw_left, regret, split_turnover = robust_split(
            covariance_scenarios,
            reference_covariance,
            left,
            right,
            previous_split=previous_split,
            minimum=minimum_split,
            maximum=maximum_split,
            grid_size=grid_size,
            turnover_penalty=turnover_penalty,
        )
        alloc_left = 0.5 + aggressiveness * (raw_left - 0.5)
        alloc_left = float(np.clip(alloc_left, minimum_split, maximum_split))

        weights.loc[left] *= alloc_left
        weights.loc[right] *= 1.0 - alloc_left
        regrets.append(regret)
        split_turnovers.append(split_turnover)

        split_and_assign(left)
        split_and_assign(right)

    split_and_assign(order)
    weights = weights.clip(lower=0.0)
    weights /= weights.sum()

    rb_weight = float(np.clip(risk_budget_weight, 0.0, 1.0))
    if rb_weight > 0.0:
        risk_budget = inverse_vol_risk_budget(reference_covariance).reindex(order)
        weights = (1.0 - rb_weight) * weights + rb_weight * risk_budget
        weights /= weights.sum()

    weights = risk_contribution_regularize(
        weights,
        reference_covariance,
        cap=risk_cap,
    )

    diagnostics = RobustAllocationDiagnostics(
        scenario_count=len(covariance_scenarios),
        split_count=len(regrets),
        mean_worst_regret=float(np.mean(regrets)) if regrets else 0.0,
        max_worst_regret=float(np.max(regrets)) if regrets else 0.0,
        mean_split_turnover=(
            float(np.mean(split_turnovers)) if split_turnovers else 0.0
        ),
    )
    return weights, diagnostics
