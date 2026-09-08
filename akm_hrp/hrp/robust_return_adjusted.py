from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from akm_hrp.hrp.allocation import (
    inverse_vol_risk_budget,
    risk_contribution_regularize,
)

_EPS = 1e-12


@dataclass(frozen=True)
class RobustReturnAdjustedDiagnostics:
    """Summary of the robust decisions made down one consensus tree."""

    scenario_count: int
    node_count: int
    floor_activation_count: int
    mean_target_deviation: float
    mean_worst_risk_regret: float
    max_worst_risk_regret: float
    mean_split_turnover: float
    interpolation: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "scenario_count": self.scenario_count,
            "node_count": self.node_count,
            "floor_activation_count": self.floor_activation_count,
            "mean_target_deviation": self.mean_target_deviation,
            "mean_worst_risk_regret": self.mean_worst_risk_regret,
            "max_worst_risk_regret": self.max_worst_risk_regret,
            "mean_split_turnover": self.mean_split_turnover,
            "interpolation": self.interpolation,
        }


def _inverse_variance_weights(
    covariance: pd.DataFrame,
    assets: list[str],
) -> np.ndarray:
    variances = np.diag(covariance.loc[assets, assets].to_numpy(dtype=float))
    inverse = 1.0 / np.clip(variances, _EPS, None)
    return inverse / inverse.sum()


def _cluster_moments(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    assets: list[str],
) -> tuple[float, float, np.ndarray]:
    weights = _inverse_variance_weights(covariance, assets)
    matrix = covariance.loc[assets, assets].to_numpy(dtype=float)
    mean = float(weights @ expected_returns.loc[assets].to_numpy(dtype=float))
    variance = max(float(weights @ matrix @ weights), _EPS)
    return mean, variance, weights


def _scenario_branch_variance(
    grid: np.ndarray,
    scenario: pd.DataFrame,
    left: list[str],
    right: list[str],
    left_ivp: np.ndarray,
    right_ivp: np.ndarray,
) -> np.ndarray:
    left_cov = scenario.loc[left, left].to_numpy(dtype=float)
    right_cov = scenario.loc[right, right].to_numpy(dtype=float)
    cross_cov = scenario.loc[left, right].to_numpy(dtype=float)
    left_variance = max(float(left_ivp @ left_cov @ left_ivp), _EPS)
    right_variance = max(float(right_ivp @ right_cov @ right_ivp), _EPS)
    cross = float(left_ivp @ cross_cov @ right_ivp)
    return np.maximum(
        grid**2 * left_variance
        + (1.0 - grid) ** 2 * right_variance
        + 2.0 * grid * (1.0 - grid) * cross,
        _EPS,
    )


def robust_return_adjusted_hrp_allocate(
    expected_returns: pd.Series,
    reference_covariance: pd.DataFrame,
    covariance_scenarios: dict[str, pd.DataFrame],
    tree: np.ndarray,
    asset_names: list[str],
    *,
    previous_weights: pd.Series | None = None,
    score_floor: float = 1e-4,
    interpolation: float = 0.75,
    return_target_weight: float = 1.0,
    risk_regret_weight: float = 1.0,
    turnover_penalty: float = 0.02,
    minimum_split: float = 0.10,
    maximum_split: float = 0.90,
    grid_size: int = 101,
    bisection_aggressiveness: float = 1.0,
    risk_budget_weight: float = 0.0,
    risk_cap: float = 0.20,
) -> tuple[pd.Series, RobustReturnAdjustedDiagnostics]:
    """Allocate down an exact consensus tree with return and risk robustness.

    The paper's RA-HRP split remains the return target. A grid search then
    balances distance from that target against the worst normalized variance
    regret over all covariance scenarios and, when available, split turnover.
    """
    assets = list(asset_names)
    if len(assets) < 2:
        raise ValueError("Robust RA-HRP requires at least two assets.")
    if score_floor <= 0.0:
        raise ValueError("score_floor must be positive.")
    if not 0.0 <= interpolation <= 1.0:
        raise ValueError("interpolation must be between zero and one.")
    if return_target_weight < 0.0 or risk_regret_weight < 0.0:
        raise ValueError("objective weights must be non-negative.")
    if not covariance_scenarios:
        raise ValueError("at least one covariance scenario is required.")

    reference = reference_covariance.loc[assets, assets].astype(float)
    expected = expected_returns.reindex(assets).astype(float)
    scenarios = {
        name: covariance.loc[assets, assets].astype(float)
        for name, covariance in covariance_scenarios.items()
    }
    matrices = [reference, *scenarios.values()]
    if not np.isfinite(expected.to_numpy()).all():
        raise ValueError("expected_returns contains non-finite values.")
    if any(not np.isfinite(matrix.to_numpy()).all() for matrix in matrices):
        raise ValueError("one or more covariance matrices contain non-finite values.")

    tree = np.asarray(tree, dtype=float)
    expected_shape = (len(assets) - 1, 4)
    if tree.shape != expected_shape:
        raise ValueError(f"tree must have shape {expected_shape}; received {tree.shape}.")

    n_assets = len(assets)
    children: dict[int, tuple[int, int]] = {}
    members: dict[int, list[str]] = {
        leaf_id: [asset] for leaf_id, asset in enumerate(assets)
    }
    for row_number, row in enumerate(tree):
        node_id = n_assets + row_number
        left_id, right_id = int(row[0]), int(row[1])
        if left_id not in members or right_id not in members:
            raise ValueError("tree contains an invalid or forward child reference.")
        children[node_id] = (left_id, right_id)
        members[node_id] = members[left_id] + members[right_id]

    root_id = n_assets + len(tree) - 1
    if len(members[root_id]) != len(set(members[root_id])):
        raise ValueError("tree contains duplicate leaves.")
    if set(members[root_id]) != set(assets):
        raise ValueError("tree root does not contain every asset exactly once.")

    lo = float(np.clip(minimum_split, 0.0, 0.5))
    hi = float(np.clip(maximum_split, 0.5, 1.0))
    if lo > hi:
        raise ValueError("minimum_split exceeds maximum_split.")
    grid = np.linspace(lo, hi, max(11, int(grid_size)), dtype=float)
    aggressiveness = float(np.clip(bisection_aggressiveness, 0.0, 1.0))
    previous = (
        previous_weights.reindex(assets).fillna(0.0).clip(lower=0.0)
        if previous_weights is not None
        else None
    )

    weights = pd.Series(1.0, index=assets, dtype=float)
    floor_activations = 0
    target_deviations: list[float] = []
    worst_regrets: list[float] = []
    split_turnovers: list[float] = []

    # Traverse explicitly: valid, highly unbalanced linkage trees can be much
    # deeper than Python's recursion limit in broad equity universes.
    pending = [root_id]
    while pending:
        node_id = pending.pop()
        if node_id < n_assets:
            continue

        left_id, right_id = children[node_id]
        left, right = members[left_id], members[right_id]
        left_mean, left_variance, left_ivp = _cluster_moments(
            expected, reference, left
        )
        right_mean, right_variance, right_ivp = _cluster_moments(
            expected, reference, right
        )

        left_raw_score = left_mean / np.sqrt(left_variance)
        right_raw_score = right_mean / np.sqrt(right_variance)
        floor_activations += int(left_raw_score < score_floor)
        floor_activations += int(right_raw_score < score_floor)
        left_score = max(left_raw_score, score_floor)
        right_score = max(right_raw_score, score_floor)

        hrp_target = right_variance / (left_variance + right_variance)
        ra_target = left_score / (left_score + right_score)
        target = (1.0 - interpolation) * hrp_target + interpolation * ra_target
        target = float(np.clip(target, lo, hi))

        regrets = []
        for scenario in scenarios.values():
            variance = _scenario_branch_variance(
                grid,
                scenario,
                left,
                right,
                left_ivp,
                right_ivp,
            )
            regrets.append(variance / float(variance.min()) - 1.0)
        worst_regret = np.max(np.vstack(regrets), axis=0)

        previous_split = None
        if previous is not None:
            node_mass = float(previous.loc[left + right].sum())
            if node_mass > _EPS:
                previous_split = float(previous.loc[left].sum() / node_mass)

        target_penalty = (grid - target) ** 2
        turnover = (
            np.abs(grid - np.clip(previous_split, lo, hi))
            if previous_split is not None
            else np.zeros_like(grid)
        )
        objective = (
            return_target_weight * target_penalty
            + risk_regret_weight * worst_regret
            + max(turnover_penalty, 0.0) * turnover
            + 1e-10 * np.abs(grid - 0.5)
        )
        selected = int(np.argmin(objective))
        raw_left = float(grid[selected])
        left_split = 0.5 + aggressiveness * (raw_left - 0.5)
        left_split = float(np.clip(left_split, lo, hi))

        weights.loc[left] *= left_split
        weights.loc[right] *= 1.0 - left_split
        target_deviations.append(abs(raw_left - target))
        worst_regrets.append(float(worst_regret[selected]))
        split_turnovers.append(float(turnover[selected]))

        pending.append(right_id)
        pending.append(left_id)
    weights = weights.clip(lower=0.0)
    weights /= weights.sum()

    rb_weight = float(np.clip(risk_budget_weight, 0.0, 1.0))
    if rb_weight > 0.0:
        risk_budget = inverse_vol_risk_budget(reference).reindex(assets)
        weights = (1.0 - rb_weight) * weights + rb_weight * risk_budget
        weights /= weights.sum()

    weights = risk_contribution_regularize(weights, reference, cap=risk_cap)
    diagnostics = RobustReturnAdjustedDiagnostics(
        scenario_count=len(scenarios),
        node_count=len(target_deviations),
        floor_activation_count=floor_activations,
        mean_target_deviation=(
            float(np.mean(target_deviations)) if target_deviations else 0.0
        ),
        mean_worst_risk_regret=(
            float(np.mean(worst_regrets)) if worst_regrets else 0.0
        ),
        max_worst_risk_regret=(
            float(np.max(worst_regrets)) if worst_regrets else 0.0
        ),
        mean_split_turnover=(
            float(np.mean(split_turnovers)) if split_turnovers else 0.0
        ),
        interpolation=float(interpolation),
    )
    return weights, diagnostics
