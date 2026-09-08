from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from akm_hrp.overlay.bounds import apply_bounds

_EPS = 1e-12


def _ledoit_wolf_dual_solution(
    observations: np.ndarray,
    ridge_strength: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Solve the shrunk GMV system without constructing a wide covariance.

    When assets greatly outnumber observations, forming and factorizing the
    ``assets x assets`` matrix dominates runtime. Ledoit-Wolf has the form
    ``lambda I + c X'X``; Woodbury therefore moves the solve into the much
    smaller observation space while preserving the exact estimator.
    """

    x = np.asarray(observations, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1:
        raise ValueError("observations must be a non-empty two-dimensional array.")
    x = x - x.mean(axis=0, keepdims=True)
    n_samples, n_assets = x.shape
    squared = x * x
    empirical_variance = squared.sum(axis=0) / n_samples
    mu = float(empirical_variance.mean())

    # These identities reproduce sklearn's Ledoit-Wolf beta and delta without
    # materializing either X2'X2 or the p-by-p empirical covariance.
    row_squared_norm = squared.sum(axis=1)
    beta_sum = float(row_squared_norm @ row_squared_norm)
    gram = x @ x.T
    delta_sum = float(np.sum(gram * gram)) / n_samples**2
    beta = (
        beta_sum / n_samples - delta_sum
    ) / (n_assets * n_samples)
    delta = (
        delta_sum
        - 2.0 * mu * float(empirical_variance.sum())
        + n_assets * mu**2
    ) / n_assets
    beta = min(beta, delta)
    shrinkage = (
        0.0
        if abs(delta) <= _EPS or beta == 0.0
        else float(np.clip(beta / delta, 0.0, 1.0))
    )

    covariance_diagonal = (
        (1.0 - shrinkage) * empirical_variance + shrinkage * mu
    )
    covariance_scale = float(np.median(np.clip(covariance_diagonal, _EPS, None)))
    ridge = max(float(ridge_strength), 0.0) * covariance_scale
    diagonal_shift = shrinkage * mu
    regularized_shift = diagonal_shift + ridge
    if regularized_shift <= _EPS:
        # The caller supplies a positive ridge by default. Keep an explicitly
        # zero ridge numerically solvable for rank-deficient wide universes.
        regularized_shift = _EPS

    covariance_multiplier = (1.0 - shrinkage) / n_samples
    ones = np.ones(n_assets, dtype=float)
    if covariance_multiplier <= _EPS:
        precision_sum = ones / regularized_shift
    else:
        dual = np.eye(n_samples) + (
            covariance_multiplier / regularized_shift
        ) * gram
        correction = np.linalg.solve(dual, x @ ones)
        precision_sum = (
            ones / regularized_shift
            - covariance_multiplier
            * (x.T @ correction)
            / regularized_shift**2
        )

    # Report exact spectral condition numbers from the smaller of primal and
    # dual Gram matrices. Centering guarantees rank <= n_samples - 1.
    if n_assets <= n_samples:
        empirical_eigenvalues = np.linalg.eigvalsh((x.T @ x) / n_samples)
        empirical_min = max(float(empirical_eigenvalues[0]), 0.0)
        empirical_max = max(float(empirical_eigenvalues[-1]), 0.0)
    else:
        empirical_eigenvalues = np.linalg.eigvalsh(gram / n_samples)
        empirical_min = 0.0
        empirical_max = max(float(empirical_eigenvalues[-1]), 0.0)
    covariance_min = diagonal_shift + (1.0 - shrinkage) * empirical_min
    covariance_max = diagonal_shift + (1.0 - shrinkage) * empirical_max
    condition_before = (
        float("inf")
        if covariance_min <= _EPS
        else float(covariance_max / covariance_min)
    )
    condition_after = float(
        (covariance_max + ridge) / max(covariance_min + ridge, _EPS)
    )
    return (
        precision_sum,
        covariance_diagonal,
        shrinkage,
        ridge,
        condition_before,
        condition_after,
    )


@dataclass(frozen=True)
class RegularizedMinimumVarianceConfig:
    """Controls the long-only double-shrinkage minimum-variance allocator."""

    min_weight: float = 0.0
    max_weight: float = 0.10
    ridge_strength: float = 0.10
    inverse_volatility_shrinkage: float = 0.25
    previous_weight_shrinkage: float = 0.50


@dataclass(frozen=True)
class RegularizedMinimumVarianceDiagnostics:
    ledoit_wolf_shrinkage: float
    ridge: float
    condition_number_before_ridge: float
    condition_number_after_ridge: float
    negative_unconstrained_weight_share: float


class RegularizedMinimumVarianceAllocator:
    """Stable long-only global minimum-variance portfolio.

    The covariance matrix is shrunk by Ledoit-Wolf and regularized again with
    a scale-aware ridge. Portfolio weights are then shrunk toward inverse
    volatility and the previous target before projection onto the bounded
    simplex. The two weight anchors reduce the instability and turnover that
    make an unregularized minimum-variance portfolio fragile out of sample.
    """

    def __init__(
        self,
        config: RegularizedMinimumVarianceConfig | None = None,
    ) -> None:
        self.config = config or RegularizedMinimumVarianceConfig()
        self.last_diagnostics: RegularizedMinimumVarianceDiagnostics | None = None
        self._previous_weights: pd.Series | None = None

    def reset_state(self) -> None:
        self.last_diagnostics = None
        self._previous_weights = None

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame.")
        if returns.shape[1] < 2:
            raise ValueError("Minimum variance requires at least two assets.")

        window = (
            returns.copy()
            .astype(float)
            .sort_index()
            .replace([np.inf, -np.inf], np.nan)
        )
        means = window.mean(axis=0)
        valid = means.notna() & (window.std(axis=0, skipna=True) > _EPS)
        window = window.loc[:, valid]
        if window.shape[1] < 2:
            raise ValueError("Too few non-constant assets remain for allocation.")

        # Eligibility already limits interior gaps. Mean imputation retains the
        # remaining assets without injecting a directional return forecast.
        window = window.fillna(window.mean(axis=0))
        (
            precision_sum,
            variance,
            shrinkage,
            ridge,
            condition_before,
            condition_after,
        ) = _ledoit_wolf_dual_solution(
            window.to_numpy(dtype=float), self.config.ridge_strength
        )
        variance = np.clip(variance, _EPS, None)
        ones = np.ones(len(variance), dtype=float)
        denominator = float(ones @ precision_sum)
        if not np.isfinite(denominator) or abs(denominator) <= _EPS:
            raise RuntimeError("Regularized covariance produced an invalid solution.")

        unconstrained = precision_sum / denominator
        negative_share = float(np.mean(unconstrained < 0.0))
        gmv = np.clip(unconstrained, 0.0, None)
        if float(gmv.sum()) <= _EPS:
            gmv = 1.0 / np.sqrt(variance)
        gmv /= gmv.sum()

        inverse_volatility = 1.0 / np.sqrt(variance)
        inverse_volatility /= inverse_volatility.sum()
        target_shrinkage = float(
            np.clip(self.config.inverse_volatility_shrinkage, 0.0, 1.0)
        )
        weights = (
            (1.0 - target_shrinkage) * gmv
            + target_shrinkage * inverse_volatility
        )

        previous_shrinkage = float(
            np.clip(self.config.previous_weight_shrinkage, 0.0, 1.0)
        )
        if self._previous_weights is not None and previous_shrinkage > 0.0:
            previous = self._previous_weights.reindex(window.columns).fillna(0.0)
            if float(previous.sum()) > _EPS:
                previous /= previous.sum()
                weights = (
                    (1.0 - previous_shrinkage) * weights
                    + previous_shrinkage * previous.to_numpy(dtype=float)
                )

        result = apply_bounds(
            pd.Series(weights, index=window.columns, dtype=float),
            min_weight=self.config.min_weight,
            max_weight=self.config.max_weight,
        )
        self._previous_weights = result.copy()
        self.last_diagnostics = RegularizedMinimumVarianceDiagnostics(
            ledoit_wolf_shrinkage=shrinkage,
            ridge=ridge,
            condition_number_before_ridge=condition_before,
            condition_number_after_ridge=condition_after,
            negative_unconstrained_weight_share=negative_share,
        )
        return result
