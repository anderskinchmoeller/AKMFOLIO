from __future__ import annotations

"""A transparent Barra-style statistical risk model followed by HRP.

This is not a licensed MSCI Barra model.  It implements the public factor-risk
architecture: regress assets on a compact factor panel, shrink factor covariance,
retain diagonal specific risk, and cluster the reconstructed asset covariance.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from akm_hrp.cov.ensemble import covariance_to_correlation
from akm_hrp.hrp.allocation import hrp_allocate
from akm_hrp.overlay.bounds import apply_bounds

_EPS = 1e-12


@dataclass(frozen=True)
class BarraFactorHRPConfig:
    """Controls for the public, factor-plus-specific-risk covariance model."""

    max_factors: int = 8
    explained_variance: float = 0.80
    minimum_external_factor_coverage: float = 0.80
    regression_ridge: float = 1e-4
    specific_variance_shrinkage: float = 0.35
    # Prevent the factor model from explaining away nearly all single-name
    # risk, a common source of unstable extreme weights.
    specific_variance_floor_fraction: float = 0.05
    # Calibrated for the balanced CRSP universe (roughly 49 names currently):
    # 3% remains feasible with >=34 names and yields about 41 effective names.
    max_weight: float = 0.03
    risk_contribution_cap: float = 0.08
    linkage_method: str = "average"


@dataclass(frozen=True)
class BarraFactorHRPDiagnostics:
    factor_source: str
    factor_count: int
    factor_covariance_shrinkage: float
    specific_variance_shrinkage: float
    covariance_condition_number: float
    covariance_minimum_eigenvalue: float
    effective_asset_count: float = float("nan")
    maximum_weight: float = float("nan")
    maximum_risk_contribution: float = float("nan")

    def as_dict(self) -> dict[str, float | int | str]:
        return self.__dict__.copy()


def _statistical_factors(
    returns: pd.DataFrame, config: BarraFactorHRPConfig
) -> pd.DataFrame:
    standardized = (returns - returns.mean()) / returns.std(ddof=1).clip(lower=_EPS)
    pca = PCA(
        n_components=min(
            config.max_factors, standardized.shape[0], standardized.shape[1]
        )
    )
    scores = pca.fit_transform(standardized)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    count = min(
        int(np.searchsorted(cumulative, config.explained_variance) + 1),
        config.max_factors,
    )
    return pd.DataFrame(
        scores[:, :count],
        index=returns.index,
        columns=[f"PCA_{i + 1}" for i in range(count)],
    )


def barra_factor_covariance(
    returns: pd.DataFrame,
    config: BarraFactorHRPConfig | None = None,
    *,
    factor_returns: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, BarraFactorHRPDiagnostics]:
    """Estimate ``B F B' + D`` using factor shrinkage and diagonal specific risk."""

    cfg = config or BarraFactorHRPConfig()
    assets = returns.columns.astype(str)
    x_frame = returns.loc[:, assets].astype(float).replace([np.inf, -np.inf], np.nan)
    use_external = factor_returns is not None
    if use_external:
        aligned = (
            factor_returns.copy()
            .astype(float)
            .reindex(x_frame.index)
            .replace([np.inf, -np.inf], np.nan)
        )
        complete_asset_rows = x_frame.notna().all(axis=1)
        minimum_rows = max(
            3,
            int(np.ceil(cfg.minimum_external_factor_coverage * len(x_frame))),
        )
        usable_columns = aligned.columns[
            aligned.loc[complete_asset_rows].notna().sum(axis=0) >= minimum_rows
        ]
        factors = aligned.loc[:, usable_columns[: cfg.max_factors]]
        valid = complete_asset_rows & factors.notna().all(axis=1)
        x_external, factors = x_frame.loc[valid], factors.loc[valid]
        nonconstant = factors.std(ddof=1) > _EPS
        factors = factors.loc[:, nonconstant]
        use_external = len(x_external) >= 3 and factors.shape[1] >= 1

    if not use_external:
        x_frame = x_frame.dropna(axis=0, how="any")
        factors = _statistical_factors(x_frame, cfg)
        source = "statistical_pca"
    else:
        x_frame = x_external
        source = "external"
    if len(x_frame) < 3 or factors.shape[1] < 1:
        raise ValueError(
            "Factor model requires at least three complete observations and one factor."
        )

    x = x_frame.to_numpy(dtype=float)
    f = factors.to_numpy(dtype=float)
    x_centered, f_centered = x - x.mean(0), f - f.mean(0)
    gram = f_centered.T @ f_centered
    scale = max(float(np.trace(gram) / len(gram)), _EPS)
    beta = np.linalg.solve(
        gram + cfg.regression_ridge * scale * np.eye(len(gram)),
        f_centered.T @ x_centered,
    )
    residuals = x_centered - f_centered @ beta
    factor_estimator = LedoitWolf().fit(f_centered)
    factor_covariance = factor_estimator.covariance_
    specific = np.var(residuals, axis=0, ddof=1).clip(min=_EPS)
    specific = (
        1.0 - cfg.specific_variance_shrinkage
    ) * specific + cfg.specific_variance_shrinkage * np.median(specific)
    asset_variance = np.var(x_centered, axis=0, ddof=1).clip(min=_EPS)
    specific = np.maximum(
        specific,
        cfg.specific_variance_floor_fraction * asset_variance,
    )
    values = beta.T @ factor_covariance @ beta + np.diag(specific)
    values = 0.5 * (values + values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    floor = max(float(np.median(np.diag(values))) * 1e-8, _EPS)
    values = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    covariance = pd.DataFrame(values, index=assets, columns=assets)
    diagnostics = BarraFactorHRPDiagnostics(
        factor_source=source,
        factor_count=factors.shape[1],
        factor_covariance_shrinkage=float(factor_estimator.shrinkage_),
        specific_variance_shrinkage=float(cfg.specific_variance_shrinkage),
        covariance_condition_number=float(np.linalg.cond(values)),
        covariance_minimum_eigenvalue=float(np.linalg.eigvalsh(values)[0]),
    )
    return covariance, diagnostics


class BarraFactorHRPAllocator:
    """Long-only HRP allocation using a factor-model-denoised covariance matrix."""

    def __init__(
        self,
        config: BarraFactorHRPConfig | None = None,
        *,
        factor_returns: pd.DataFrame | None = None,
    ) -> None:
        self.config = config or BarraFactorHRPConfig()
        self.factor_returns = factor_returns
        self.last_diagnostics: BarraFactorHRPDiagnostics | None = None
        self.last_covariance: pd.DataFrame | None = None
        if not 0.0 < self.config.max_weight <= 1.0:
            raise ValueError("max_weight must be in (0, 1].")
        if not 0.0 < self.config.risk_contribution_cap <= 1.0:
            raise ValueError("risk_contribution_cap must be in (0, 1].")
        if not 0.0 <= self.config.specific_variance_shrinkage <= 1.0:
            raise ValueError("specific_variance_shrinkage must be in [0, 1].")
        if not 0.0 <= self.config.specific_variance_floor_fraction <= 1.0:
            raise ValueError("specific_variance_floor_fraction must be in [0, 1].")
        if not 0.0 < self.config.minimum_external_factor_coverage <= 1.0:
            raise ValueError("minimum_external_factor_coverage must be in (0, 1].")

    def reset_state(self) -> None:
        self.last_diagnostics = None
        self.last_covariance = None

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        window = returns.copy().astype(float).sort_index()
        valid_assets = window.columns[window.notna().sum() >= 3]
        window = window.loc[:, valid_assets].fillna(window.mean())
        if window.shape[1] < 2:
            raise ValueError("Too few assets remain for Barra Factor HRP.")
        covariance, diagnostics = barra_factor_covariance(
            window, self.config, factor_returns=self.factor_returns
        )
        correlation = covariance_to_correlation(covariance)
        distance = np.sqrt(
            np.maximum(0.5 * (1.0 - correlation.to_numpy(dtype=float)), 0.0)
        )
        np.fill_diagonal(distance, 0.0)
        tree = linkage(
            squareform(distance, checks=False), method=self.config.linkage_method
        )
        weights = hrp_allocate(
            correlation,
            covariance,
            tree,
            list(covariance.columns),
            risk_cap=self.config.risk_contribution_cap,
        )
        final = apply_bounds(weights, 0.0, self.config.max_weight)
        values = final.to_numpy(dtype=float)
        # HRP returns its quasi-diagonal leaf order, which is generally not the
        # original covariance-column order.  Align by label before calculating
        # portfolio risk contributions.
        aligned_covariance = covariance.loc[final.index, final.index]
        marginal_risk = aligned_covariance.to_numpy(dtype=float) @ values
        risk_contribution = values * marginal_risk
        total_risk = float(risk_contribution.sum())
        maximum_risk_contribution = (
            float(risk_contribution.max() / total_risk)
            if total_risk > _EPS
            else float("nan")
        )
        self.last_diagnostics = replace(
            diagnostics,
            effective_asset_count=float(1.0 / np.sum(values**2)),
            maximum_weight=float(final.max()),
            maximum_risk_contribution=maximum_risk_contribution,
        )
        self.last_covariance = covariance
        return final
