from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from akm_hrp.cov.ewma import ewma_covariance_array
from akm_hrp.cov.pca import pca_covariance_from_sample


_EPS = 1e-12


@dataclass(frozen=True)
class AdaptiveCovarianceConfig:
    """
    Configuration for shrinkage-adaptive LW/EWMA/PCA covariance blending.

    The neutral ensemble remains:
        0.50 Ledoit-Wolf
        0.30 EWMA
        0.20 PCA

    In stressed / poorly conditioned regimes, weight is shifted away from
    EWMA toward the more strongly regularized LW and PCA estimators.
    """

    base_lw_weight: float = 0.50
    base_ewma_weight: float = 0.30
    base_pca_weight: float = 0.20

    ewma_halflife: float = 26.0
    pca_explained_variance: float = 0.90

    # log10(condition number) >= this level is treated as maximum condition stress.
    condition_log10_cap: float = 4.0

    # Relative contribution of the three diagnostics to the regime stress score.
    condition_stress_weight: float = 0.45
    rank_stress_weight: float = 0.30
    shrinkage_stress_weight: float = 0.25

    # Prevent any estimator from disappearing completely.
    min_component_weight: float = 0.05


@dataclass(frozen=True)
class CovarianceBlendDiagnostics:
    condition_number: float
    effective_rank: float
    effective_rank_ratio: float
    shrinkage_intensity: float
    stress_score: float
    lw_weight: float
    ewma_weight: float
    pca_weight: float
    pca_components: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "condition_number": self.condition_number,
            "effective_rank": self.effective_rank,
            "effective_rank_ratio": self.effective_rank_ratio,
            "shrinkage_intensity": self.shrinkage_intensity,
            "stress_score": self.stress_score,
            "lw_weight": self.lw_weight,
            "ewma_weight": self.ewma_weight,
            "pca_weight": self.pca_weight,
            "pca_components": self.pca_components,
        }


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")

    x = returns.copy().astype(float).replace([np.inf, -np.inf], np.nan)

    if x.shape[0] < 2:
        raise ValueError("At least two observations are required.")

    if x.shape[1] < 2:
        raise ValueError("At least two assets are required.")

    # The backtest engine already removes sparse assets. Mean imputation here
    # only handles the small amount of interior missingness that remains.
    means = x.mean(axis=0)
    if means.isna().any():
        bad = means.index[means.isna()].tolist()
        raise ValueError(f"Assets with no usable observations: {bad}")

    x = x.fillna(means)

    std = x.std(axis=0, ddof=1)
    valid = std.notna() & (std > _EPS)
    x = x.loc[:, valid]

    if x.shape[1] < 2:
        raise ValueError("Too few non-constant assets for covariance estimation.")

    return x


def _nearest_psd(matrix: np.ndarray, floor: float = _EPS) -> np.ndarray:
    a = np.asarray(matrix, dtype=float)
    a = 0.5 * (a + a.T)

    eigvals, eigvecs = np.linalg.eigh(a)
    scale = max(float(np.max(np.abs(eigvals))), 1.0)
    eigvals = np.clip(eigvals, floor * scale, None)

    out = (eigvecs * eigvals) @ eigvecs.T
    return 0.5 * (out + out.T)


def _effective_rank(cov: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(_nearest_psd(cov))
    eigvals = np.clip(eigvals, 0.0, None)

    total = float(eigvals.sum())
    if total <= _EPS:
        return 1.0

    p = eigvals / total
    p = p[p > _EPS]

    entropy = -float(np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def _ewma_covariance(x: pd.DataFrame, halflife: float) -> np.ndarray:
    """Numba/BLAS-accelerated EWMA covariance."""
    values = np.ascontiguousarray(x.to_numpy(dtype=np.float64))
    return _nearest_psd(ewma_covariance_array(values, halflife))


def _pca_covariance(
    sample_cov: np.ndarray,
    explained_variance: float,
) -> tuple[np.ndarray, int]:
    """Numba-assisted PCA shrinkage covariance."""
    cov, k = pca_covariance_from_sample(
        sample_cov,
        explained_variance=explained_variance,
    )
    return _nearest_psd(cov), k


def _adaptive_weights(
    condition_number: float,
    effective_rank_ratio: float,
    shrinkage_intensity: float,
    cfg: AdaptiveCovarianceConfig,
) -> tuple[float, float, float, float]:
    """
    Map diagnostics to dynamic ensemble weights.

    Interpretation:
      - high condition number -> more regularization
      - low effective rank -> more factor/PCA support
      - high LW shrinkage intensity -> stronger evidence that the raw
        covariance is noisy, so more weight goes to LW
    """
    cond_log = np.log10(max(float(condition_number), 1.0))
    cond_stress = float(
        np.clip(cond_log / max(cfg.condition_log10_cap, _EPS), 0.0, 1.0)
    )

    rank_stress = float(np.clip(1.0 - effective_rank_ratio, 0.0, 1.0))
    shrink_stress = float(np.clip(shrinkage_intensity, 0.0, 1.0))

    stress_weights = np.asarray(
        [
            cfg.condition_stress_weight,
            cfg.rank_stress_weight,
            cfg.shrinkage_stress_weight,
        ],
        dtype=float,
    )
    stress_weights /= stress_weights.sum()

    stress = float(
        np.dot(
            stress_weights,
            np.asarray([cond_stress, rank_stress, shrink_stress], dtype=float),
        )
    )

    base = np.asarray(
        [
            cfg.base_lw_weight,
            cfg.base_ewma_weight,
            cfg.base_pca_weight,
        ],
        dtype=float,
    )
    if (base < 0).any() or base.sum() <= 0:
        raise ValueError("Base covariance ensemble weights must be non-negative.")
    base /= base.sum()

    # Stress target:
    # - LW rises with realized shrinkage and poor conditioning.
    # - PCA rises when effective rank collapses.
    # - EWMA is deliberately reduced when estimation risk is high.
    target_lw = (
        0.55
        + 0.25 * shrink_stress
        + 0.10 * cond_stress
    )
    target_pca = (
        0.20
        + 0.20 * rank_stress
        + 0.05 * cond_stress
        - 0.10 * shrink_stress
    )
    target_ewma = max(
        cfg.min_component_weight,
        1.0 - target_lw - target_pca,
    )

    target = np.asarray(
        [target_lw, target_ewma, target_pca],
        dtype=float,
    )
    target = np.clip(target, cfg.min_component_weight, None)
    target /= target.sum()

    adaptive = (1.0 - stress) * base + stress * target
    adaptive = np.clip(adaptive, cfg.min_component_weight, None)
    adaptive /= adaptive.sum()

    return (
        float(adaptive[0]),
        float(adaptive[1]),
        float(adaptive[2]),
        stress,
    )


def covariance_to_correlation(cov: pd.DataFrame) -> pd.DataFrame:
    values = cov.to_numpy(dtype=float)
    std = np.sqrt(np.clip(np.diag(values), _EPS, None))
    corr = values / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def adaptive_covariance_ensemble(
    returns: pd.DataFrame,
    config: AdaptiveCovarianceConfig | None = None,
) -> tuple[pd.DataFrame, CovarianceBlendDiagnostics]:
    """
    Estimate a shrinkage-adaptive covariance ensemble.

    Returns:
        blended covariance DataFrame
        diagnostics including dynamic estimator weights
    """
    scenarios, diagnostics = covariance_scenarios(
        returns,
        config=config,
    )
    return scenarios["blend"], diagnostics


def covariance_scenarios(
    returns: pd.DataFrame,
    config: AdaptiveCovarianceConfig | None = None,
    *,
    downside_quantile: float = 0.25,
    min_downside_observations: int = 20,
) -> tuple[dict[str, pd.DataFrame], CovarianceBlendDiagnostics]:
    """Estimate a small committee of plausible covariance matrices.

    The adaptive blend remains the reference estimator.  The component and
    downside matrices are retained as uncertainty scenarios for robust HRP.
    A shrinkage estimator is used for the downside sample because stressed
    observations are necessarily fewer and noisier.
    """
    cfg = config or AdaptiveCovarianceConfig()
    x = _clean_returns(returns)
    cols = x.columns
    values = x.to_numpy(dtype=float)

    lw_estimator = LedoitWolf(assume_centered=False).fit(values)
    lw_cov = _nearest_psd(lw_estimator.covariance_)
    shrinkage = float(np.clip(lw_estimator.shrinkage_, 0.0, 1.0))

    eigvals = np.linalg.eigvalsh(lw_cov)
    positive = eigvals[eigvals > _EPS]
    condition_number = (
        float(positive.max() / positive.min())
        if positive.size
        else float("inf")
    )

    effective_rank = _effective_rank(lw_cov)
    rank_ratio = float(np.clip(effective_rank / len(cols), 0.0, 1.0))
    ewma_cov = _ewma_covariance(x, cfg.ewma_halflife)

    sample_cov = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    pca_cov, pca_components = _pca_covariance(
        sample_cov,
        cfg.pca_explained_variance,
    )

    lw_w, ewma_w, pca_w, stress = _adaptive_weights(
        condition_number=condition_number,
        effective_rank_ratio=rank_ratio,
        shrinkage_intensity=shrinkage,
        cfg=cfg,
    )
    blended = _nearest_psd(
        lw_w * lw_cov + ewma_w * ewma_cov + pca_w * pca_cov
    )

    q = float(np.clip(downside_quantile, 0.05, 0.50))
    market_proxy = x.mean(axis=1)
    downside = x.loc[market_proxy <= market_proxy.quantile(q)]
    if len(downside) >= max(2, int(min_downside_observations)):
        downside_cov = LedoitWolf(assume_centered=False).fit(
            downside.to_numpy(dtype=float)
        ).covariance_
        downside_cov = _nearest_psd(downside_cov)
    else:
        downside_cov = blended.copy()

    diagnostics = CovarianceBlendDiagnostics(
        condition_number=condition_number,
        effective_rank=effective_rank,
        effective_rank_ratio=rank_ratio,
        shrinkage_intensity=shrinkage,
        stress_score=stress,
        lw_weight=lw_w,
        ewma_weight=ewma_w,
        pca_weight=pca_w,
        pca_components=pca_components,
    )

    def frame(matrix: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(matrix, index=cols, columns=cols)

    return {
        "blend": frame(blended),
        "ledoit_wolf": frame(lw_cov),
        "ewma": frame(ewma_cov),
        "pca": frame(pca_cov),
        "downside": frame(downside_cov),
    }, diagnostics
