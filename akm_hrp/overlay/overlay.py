import pandas as pd

from .signal_budget import signal_budget_overlay
from .cov_inverse import cov_inverse_overlay
from .bounds import apply_bounds


def apply_overlays(
    hrp_weights: pd.Series,
    alpha: pd.Series,
    cov: pd.DataFrame,
    *,
    use_signal_budget: bool = True,
    use_cov_inverse: bool = True,
    budget_beta: float = 0.20,
    budget_tau: float = 0.75,
    beamforming_beta: float = 0.20,
    beamforming_loading: float = 0.10,
    beamforming_strength: float = 0.50,
    min_weight: float = 0.00,
    max_weight: float = 0.10,
    asset_caps: dict[str, float] | None = None,
    max_active_share: float | None = 0.10,
) -> pd.Series:
    """
    Combine HRP weights with overlays:
      w_final ∝ w_hrp * overlay_signal * overlay_capon
    """
    base = apply_bounds(
        hrp_weights,
        min_weight=min_weight,
        max_weight=max_weight,
        asset_caps=asset_caps,
    )
    overlay = pd.Series(1.0, index=base.index)

    if use_signal_budget:
        sb = signal_budget_overlay(alpha, beta=budget_beta, tau=budget_tau)
        overlay *= sb

    if use_cov_inverse:
        ci = cov_inverse_overlay(
            cov,
            alpha,
            beta=beamforming_beta,
            loading=beamforming_loading,
            strength=beamforming_strength,
        )
        overlay *= ci

    w = base * overlay
    w = w.clip(lower=0.0)

    if w.sum() > 0:
        w /= w.sum()
    else:
        w[:] = 1.0 / len(w)

    w = apply_bounds(
        w,
        min_weight=min_weight,
        max_weight=max_weight,
        asset_caps=asset_caps,
    )

    if max_active_share is not None:
        cap = float(max_active_share)
        if not 0.0 <= cap <= 1.0:
            raise ValueError("max_active_share must be between zero and one.")
        active_share = 0.5 * float((w - base).abs().sum())
        if active_share > cap and active_share > 0.0:
            scale = cap / active_share
            w = base + scale * (w - base)
            w = apply_bounds(
                w,
                min_weight=min_weight,
                max_weight=max_weight,
                asset_caps=asset_caps,
            )

    return w
