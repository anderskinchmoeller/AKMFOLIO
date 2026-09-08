import numpy as np
import pandas as pd

def capon_direction(
    cov: pd.DataFrame,
    alpha: pd.Series,
    loading: float = 0.10,
) -> pd.Series:
    """
    Covariance-inverse direction with diagonal loading:
      w ∝ (R + λI)^(-1) a
    """
    cols = alpha.index
    R = cov.loc[cols, cols].values
    a = alpha.fillna(0.0).values

    # Scale diagonal loading to the covariance magnitude.  A raw 0.10 added
    # to weekly-return variances (~1e-3) overwhelms the covariance structure.
    diag_scale = float(np.trace(R) / len(cols)) if len(cols) else 0.0
    if not np.isfinite(diag_scale) or diag_scale <= 0.0:
        diag_scale = 1.0
    R_loaded = R + (loading * diag_scale) * np.eye(len(cols))

    try:
        inv = np.linalg.inv(R_loaded)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(R_loaded)

    w = inv @ a
    w = np.maximum(w, 0.0)

    if w.sum() == 0:
        return pd.Series(0.0, index=cols)

    return pd.Series(w / w.sum(), index=cols)


def cov_inverse_overlay(
    cov: pd.DataFrame,
    alpha: pd.Series,
    beta: float = 0.20,
    loading: float = 0.10,
    strength: float = 0.50,
) -> pd.Series:
    """
    Covariance-inverse overlay:
      w_capon = capon_direction(cov, alpha)
      overlay = 1 + beta * strength * (w_capon - 1/N)
    """
    w_capon = capon_direction(cov, alpha, loading=loading)
    n = len(alpha)

    overlay = 1.0 + beta * strength * (w_capon - (1.0 / n))
    overlay = overlay.clip(lower=0.0)

    return overlay / overlay.sum()

