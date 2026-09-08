import numpy as np
import pandas as pd

def softmax(x: pd.Series, tau: float = 1.0) -> pd.Series:
    """
    Cross-sectional softmax allocation.
    """
    z = x - x.max()
    e = np.exp(z / tau)
    w = e / e.sum()
    return pd.Series(w, index=x.index)


def signal_budget_overlay(
    alpha: pd.Series,
    beta: float = 0.20,
    tau: float = 0.75,
) -> pd.Series:
    """
    Signal-budget overlay:
      w_signal = softmax(alpha, tau)
      overlay = 1 + beta * (w_signal - 1/N)
    """
    if alpha.empty:
        return alpha.astype(float)
    if alpha.isna().all():
        # Uniform multiplicative overlay is neutral after final renormalization.
        return pd.Series(1.0 / len(alpha), index=alpha.index)

    w_signal = softmax(alpha.fillna(0.0), tau=tau)
    n = len(alpha)

    overlay = 1.0 + beta * (w_signal - (1.0 / n))
    overlay = overlay.clip(lower=0.0)

    return overlay / overlay.sum()

