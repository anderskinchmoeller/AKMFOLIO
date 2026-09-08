import numpy as np
import pandas as pd


def compute_drawdowns(returns: pd.Series) -> pd.Series:
    """
    Compute drawdown series from cumulative returns.
    """
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd


def compute_risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """
    Marginal and total risk contributions.
    """
    w = weights.values
    C = cov.loc[weights.index, weights.index].values
    port_var = float(w @ C @ w)

    mrc = (C @ w) / np.sqrt(port_var)
    trc = w * mrc
    return pd.Series(trc / trc.sum(), index=weights.index)


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    L1 turnover per period.
    """
    return weights.diff().abs().sum(axis=1)


def compute_stability(weights: pd.DataFrame) -> float:
    """
    Weight stability: average cosine similarity between consecutive weight vectors.
    """
    sims = []
    for t in range(1, len(weights)):
        w_prev = weights.iloc[t - 1].values
        w_curr = weights.iloc[t].values
        if np.linalg.norm(w_prev) == 0 or np.linalg.norm(w_curr) == 0:
            sims.append(0.0)
        else:
            sims.append(float(np.dot(w_prev, w_curr) /
                              (np.linalg.norm(w_prev) * np.linalg.norm(w_curr))))
    return float(np.mean(sims)) if sims else 0.0

