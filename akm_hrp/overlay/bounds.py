import numpy as np
import pandas as pd


def apply_bounds(
    weights: pd.Series,
    min_weight: float,
    max_weight: float,
    asset_caps: dict[str, float] | None = None,
) -> pd.Series:
    """Project weights onto the simplex while respecting lower/upper bounds.

    Guarantees sum(weights) == 1 (within numerical tolerance) and preserves
    all global / asset-specific bounds, or raises ValueError if infeasible.
    """
    if weights.empty:
        return weights.copy()
    if min_weight < 0 or max_weight < 0 or min_weight > max_weight:
        raise ValueError("Invalid global weight bounds.")

    idx = weights.index
    lower = pd.Series(float(min_weight), index=idx)
    upper = pd.Series(float(max_weight), index=idx)

    if asset_caps:
        for asset, cap in asset_caps.items():
            if asset in upper.index:
                if cap < 0:
                    raise ValueError(f"Negative cap for {asset}: {cap}")
                upper.loc[asset] = min(upper.loc[asset], float(cap))

    if (lower > upper).any():
        bad = lower.index[lower > upper].tolist()
        raise ValueError(f"Lower bound exceeds upper bound for: {bad}")

    lo_sum = float(lower.sum())
    hi_sum = float(upper.sum())
    tol = 1e-12
    if lo_sum > 1.0 + tol or hi_sum < 1.0 - tol:
        raise ValueError(
            f"Infeasible bounds: sum(lower)={lo_sum:.6f}, sum(upper)={hi_sum:.6f}."
        )

    v = weights.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    lo = lower.to_numpy()
    hi = upper.to_numpy()

    # Euclidean projection: x_i = clip(v_i - lambda, lo_i, hi_i).
    lam_low = float(np.min(v - hi)) - 1.0
    lam_high = float(np.max(v - lo)) + 1.0
    for _ in range(200):
        lam = 0.5 * (lam_low + lam_high)
        x = np.clip(v - lam, lo, hi)
        total = float(x.sum())
        if abs(total - 1.0) <= tol:
            break
        if total > 1.0:
            lam_low = lam
        else:
            lam_high = lam

    x = np.clip(v - 0.5 * (lam_low + lam_high), lo, hi)
    residual = 1.0 - float(x.sum())
    if abs(residual) > 1e-10:
        # Numerical cleanup without violating bounds.
        room = (hi - x) if residual > 0 else (x - lo)
        candidates = np.flatnonzero(room > tol)
        for i in candidates:
            step = min(abs(residual), room[i])
            x[i] += step if residual > 0 else -step
            residual += -step if residual > 0 else step
            if abs(residual) <= 1e-10:
                break

    result = pd.Series(x, index=idx)
    if not np.isclose(result.sum(), 1.0, atol=1e-9):
        raise RuntimeError("Bound projection failed to sum to one.")
    return result
