from __future__ import annotations

import numpy as np
import pandas as pd


def check_returns_sanity(
    returns: pd.DataFrame,
    min_history_weeks: int,
    pit_mask: pd.DataFrame | None = None,
) -> dict:
    """
    Comprehensive returns sanity checker for HRP/CPCV/Rustuna pipelines.

    Returns a dictionary of diagnostics you can print or log.
    """

    diagnostics = {}

    # ------------------------------------------------------------
    # 1. Basic shape
    # ------------------------------------------------------------
    diagnostics["n_weeks"] = len(returns)
    diagnostics["n_assets"] = returns.shape[1]

    # ------------------------------------------------------------
    # 2. Duplicate dates
    # ------------------------------------------------------------
    diagnostics["has_duplicate_dates"] = returns.index.has_duplicates

    # ------------------------------------------------------------
    # 3. All-zero or constant assets
    # ------------------------------------------------------------
    constant_assets = []
    for col in returns.columns:
        series = returns[col].dropna()
        if series.empty:
            continue
        if series.std() == 0:
            constant_assets.append(col)

    diagnostics["constant_assets"] = constant_assets

    # ------------------------------------------------------------
    # 4. Assets with too few observations
    # ------------------------------------------------------------
    insufficient_history = [
        col for col in returns.columns
        if returns[col].count() < min_history_weeks
    ]
    diagnostics["insufficient_history"] = insufficient_history

    # ------------------------------------------------------------
    # 5. Assets with all NaN
    # ------------------------------------------------------------
    all_nan_assets = [
        col for col in returns.columns
        if returns[col].isna().all()
    ]
    diagnostics["all_nan_assets"] = all_nan_assets

    # ------------------------------------------------------------
    # 6. Extreme outliers (weekly returns > ±50%)
    # ------------------------------------------------------------
    outlier_assets = []
    for col in returns.columns:
        if (returns[col].abs() > 0.50).any():
            outlier_assets.append(col)

    diagnostics["outlier_assets"] = outlier_assets

    # ------------------------------------------------------------
    # 7. Unrealistic volatility (std < 0.0001 or > 0.20 weekly)
    # ------------------------------------------------------------
    low_vol = []
    high_vol = []
    for col in returns.columns:
        series = returns[col].dropna()
        if series.empty:
            continue
        vol = series.std()
        if vol < 0.0001:
            low_vol.append(col)
        if vol > 0.20:
            high_vol.append(col)

    diagnostics["low_vol_assets"] = low_vol
    diagnostics["high_vol_assets"] = high_vol

    # ------------------------------------------------------------
    # 8. PIT universe filtering
    # ------------------------------------------------------------
    if pit_mask is not None:
        pit_active_counts = pit_mask.sum(axis=0)
        zero_pit_assets = [
            col for col in returns.columns
            if pit_active_counts.get(col, 0) == 0
        ]
        diagnostics["pit_zero_assets"] = zero_pit_assets

    # ------------------------------------------------------------
    # 9. Summary flags
    # ------------------------------------------------------------
    diagnostics["is_clean"] = (
        not diagnostics["has_duplicate_dates"]
        and len(constant_assets) == 0
        and len(insufficient_history) == 0
        and len(all_nan_assets) == 0
    )

    return diagnostics

