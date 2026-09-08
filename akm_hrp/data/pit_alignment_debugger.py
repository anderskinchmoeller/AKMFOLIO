from __future__ import annotations

import numpy as np
import pandas as pd


def debug_pit_alignment(returns: pd.DataFrame, pit_mask: pd.DataFrame) -> dict:
    """
    Diagnose PIT universe alignment issues:
      - date alignment
      - column alignment
      - forward/backward shifts
      - inverted eligibility (0/1 swapped)
      - weekly vs monthly mismatch
      - missing early PIT history
      - PIT collapse windows
    """

    diag = {}

    # ------------------------------------------------------------
    # 1. Basic shape checks
    # ------------------------------------------------------------
    diag["returns_shape"] = returns.shape
    diag["pit_shape"] = pit_mask.shape

    diag["same_dates"] = list(returns.index) == list(pit_mask.index)
    diag["same_assets"] = list(returns.columns) == list(pit_mask.columns)

    # ------------------------------------------------------------
    # 2. Check for inverted PIT mask (0 means eligible)
    # ------------------------------------------------------------
    # If PIT is inverted, the mean eligibility will be extremely low
    pit_mean = pit_mask.mean().mean()
    diag["pit_mean"] = float(pit_mean)

    if pit_mean < 0.05:
        diag["pit_inverted_suspected"] = True
    else:
        diag["pit_inverted_suspected"] = False

    # ------------------------------------------------------------
    # 3. Check for forward/backward shifts
    # ------------------------------------------------------------
    # Compare PIT eligibility with returns non-NaN availability
    shifts = {}
    for shift in range(-4, 5):  # ±4 weeks
        shifted = pit_mask.shift(shift)
        overlap = (shifted * returns.notna()).sum().sum()
        shifts[shift] = int(overlap)

    diag["pit_shift_overlap"] = shifts
    diag["best_shift"] = max(shifts, key=shifts.get)

    # ------------------------------------------------------------
    # 4. Check weekly vs monthly mismatch
    # ------------------------------------------------------------
    # Monthly PIT masks often have long runs of identical rows
    identical_runs = 0
    prev = pit_mask.iloc[0].values
    for row in pit_mask.iloc[1:].values:
        if np.array_equal(row, prev):
            identical_runs += 1
        prev = row

    diag["identical_row_runs"] = identical_runs
    diag["monthly_pit_suspected"] = identical_runs > len(pit_mask) * 0.5

    # ------------------------------------------------------------
    # 5. Check early PIT coverage collapse
    # ------------------------------------------------------------
    early_period = pit_mask.iloc[:52].sum(axis=1)
    diag["early_min_assets"] = int(early_period.min())
    diag["early_zero_weeks"] = early_period[early_period == 0].index.tolist()

    # ------------------------------------------------------------
    # 6. Check late PIT coverage collapse
    # ------------------------------------------------------------
    late_period = pit_mask.iloc[-52:].sum(axis=1)
    diag["late_min_assets"] = int(late_period.min())
    diag["late_zero_weeks"] = late_period[late_period == 0].index.tolist()

    # ------------------------------------------------------------
    # 7. Check asset-level PIT continuity
    # ------------------------------------------------------------
    continuity = {}
    for col in pit_mask.columns:
        series = pit_mask[col]
        transitions = (series.diff().abs() == 1).sum()
        continuity[col] = int(transitions)

    diag["asset_continuity"] = continuity

    return diag

