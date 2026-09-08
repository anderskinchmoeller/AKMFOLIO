from __future__ import annotations

import numpy as np
import pandas as pd


def validate_pit_universe(
    returns: pd.DataFrame,
    pit_mask: pd.DataFrame,
    min_required_assets: int = 5,
) -> dict:
    """
    Full PIT universe validator for HRP/CPCV/Rustuna pipelines.

    Returns a dictionary of diagnostics.
    """

    diagnostics = {}

    # ------------------------------------------------------------
    # 1. Basic alignment
    # ------------------------------------------------------------
    diagnostics["pit_index_matches_returns"] = (
        list(pit_mask.index) == list(returns.index)
    )
    diagnostics["pit_columns_match_returns"] = (
        list(pit_mask.columns) == list(returns.columns)
    )

    # ------------------------------------------------------------
    # 2. PIT coverage per asset
    # ------------------------------------------------------------
    pit_active_counts = pit_mask.sum(axis=0)
    zero_assets = pit_active_counts[pit_active_counts == 0].index.tolist()
    sparse_assets = pit_active_counts[pit_active_counts < 10].index.tolist()

    diagnostics["zero_eligibility_assets"] = zero_assets
    diagnostics["sparse_eligibility_assets"] = sparse_assets

    # ------------------------------------------------------------
    # 3. PIT coverage per week
    # ------------------------------------------------------------
    pit_active_per_week = pit_mask.sum(axis=1)
    diagnostics["min_assets_per_week"] = int(pit_active_per_week.min())
    diagnostics["max_assets_per_week"] = int(pit_active_per_week.max())
    diagnostics["weeks_with_zero_assets"] = pit_active_per_week[
        pit_active_per_week == 0
    ].index.tolist()

    # ------------------------------------------------------------
    # 4. PIT drift detection
    # ------------------------------------------------------------
    drift_points = []
    prev = pit_active_per_week.iloc[0]
    for date, count in pit_active_per_week.items():
        if abs(count - prev) > 20:  # large sudden jump
            drift_points.append((date, int(prev), int(count)))
        prev = count

    diagnostics["pit_drift_points"] = drift_points

    # ------------------------------------------------------------
    # 5. PIT gaps (assets disappearing for long periods)
    # ------------------------------------------------------------
    pit_gaps = {}
    for col in pit_mask.columns:
        series = pit_mask[col]
        zero_runs = []
        current_run = 0

        for val in series:
            if val == 0:
                current_run += 1
            else:
                if current_run >= 20:  # long gap
                    zero_runs.append(current_run)
                current_run = 0

        if current_run >= 20:
            zero_runs.append(current_run)

        if zero_runs:
            pit_gaps[col] = zero_runs

    diagnostics["pit_gaps"] = pit_gaps

    # ------------------------------------------------------------
    # 6. Universe collapse detection
    # ------------------------------------------------------------
    diagnostics["universe_collapsed"] = (
        diagnostics["min_assets_per_week"] < min_required_assets
    )

    return diagnostics

