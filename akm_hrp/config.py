from dataclasses import dataclass

@dataclass(frozen=True)
class HRPConfig:
    returns_file: str = "weekly_returns.csv"
    lookback_years: int = 5
    lookback_weeks: int = 260
    max_weight: float = 0.2
    min_weight: float = 0.01
    risk_free_rate: float = 0.00
    min_history_weeks: int = 52
    # Require a full year of observations inside the estimation window.
    min_window_obs: int = 52
    cov_max_interior_missing_fraction: float = 0.05

    # Realistic implementation controls.  Turnover is L1 weight change, so a
    # value of 0.20 corresponds to roughly 10% one-way portfolio turnover.
    tc_bps: float = 10.0
    drift_threshold: float = 0.05
    min_holding_weeks: int = 12
    max_rebalance_turnover_l1: float | None = 0.10

    pit_universe_file: str | None = None
    strict_pit_universe: bool = False
    pit_require_current_observation: bool = False
    pit_require_historical_exits: bool = False

    test_holdout_weeks: int = 104
    cv_window_weeks: int = 260

    cpcv_n_groups: int = 6
    cpcv_test_groups: int = 2
    # Broad-history research preset: 30 years across 15 train/test splits.
    # Requires at least 1,560 weekly observations in the supplied CV data.
    cpcv_group_weeks: int = 104
    # Four-week boundary buffers are a sensitivity baseline, not a substitute
    # for rejecting labels whose formation/realization intervals cross splits.
    cpcv_purge_weeks: int = 4
    cpcv_embargo_weeks: int = 4
    # Auto restricts online learners to split training labels. False is only
    # valid for shared-path resampling of non-learning allocators.
    cpcv_refit_per_split: bool | None = None

    lw_alpha: float = 0.05
    lw_n_bootstraps: int = 4999
    lw_block_size: int = 6
