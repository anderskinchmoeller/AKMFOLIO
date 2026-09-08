from dataclasses import dataclass

@dataclass(frozen=True)
class HRPConfig:
    returns_file: str = "weekly_returns.csv"
    lookback_years: int = 5
    lookback_weeks: int = 260
    max_weight: float = 0.1
    min_weight: float = 0.01
    risk_free_rate: float = 0.00
    min_history_weeks: int = 52
    min_window_obs: int = 26
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
    cpcv_group_weeks: int = 52
    cpcv_purge_weeks: int = 1
    cpcv_embargo_weeks: int = 1
    # The HRP allocator is estimated walk-forward and has no trainable global
    # state, so one chronological run can be scored across all CPCV paths.
    # Enable only for an allocator_factory that genuinely fits on train rows.
    cpcv_refit_per_split: bool = False

    lw_alpha: float = 0.05
    lw_n_bootstraps: int = 4999
    lw_block_size: int = 6
