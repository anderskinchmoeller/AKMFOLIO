def optuna_search_space(trial):
    return {
        "lookback_weeks": trial.suggest_int("lookback_weeks", 104, 208),
        "max_weight": trial.suggest_float("max_weight", 0.06, 0.18),
        "min_history_weeks": trial.suggest_int("min_history_weeks", 52, 130),
        "cov_max_interior_missing_fraction": trial.suggest_float(
            "cov_max_interior_missing_fraction", 0.02, 0.10
        ),
        "lw_alpha": trial.suggest_float("lw_alpha", 0.04, 0.20),
        "lw_block_size": trial.suggest_int("lw_block_size", 4, 12),
        "cpcv_n_groups": trial.suggest_int("cpcv_n_groups", 6, 12),
        "cpcv_group_weeks": trial.suggest_int("cpcv_group_weeks", 26, 52),
        "risk_free_rate": trial.suggest_float("risk_free_rate", 0.005, 0.03),
    }

from dataclasses import replace
from akm_hrp.config import HRPConfig

def apply_optuna_params(cfg: HRPConfig, params: dict) -> HRPConfig:
    return replace(
        cfg,
        lookback_weeks=params["lookback_weeks"],
        max_weight=params["max_weight"],
        min_history_weeks=params["min_history_weeks"],
        cov_max_interior_missing_fraction=params["cov_max_interior_missing_fraction"],
        lw_alpha=params["lw_alpha"],
        lw_block_size=params["lw_block_size"],
        cpcv_n_groups=params["cpcv_n_groups"],
        cpcv_group_weeks=params["cpcv_group_weeks"],
        risk_free_rate=params["risk_free_rate"],
    )
