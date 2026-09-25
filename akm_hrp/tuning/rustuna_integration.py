from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import os
from typing import Any, Callable

import numpy as np
import pandas as pd
import rustuna
import optuna  # <-- Added

from akm_hrp.allocators.hrp_overlay_allocator import (
    AllocatorConfig,
    HRPOverlayAllocator,
)
from akm_hrp.backtest.cpcv import CPCVConfig
from akm_hrp.backtest.engine import run_cpcv_backtest
from akm_hrp.config import HRPConfig


@dataclass(frozen=True)
class RustunaTuningConfig:
    n_trials: int = 40
    baseline_n_trials: int = 30
    storage_file: str = "rustuna_studies.db"
    overlay_study_name: str = "signal_tilted_hrp_overlay_v4_0"
    baseline_study_name: str = "pure_hrp_baseline_v4_0"
    random_seed: int = 42
    turnover_penalty: float = 0.005

    # 0 or negative => use all available logical CPUs, capped by trial count.
    n_jobs: int = 0


ProgressCallback = Callable[
    [str, int, int, float | None, float | None],
    None,
]


class _BacktestConfigProxy:
    """
    Bridge flattened HRPConfig fields to engine.py, which expects cfg.cpcv.
    """

    def __init__(self, base: HRPConfig, cpcv: CPCVConfig):
        self._base = base
        self.cpcv = cpcv

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


# ---------------------------------------------------------------------------
# Worker-process state
# ---------------------------------------------------------------------------

_WORKER_RETURNS: pd.DataFrame | None = None
_WORKER_BT_CFG: HRPConfig | None = None
_WORKER_CPCV_CFG: CPCVConfig | None = None
_WORKER_TURNOVER_PENALTY: float = 0.005


def _worker_init(
    returns: pd.DataFrame,
    bt_cfg: HRPConfig,
    cpcv_cfg: CPCVConfig,
    turnover_penalty: float,
) -> None:
    global _WORKER_RETURNS
    global _WORKER_BT_CFG
    global _WORKER_CPCV_CFG
    global _WORKER_TURNOVER_PENALTY

    _WORKER_RETURNS = returns
    _WORKER_BT_CFG = bt_cfg
    _WORKER_CPCV_CFG = cpcv_cfg
    _WORKER_TURNOVER_PENALTY = float(turnover_penalty)


def _finite_mean(values: list[float], default: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return float(default)

    return float(x.mean())


def _evaluate_allocator_config(
    returns: pd.DataFrame,
    bt_cfg: HRPConfig,
    cpcv_cfg: CPCVConfig,
    alloc_cfg: AllocatorConfig,
    turnover_penalty: float,
    optuna_trial: optuna.Trial | None = None,  # <-- Added
    completed: int | None = None,              # <-- Added
    stage_total: int | None = None,            # <-- Added
) -> float:
    """
    Pure expensive evaluation step. No Rustuna storage access occurs here.
    """

    def allocator_factory(train_returns: pd.DataFrame):
        _ = train_returns
        return HRPOverlayAllocator(alloc_cfg)

    cfg = _BacktestConfigProxy(
        base=bt_cfg,
        cpcv=cpcv_cfg,
    )

    results = run_cpcv_backtest(
        returns,
        allocator_factory,
        cfg,
    )

    if not results:
        return -1e9

    sharpes: list[float] = []
    turnovers: list[float] = []
    drawdowns: list[float] = []

    for result in results.values():
        sharpe = result.metrics.get("sharpe", np.nan)
        ann_turnover = result.metrics.get("ann_turnover_l1", np.nan)
        max_drawdown = result.metrics.get("max_drawdown", np.nan)

        if np.isfinite(sharpe):
            sharpes.append(float(sharpe))

        if np.isfinite(ann_turnover):
            turnovers.append(float(ann_turnover))

        if np.isfinite(max_drawdown):
            drawdowns.append(abs(float(max_drawdown)))

    if not sharpes:
        return -1e9

    sharpe_array = np.asarray(sharpes, dtype=float)
    median_sharpe = float(np.median(sharpe_array))
    lower_quartile_sharpe = float(np.quantile(sharpe_array, 0.25))
    avg_turnover = _finite_mean(turnovers, default=0.0)
    avg_drawdown = _finite_mean(drawdowns, default=0.0)

    # Reward the typical path, then charge for instability in the lower tail.
    # This avoids selecting a configuration because a few CPCV paths were
    # spectacular while the rest were fragile.
    path_instability = max(0.0, median_sharpe - lower_quartile_sharpe)

    score = (
        median_sharpe
        - 0.35 * path_instability
        - float(turnover_penalty) * avg_turnover
        - 0.10 * avg_drawdown
    )

    # -----------------------------
    # Optuna pruning integration
    # -----------------------------
    if optuna_trial is not None and completed is not None:
        optuna_trial.report(score, step=completed)

        if optuna_trial.should_prune():
            raise optuna.TrialPruned()

    return float(score) if np.isfinite(score) else -1e9


def _worker_evaluate(alloc_cfg: AllocatorConfig) -> float:
    if (
        _WORKER_RETURNS is None
        or _WORKER_BT_CFG is None
        or _WORKER_CPCV_CFG is None
    ):
        raise RuntimeError("Rustuna worker was not initialized.")

    return _evaluate_allocator_config(
        returns=_WORKER_RETURNS,
        bt_cfg=_WORKER_BT_CFG,
        cpcv_cfg=_WORKER_CPCV_CFG,
        alloc_cfg=alloc_cfg,
        turnover_penalty=_WORKER_TURNOVER_PENALTY,
    )


# def _study_best_value(study) -> float | None:
#     try:
#         value = getattr(study, "best_value", None)
#         if value is not None:
#             return float(value)
#     except Exception:
#         pass
#
#     try:
#         trial = study.best_trial
#         value = getattr(trial, "value", None)
#         if value is not None:
#             return float(value)
#     except Exception:
#         pass
#
#     return None
#

def _study_best_value(study) -> float | None:
    try:
        value = getattr(study, "best_value", None)
        if value is not None:
            return float(value)
    except Exception:
        pass

    try:
        trial = study.best_trial

        # Standard Optuna
        value = getattr(trial, "value", None)
        if value is not None:
            return float(value)

        # Rustuna
        values = getattr(trial, "values", None)
        if values is not None and len(values) > 0 and values[0] is not None:
            return float(values[0])
    except Exception:
        pass

    return None

def _trial_number(study, trial) -> int:
    number = getattr(trial, "number", None)

    if number is not None:
        return int(number)

    trials = study.trials

    if not trials:
        raise RuntimeError("Rustuna ask() created no visible trial.")

    return int(trials[-1].number)


def _safe_min_weight_upper_bound(
    returns: pd.DataFrame,
    requested_upper: float = 0.01,
) -> float:
    n_assets = max(int(returns.shape[1]), 1)
    return max(
        0.0,
        min(float(requested_upper), 0.95 / n_assets),
    )


def _suggest_allocator_config(
    study,
    returns: pd.DataFrame,
    bt_cfg: HRPConfig,
    *,
    is_overlay: bool,
) -> tuple[int, AllocatorConfig]:
    trial = study.ask()
    number = _trial_number(study, trial)

    min_weight_upper = _safe_min_weight_upper_bound(returns)

    if min_weight_upper > 0.0:
        min_weight = trial.suggest_float(
            "min_weight",
            0.0,
            min_weight_upper,
        )
    else:
        min_weight = 0.0

    if is_overlay:
        budget_beta = trial.suggest_float("budget_beta", 0.10, 0.30)
        budget_tau = trial.suggest_float("budget_tau", 0.50, 1.00)
        beamforming_beta = trial.suggest_float("beamforming_beta", 0.10, 0.30)
        beamforming_loading = trial.suggest_float("beamforming_loading", 0.05, 0.20)
        beamforming_strength = trial.suggest_float("beamforming_strength", 0.25, 0.75)
    else:
        budget_beta = 0.20
        budget_tau = 0.75
        beamforming_beta = 0.20
        beamforming_loading = 0.10
        beamforming_strength = 0.50

    alloc_cfg = AllocatorConfig(
        min_weight=min_weight,
        max_weight=bt_cfg.max_weight,
        use_signal_budget=is_overlay,
        budget_beta=budget_beta,
        budget_tau=budget_tau,
        use_cov_inverse=is_overlay,
        beamforming_beta=beamforming_beta,
        beamforming_loading=beamforming_loading,
        beamforming_strength=beamforming_strength,
    )

    return number, alloc_cfg


def _resolved_jobs(requested: int, n_trials: int) -> int:
    if n_trials <= 0:
        return 1

    if requested <= 0:
        requested = os.cpu_count() or 1

    return max(
        1,
        min(int(requested), int(n_trials)),
    )


def _run_serial_stage(
    study,
    returns: pd.DataFrame,
    bt_cfg: HRPConfig,
    cpcv_cfg: CPCVConfig,
    tuning_cfg: RustunaTuningConfig,
    *,
    n_trials: int,
    stage: str,
    is_overlay: bool,
    progress_callback: ProgressCallback | None,
    optuna_trial: optuna.Trial | None = None,  # <-- Added
) -> None:
    for completed in range(1, n_trials + 1):
        number, alloc_cfg = _suggest_allocator_config(
            study,
            returns,
            bt_cfg,
            is_overlay=is_overlay,
        )

        try:
            value = _evaluate_allocator_config(
                returns=returns,
                bt_cfg=bt_cfg,
                cpcv_cfg=cpcv_cfg,
                alloc_cfg=alloc_cfg,
                turnover_penalty=tuning_cfg.turnover_penalty,
                optuna_trial=optuna_trial,   # <-- Added
                completed=completed,         # <-- Added
                stage_total=n_trials,        # <-- Added
            )
        except optuna.TrialPruned:
            study.tell(number, values=None)
            raise
        except Exception:
            study.tell(number, values=None)

            if progress_callback is not None:
                progress_callback(
                    stage,
                    completed,
                    n_trials,
                    None,
                    _study_best_value(study),
                )
            raise

        study.tell(number, values=value)

        if progress_callback is not None:
            progress_callback(
                stage,
                completed,
                n_trials,
                value,
                _study_best_value(study),
            )


def _run_parallel_stage(
    study,
    returns: pd.DataFrame,
    bt_cfg: HRPConfig,
    cpcv_cfg: CPCVConfig,
    tuning_cfg: RustunaTuningConfig,
    *,
    n_trials: int,
    stage: str,
    is_overlay: bool,
    progress_callback: ProgressCallback | None,
    optuna_trial: optuna.Trial | None = None,  # <-- Added
) -> None:
    workers = _resolved_jobs(
        tuning_cfg.n_jobs,
        n_trials,
    )

    if workers <= 1:
        _run_serial_stage(
            study,
            returns,
            bt_cfg,
            cpcv_cfg,
            tuning_cfg,
            n_trials=n_trials,
            stage=stage,
            is_overlay=is_overlay,
            progress_callback=progress_callback,
            optuna_trial=optuna_trial,  # <-- Added
        )
        return

    submitted = 0
    completed = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(
            returns,
            bt_cfg,
            cpcv_cfg,
            tuning_cfg.turnover_penalty,
        ),
    ) as executor:
        pending = {}

        def submit_one() -> None:
            nonlocal submitted

            number, alloc_cfg = _suggest_allocator_config(
                study,
                returns,
                bt_cfg,
                is_overlay=is_overlay,
            )

            future = executor.submit(
                _worker_evaluate,
                alloc_cfg,
            )
            pending[future] = number
            submitted += 1

        while submitted < min(workers, n_trials):
            submit_one()

        while pending:
            done, _ = wait(
                pending,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                number = pending.pop(future)
                completed += 1

                try:
                    value = float(future.result())
                except Exception:
                    study.tell(number, values=None)

                    if progress_callback is not None:
                        progress_callback(
                            stage,
                            completed,
                            n_trials,
                            None,
                            _study_best_value(study),
                        )

                    for other in pending:
                        other.cancel()

                    raise

                # Optuna pruning inside parallel stage
                if optuna_trial is not None:
                    optuna_trial.report(value, step=completed)
                    if optuna_trial.should_prune():
                        study.tell(number, values=None)
                        for other in pending:
                            other.cancel()
                        raise optuna.TrialPruned()

                study.tell(number, values=value)

                if progress_callback is not None:
                    progress_callback(
                        stage,
                        completed,
                        n_trials,
                        value,
                        _study_best_value(study),
                    )

                if submitted < n_trials:
                    submit_one()


def run_rustuna_tuning(
    returns: pd.DataFrame,
    bt_cfg: HRPConfig,
    cpcv_cfg: CPCVConfig,
    tuning_cfg: RustunaTuningConfig,
    progress_callback: ProgressCallback | None = None,
    optuna_trial: optuna.Trial | None = None,          # <-- Added
    optuna_param_callback: Callable | None = None,      # <-- Added
):
    """
    Run parallel Rustuna tuning for:
      1. signal/covariance-overlay HRP
      2. pure-HRP baseline
    """

    if returns.empty:
        raise ValueError("returns is empty.")

    # Apply Optuna parameters to HRPConfig
    if optuna_param_callback is not None and optuna_trial is not None:
        bt_cfg = optuna_param_callback(optuna_trial)

    storage = rustuna.storages.SQLite3Storage(
        tuning_cfg.storage_file,
        create_database=True,
    )

    overlay_study = rustuna.create_study(
        study_name=tuning_cfg.overlay_study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=rustuna.samplers.TPESampler(
            seed=tuning_cfg.random_seed,
        ),
    )

    _run_parallel_stage(
        overlay_study,
        returns,
        bt_cfg,
        cpcv_cfg,
        tuning_cfg,
        n_trials=int(tuning_cfg.n_trials),
        stage="overlay",
        is_overlay=True,
        progress_callback=progress_callback,
        optuna_trial=optuna_trial,  # <-- Added
    )

    baseline_study = rustuna.create_study(
        study_name=tuning_cfg.baseline_study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=rustuna.samplers.TPESampler(
            seed=tuning_cfg.random_seed + 1,
        ),
    )

    _run_parallel_stage(
        baseline_study,
        returns,
        bt_cfg,
        cpcv_cfg,
        tuning_cfg,
        n_trials=int(tuning_cfg.baseline_n_trials),
        stage="baseline",
        is_overlay=False,
        progress_callback=progress_callback,
        optuna_trial=optuna_trial,  # <-- Added
    )

    return overlay_study, baseline_study
