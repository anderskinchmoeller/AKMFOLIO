from __future__ import annotations

import argparse
from dataclasses import replace
import inspect
import pandas as pd

from tqdm.auto import tqdm
from akm_hrp.data.pit_validator import validate_pit_universe
from akm_hrp.data.returns_sanity_checker import check_returns_sanity
from akm_hrp.data.pit_alignment_debugger import debug_pit_alignment
from akm_hrp.backtest.cpcv import CPCVConfig
from akm_hrp.config import HRPConfig
from akm_hrp.data.pit_universe import load_pit_universe_mask, pit_fingerprint
from akm_hrp.data.returns import load_and_clean_returns
from akm_hrp.tuning.rustuna_integration import (
    RustunaTuningConfig,
    run_rustuna_tuning,
)
from akm_hrp.tuning.rustuna_dashboard import RustunaRealtimeDashboard


def _build_cpcv_config(cfg: HRPConfig) -> CPCVConfig:
    """
    Build CPCVConfig from the flattened CPCV fields in HRPConfig.

    This tolerates CPCVConfig versions using either `test_groups`
    or `n_test_groups`.
    """
    values = {
        "n_groups": cfg.cpcv_n_groups,
        "test_groups": cfg.cpcv_test_groups,
        "n_test_groups": cfg.cpcv_test_groups,
        "group_weeks": cfg.cpcv_group_weeks,
        "purge_weeks": cfg.cpcv_purge_weeks,
        "embargo_weeks": cfg.cpcv_embargo_weeks,
    }

    sig = inspect.signature(CPCVConfig)
    kwargs = {}
    missing_required = []

    for name, param in sig.parameters.items():
        if name in values:
            kwargs[name] = values[name]
        elif (
            param.default is inspect.Parameter.empty
            and param.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ):
            missing_required.append(name)

    if missing_required:
        raise RuntimeError(
            "Cannot construct CPCVConfig from HRPConfig. "
            f"Unsupported required fields: {missing_required}. "
            f"CPCVConfig signature: {sig}"
        )

    return CPCVConfig(**kwargs)


def _best_value(study):
    value = getattr(study, "best_value", None)
    if value is not None:
        return value

    trial = getattr(study, "best_trial", None)
    if trial is not None:
        return getattr(trial, "value", None)

    return None


def _best_params(study):
    params = getattr(study, "best_params", None)
    if params is not None:
        return params

    trial = getattr(study, "best_trial", None)
    if trial is not None:
        return getattr(trial, "params", {})

    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPCV/Rustuna research for the HRP model."
    )

    parser.add_argument(
        "--returns",
        default="weekly_returns.csv",
        help="Wide weekly returns CSV.",
    )
    parser.add_argument(
        "--pit",
        default=None,
        help="Point-in-time universe CSV.",
    )
    parser.add_argument(
        "--storage",
        default="rustuna_studies.db",
        help="Rustuna SQLite study database.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=40,
        help="Overlay Rustuna trials.",
    )
    parser.add_argument(
        "--baseline-trials",
        type=int,
        default=30,
        help="Pure-HRP baseline Rustuna trials.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="Hard information cutoff (YYYY-MM-DD). Rows after it are rejected.",
    )
    parser.add_argument(
        "--tc-bps",
        type=float,
        default=10.0,
        help="One-way transaction-cost assumption in basis points per L1 trade.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help=(
            "Parallel Rustuna worker processes. "
            "0 or negative uses all available logical CPUs."
        ),
    )
    parser.add_argument(
        "--overlay-study",
        default="signal_tilted_hrp_overlay_v4_0",
        help="Rustuna overlay study name.",
    )
    parser.add_argument(
        "--baseline-study",
        default="pure_hrp_baseline_v4_0",
        help="Rustuna baseline study name.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = HRPConfig()

    cfg = replace(
        cfg,
        returns_file=args.returns,
        pit_universe_file=args.pit,
        tc_bps=args.tc_bps,
    )

    returns = load_and_clean_returns(
        args.returns,
        min_history_weeks=cfg.min_history_weeks,
    )

    if args.as_of is not None:
        as_of = pd.Timestamp(args.as_of)
        returns = returns.loc[returns.index <= as_of]


    if returns.empty:
        raise RuntimeError("No usable returns remain after cleaning.")

    if returns.index.has_duplicates:
        raise RuntimeError("Returns file contains duplicate dates.")

    returns = returns.sort_index()

    pit_mask = None
    if args.pit:
        pit_mask = load_pit_universe_mask(
            args.pit,
            dates=returns.index,
            assets=list(returns.columns),
        )
        fingerprint = pit_fingerprint(pit_mask)
        print(f"PIT fingerprint: {fingerprint}")
    elif cfg.strict_pit_universe:
        raise RuntimeError(
            "strict_pit_universe=True but no --pit file was supplied."
        )
    if pit_mask is not None:
        pit_diag = validate_pit_universe(returns, pit_mask)
        print("\n=== PIT Universe Validation ===")
        for k, v in pit_diag.items():
            print(f"{k}: {v}")

    diagnostics = check_returns_sanity(
        returns,
        min_history_weeks=cfg.min_history_weeks,
        pit_mask=pit_mask if args.pit else None,
    )

    print("\n=== Returns Sanity Check ===")
    for k, v in diagnostics.items():
        print(f"{k}: {v}")

    if pit_mask is not None:
        pit_align = debug_pit_alignment(returns, pit_mask)

        print("\n=== PIT Universe Alignment Debugger ===")
        for k, v in pit_align.items():
            print(f"{k}: {v}")

    cpcv_cfg = _build_cpcv_config(cfg)

    tuning_cfg = RustunaTuningConfig(
        n_trials=args.trials,
        baseline_n_trials=args.baseline_trials,
        storage_file=args.storage,
        overlay_study_name=args.overlay_study,
        baseline_study_name=args.baseline_study,
        random_seed=args.seed,
        n_jobs=args.jobs,
    )

    print(
        f"Loaded {len(returns)} weeks x {returns.shape[1]} assets "
        f"from {returns.index.min().date()} to {returns.index.max().date()}."
    )
    print(f"CPCV config: {cpcv_cfg}")
    print(f"Rustuna storage: {tuning_cfg.storage_file}")
    print(
        "Rustuna workers: "
        + ("auto" if tuning_cfg.n_jobs <= 0 else str(tuning_cfg.n_jobs))
    )

    total_trials = tuning_cfg.n_trials + tuning_cfg.baseline_n_trials

    dashboard = RustunaRealtimeDashboard()
    dashboard.start(refresh_seconds=120.0)

    with tqdm(
        total=total_trials,
        desc="Rustuna",
        unit="trial",
        dynamic_ncols=True,
    ) as progress:

        def on_trial_progress(
            stage: str,
            completed: int,
            stage_total: int,
            value: float | None,
            best_value: float | None,
        ) -> None:
            # ETA
            if progress.n > 0:
                elapsed = progress.format_dict["elapsed"]
                rate = progress.n / elapsed if elapsed > 0 else 0
                remaining = (
                    (progress.total - progress.n) / rate if rate > 0 else 0
                )
                eta_str = f"{remaining/60:.1f} min"
            else:
                eta_str = "estimating..."

            postfix = {
                "stage": stage,
                "stage_progress": f"{completed}/{stage_total}",
                "eta": eta_str,
            }

            if value is not None:
                postfix["value"] = f"{value:.4f}"

            if best_value is not None:
                postfix["best"] = f"{best_value:.4f}"

            progress.set_postfix(postfix, refresh=True)
            # progress.update(1)

            dashboard.record(
                stage=stage,
                completed=completed,
                stage_total=stage_total,
                value=value,
                best_value=best_value,
            )

        overlay_study, baseline_study = run_rustuna_tuning(
            returns=returns,
            bt_cfg=cfg,
            cpcv_cfg=cpcv_cfg,
            tuning_cfg=tuning_cfg,
            progress_callback=on_trial_progress,
        )

    dashboard.stop()

    print("\n=== Overlay study ===")
    print("Best value:", _best_value(overlay_study))
    print("Best params:", _best_params(overlay_study))

    print("\n=== Pure HRP baseline ===")
    print("Best value:", _best_value(baseline_study))
    print("Best params:", _best_params(baseline_study))


if __name__ == "__main__":
    main()
