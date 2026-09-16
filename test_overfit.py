#!/usr/bin/env python3
"""
run_overfitting_diagnostics.py

Standalone anti-overfitting / anti-bias test battery for a registered
akm_hrp allocator. Run on its own, separately from compare_models.py --
it's deliberately not folded into that CLI because it's much more
expensive (CPCV + bootstrap + a placebo re-run each cost roughly as much as
a full walk-forward on their own) and because a failed check here should
never block a normal backtest run.

    python run_overfitting_diagnostics.py --model retail_alpha_ml_mpc_crowding \
        --evaluation-start 2015-01-02 --evaluation-end 2026-08-28

WHAT THIS RUNS (each check is independent -- one failing does not stop the
others; see `_run_check`):

  1. walk-forward-oos   Single-path OOS backtest. The baseline reference
                         point for every other check, NOT by itself evidence
                         against overfitting -- a single path can be lucky.
  2. kelly-buffer-check  Re-runs the same window with Kelly/crowding
                         disabled and diffs the two return series. Catches
                         the documented silent-fallback failure mode where
                         the Kelly buffer (52 weekly obs) never fills and
                         the "Kelly" run is quietly identical to baseline
                         (crowding_kappa_preregistration.md Sec 8 item 1;
                         kelly_factor_sizing_findings.md Sec 10).
  3. cpcv                Combinatorial purged CV: a *distribution* of OOS
                         Sharpes across many train/test splits, not one
                         path. Per retail_alpha_ml_mpc_phase3_notes.md this
                         has never actually been exercised against real data
                         in this repo -- treat a first run as exactly that.
  4. block-bootstrap     Calls akm_hrp.diagnostics.robustness.write_robustness_report
                         (block bootstrap, stratified block bootstrap,
                         CUSUM + vol-threshold regime detection, regime_hac,
                         cost-stress repricing, repair/relaxation frequency
                         -- all already shipped; this just runs it and folds
                         the results into the combined verdict).
  5. cost-stress         Read back out of the same robustness report,
                         surfaced separately with an explicit red-flag
                         threshold.
  6. repair-frequency    Flags if optimizer repair / CVaR relaxation fired
                         on an abnormally high fraction of rebalances --
                         i.e. the backtest may be "working" partly because
                         constraint violations are being patched over.
  7. placebo             Re-runs the SAME walk-forward with
                         ml_shuffle_labels=True. A model with real
                         predictive content should see performance collapse
                         toward noise; if it doesn't, the walk-forward
                         Sharpe is probably coming from pipeline/backtest
                         mechanics, not signal.
  8. significance        PSR / DSR-EO sensitivity table across K in
                         {4,10,50,200,1000}, matching the pre-registered
                         protocol in kelly_factor_sizing_preregistration.md
                         Sec 5. Tries the repo's own validated significance
                         module first; only falls back to a clearly-labeled
                         approximate implementation in this script if that
                         import fails.
  9. crowding-stress-window   Only if the crowding kappa overlay is
                         enabled: max-DD/vol in the highest-decile
                         crowding-shock weeks (crowding_kappa_preregistration.md
                         Sec 5's stress-window diagnostic).
 10. effective-breadth   Signal redundancy check (kelly_factor_sizing_findings.md
                         Sec 5) -- flags whether k in c* still reflects real
                         independent bets.
 11. data-caveats        Non-statistical: flags known, already-documented
                         data issues that bear on interpreting everything
                         above (2026 yfinance tail is a frozen, non-reranked
                         universe with no delisting; PIT file forward-filled
                         past 2025-12-26; Kelly/crowding windows are
                         calendar-week specified).

Every check is independently skippable (--skip-<name>) since a full battery
is a multi-hour run once CPCV and bootstrap are both on. Use --quick for a
fast structural smoke test of THIS SCRIPT (trimmed splits/bootstrap draws)
before committing to a real run.

Every numeric red-flag threshold below (repair frequency, CPCV degradation,
placebo leakage, cost-stress falloff) is a reasonable-but-arbitrary
judgment call baked into this script, NOT something derived the way
kappa/gamma/c* are in the Kelly docs. Treat FLAG as "worth a look," not as
a formal statistical test the way DSR-EO is.

=====================================================================
ADAPT BEFORE FIRST REAL RUN
=====================================================================
This script was written from project documentation, not from a live copy
of the akm_hrp package -- this session has no access to your actual
repo/venv. The import paths and function signatures below are the
best-supported reading of that documentation, but are the one part that
could not be executed against your real code. Two specific spots to check:

  - Significance module location: kelly_factor_sizing_findings.md Sec 8
    names `hrp/diagnostics/search_adjusted_inference.py`; everything else
    in the project lives under `akm_hrp/`. Both are tried (see
    SIGNIFICANCE_MODULE_CANDIDATES below) -- edit that list if neither
    matches your checkout.
  - Exact attribute names on the run_walk_forward / run_cpcv_backtest
    result objects. Only `.ending_weights` is independently confirmed in
    the docs (retail_alpha_mpc_latest_target_linear_infeasibility.md).
    Everything else in this script reads through `_extract_returns_series`
    and `_extract_diagnostics` -- edit THOSE two functions if your result
    object uses different attribute names; nothing else needs to change.

Tested in isolation against stub modules matching this documented
interface to confirm control flow, error isolation, and report generation
are correct -- NOT against your real package, which this session cannot
reach. Run --quick against a trimmed window first.
=====================================================================
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Adapter imports -- lazy/best-effort so a missing module only disables the
# checks that need it, not the whole script.
# --------------------------------------------------------------------------

def _try_import(module_name: str):
    try:
        module = __import__(module_name, fromlist=["_"])
        return module
    except ImportError:
        return None


ENGINE_MODULE = _try_import("akm_hrp.backtest.engine")
ROBUSTNESS_MODULE = _try_import("akm_hrp.diagnostics.robustness")
ALLOCATOR_MODULE = _try_import("akm_hrp.allocators.retail_alpha_ml_mpc")

SIGNIFICANCE_MODULE_CANDIDATES = [
    "akm_hrp.diagnostics.significance",
    "hrp.diagnostics.search_adjusted_inference",
]


# --------------------------------------------------------------------------
# Shared types
# --------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    status: str  # PASS, FLAG, FAIL, SKIPPED, ERROR
    headline: str
    detail: str = ""
    metrics: dict = field(default_factory=dict)
    artifacts: list = field(default_factory=list)


@dataclass
class Context:
    returns: pd.DataFrame
    pit: Optional[pd.DataFrame]
    config: Any
    allocator: Any
    evaluation_start: Optional[str]
    evaluation_end: Optional[str]
    data_start: Optional[str]
    rebalance_every_weeks: int
    model_name: str
    kelly_mix_enabled: bool
    kelly_scale_enabled: bool
    crowding_enabled: bool
    n_bootstrap: int
    block_length_weeks: int
    cpcv_splits: int
    cpcv_embargo_weeks: int
    placebo_seed: int
    output_dir: Path
    allow_approximate_significance: bool


# --------------------------------------------------------------------------
# Self-contained metrics (Sharpe/CAGR/drawdown/vol) -- deliberately NOT
# imported from the repo, since the formulas are standard and this avoids
# depending on yet another guessed internal module path.
# --------------------------------------------------------------------------

def _sharpe(r: pd.Series, periods_per_year: int = 52) -> float:
    r = r.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def _cagr(r: pd.Series, periods_per_year: int = 52) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    growth = float((1 + r).prod())
    years = len(r) / periods_per_year
    if years <= 0 or growth <= 0:
        return float("nan")
    return growth ** (1 / years) - 1


def _max_drawdown(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    wealth = (1 + r).cumprod()
    dd = wealth / wealth.cummax() - 1
    return float(dd.min())


def _annualized_vol(r: pd.Series, periods_per_year: int = 52) -> float:
    r = r.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


# --------------------------------------------------------------------------
# Approximate PSR / DSR-EO fallback -- ONLY used with
# --allow-approximate-significance, and only if the repo's own validated
# module can't be found. Standard Bailey & Lopez de Prado (2012) PSR,
# native per-period units. Do not compare directly to the annualized hurdle
# tables in kelly_factor_sizing_preregistration.md without matching
# frequency/annualization conventions -- this is a fallback for when the
# real, validated module isn't reachable, not a replacement for it.
# --------------------------------------------------------------------------

def _approx_psr(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    from scipy.stats import norm, skew, kurtosis

    x = r.dropna().to_numpy()
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return float("nan")
    sr_hat = x.mean() / x.std(ddof=1)
    g3 = float(skew(x))
    g4 = float(kurtosis(x, fisher=False))  # normal = 3
    denom = np.sqrt(max(1e-12, 1 - g3 * sr_hat + (g4 - 1) / 4 * sr_hat ** 2))
    z = (sr_hat - sr_benchmark) * np.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def _approx_dsr_eo(r: pd.Series, k: int, sr_benchmark: float = 0.0) -> float:
    p = _approx_psr(r, sr_benchmark=sr_benchmark)
    if np.isnan(p):
        return float("nan")
    return float(p ** k)


# --------------------------------------------------------------------------
# Result-object adapters -- the one place to edit if your engine's result
# objects expose different attribute names.
# --------------------------------------------------------------------------

def _extract_returns_series(result: Any) -> pd.Series:
    for attr in ("portfolio_returns", "returns", "realized_returns", "weekly_returns"):
        if hasattr(result, attr):
            series = getattr(result, attr)
            if isinstance(series, pd.Series):
                return series.dropna()
    if isinstance(result, pd.Series):
        return result.dropna()
    raise AttributeError(
        f"Could not find a returns series on result object of type {type(result)}. "
        f"Available attributes: {[a for a in dir(result) if not a.startswith('_')]}. "
        "Edit _extract_returns_series() to match your actual result object."
    )


def _extract_diagnostics(result: Any) -> Optional[pd.DataFrame]:
    for attr in ("diagnostics", "kelly_diagnostics", "last_kelly_diagnostics", "rebalance_diagnostics"):
        if hasattr(result, attr):
            d = getattr(result, attr)
            if isinstance(d, pd.DataFrame):
                return d
    return None


def _extract_cpcv_path_sharpes(result: Any) -> list:
    if hasattr(result, "path_sharpes"):
        return [float(s) for s in result.path_sharpes]
    for attr in ("path_returns", "paths", "fold_returns"):
        if hasattr(result, attr):
            paths = getattr(result, attr)
            out = []
            for p in paths:
                series = p if isinstance(p, pd.Series) else pd.Series(p)
                out.append(_sharpe(series))
            return out
    raise AttributeError(
        f"Could not find per-path returns/Sharpes on CPCV result ({type(result)}). "
        "Edit _extract_cpcv_path_sharpes() to match your actual result object."
    )


def _clone_config_with(config: Any, **overrides) -> Any:
    new_config = copy.deepcopy(config)
    for k, v in overrides.items():
        setattr(new_config, k, v)
    return new_config


def _build_config(args, crowding: bool):
    if ALLOCATOR_MODULE is None:
        raise ImportError("akm_hrp.allocators.retail_alpha_ml_mpc not importable in this environment.")
    Config = getattr(ALLOCATOR_MODULE, "RetailAlphaMLMPCConfig")
    kwargs = dict(
        ml_max_total_assets=args.max_total_assets,
        ml_kelly_mix_enabled=args.kelly_mix_enabled,
        ml_kelly_scale_enabled=args.kelly_scale_enabled,
        allow_cvar_floor_relaxation=args.allow_cvar_floor_relaxation,
    )
    if crowding:
        kwargs["ml_kelly_crowding_kappa_enabled"] = True
    return Config(**kwargs)


def _build_allocator(config: Any):
    Allocator = getattr(ALLOCATOR_MODULE, "RetailAlphaMLMPCAllocator")
    return Allocator(config=config)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_walk_forward_oos(ctx: Context, store: dict) -> CheckResult:
    if ENGINE_MODULE is None:
        raise ImportError("akm_hrp.backtest.engine not importable in this environment.")
    run_walk_forward = getattr(ENGINE_MODULE, "run_walk_forward")
    result = run_walk_forward(
        ctx.returns, ctx.allocator, ctx.config, pit=ctx.pit,
        rebalance_every_weeks=ctx.rebalance_every_weeks,
        evaluation_start=ctx.evaluation_start, evaluation_end=ctx.evaluation_end,
        data_start=ctx.data_start,
    )
    r = _extract_returns_series(result)
    store["walk_forward_result"] = result
    store["walk_forward_returns"] = r
    sharpe, cagr, dd, vol = _sharpe(r), _cagr(r), _max_drawdown(r), _annualized_vol(r)
    store["walk_forward_sharpe"] = sharpe
    diagnostics = _extract_diagnostics(result)
    if diagnostics is not None:
        store["walk_forward_diagnostics"] = diagnostics
    return CheckResult(
        name="walk-forward-oos", status="PASS",
        headline=f"Sharpe {sharpe:.2f}, CAGR {cagr:.1%}, max DD {dd:.1%}, vol {vol:.1%}, {len(r)} rebalances",
        detail="Single-path OOS result -- the baseline reference point for every check below, "
               "NOT by itself evidence against overfitting (see cpcv).",
        metrics={"sharpe": sharpe, "cagr": cagr, "max_drawdown": dd, "annualized_vol": vol, "n_rebalances": len(r)},
    )


def check_kelly_buffer_filled(ctx: Context, store: dict) -> CheckResult:
    if not (ctx.kelly_mix_enabled or ctx.kelly_scale_enabled):
        return CheckResult(name="kelly-buffer-check", status="SKIPPED",
                            headline="Kelly sizing not enabled for this run")
    if "walk_forward_returns" not in store:
        return CheckResult(name="kelly-buffer-check", status="SKIPPED",
                            headline="walk-forward-oos did not complete; nothing to compare against")
    baseline_config = _clone_config_with(
        ctx.config, ml_kelly_mix_enabled=False, ml_kelly_scale_enabled=False,
        ml_kelly_crowding_kappa_enabled=False,
    )
    baseline_allocator = _build_allocator(baseline_config)
    run_walk_forward = getattr(ENGINE_MODULE, "run_walk_forward")
    result = run_walk_forward(
        ctx.returns, baseline_allocator, baseline_config, pit=ctx.pit,
        rebalance_every_weeks=ctx.rebalance_every_weeks,
        evaluation_start=ctx.evaluation_start, evaluation_end=ctx.evaluation_end,
        data_start=ctx.data_start,
    )
    baseline_r = _extract_returns_series(result)
    kelly_r = store["walk_forward_returns"]
    a, b = baseline_r.align(kelly_r, join="inner")
    if len(a) > 0 and np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-12):
        return CheckResult(
            name="kelly-buffer-check", status="FAIL",
            headline="Kelly-enabled run is byte-identical to the no-Kelly baseline",
            detail="This is the exact silent-fallback failure mode documented in "
                   "crowding_kappa_preregistration.md Sec 8 item 1 and "
                   "kelly_factor_sizing_findings.md Sec 10: the Kelly buffer (needs 52 "
                   "weekly observations) never filled over this window. Most likely causes: "
                   "--rebalance-every-weeks is not 1 (the buffer and crowding window are "
                   "specified in calendar weeks, not rebalance calls), or --data-start "
                   "doesn't leave >= 156 weeks (104 warm-up + 52 buffer) before the "
                   "evaluation window starts.",
        )
    return CheckResult(
        name="kelly-buffer-check", status="PASS",
        headline="Kelly-enabled run differs from the no-Kelly baseline (buffer is doing something)",
    )


def check_cpcv(ctx: Context, store: dict) -> CheckResult:
    if ENGINE_MODULE is None:
        raise ImportError("akm_hrp.backtest.engine not importable in this environment.")
    run_cpcv_backtest = getattr(ENGINE_MODULE, "run_cpcv_backtest")
    result = run_cpcv_backtest(
        ctx.returns, ctx.allocator, ctx.config, pit=ctx.pit,
        n_splits=ctx.cpcv_splits, embargo_weeks=ctx.cpcv_embargo_weeks,
        rebalance_every_weeks=ctx.rebalance_every_weeks,
        evaluation_start=ctx.evaluation_start, evaluation_end=ctx.evaluation_end,
        data_start=ctx.data_start, cpcv_refit_per_split=True,
    )
    path_sharpes = np.array(_extract_cpcv_path_sharpes(result), dtype=float)
    path_sharpes = path_sharpes[~np.isnan(path_sharpes)]
    if len(path_sharpes) == 0:
        return CheckResult(name="cpcv", status="ERROR", headline="No valid CPCV paths returned")
    mean_s = float(np.mean(path_sharpes))
    std_s = float(np.std(path_sharpes, ddof=1)) if len(path_sharpes) > 1 else float("nan")
    frac_neg = float(np.mean(path_sharpes < 0))
    wf_sharpe = store.get("walk_forward_sharpe", float("nan"))

    status, notes = "PASS", []
    if not np.isnan(wf_sharpe) and wf_sharpe > 0 and mean_s < 0.5 * wf_sharpe:
        status = "FLAG"
        notes.append(
            f"CPCV mean path Sharpe ({mean_s:.3f}) is under half the single walk-forward "
            f"Sharpe ({wf_sharpe:.3f}) -- classic overfitting signature: the one path that "
            f"got reported may just have gotten lucky."
        )
    if frac_neg > 0.3:
        status = "FLAG"
        notes.append(f"{frac_neg:.0%} of CPCV paths have negative Sharpe.")
    return CheckResult(
        name="cpcv", status=status,
        headline=f"{len(path_sharpes)} paths, mean Sharpe {mean_s:.3f} (std {std_s:.3f})"
                 + (" -- " + "; ".join(notes) if notes else ""),
        detail="First real exercise of run_cpcv_backtest for this allocator per "
               "retail_alpha_ml_mpc_phase3_notes.md's known-untested-edges note -- "
               "treat a first run of this check as exactly that, not a routine re-check.",
        metrics={"mean_sharpe": mean_s, "std_sharpe": std_s, "frac_negative": frac_neg,
                 "n_paths": len(path_sharpes)},
    )


def check_block_bootstrap(ctx: Context, store: dict) -> CheckResult:
    if ROBUSTNESS_MODULE is None:
        raise ImportError("akm_hrp.diagnostics.robustness not importable in this environment.")
    write_robustness_report = getattr(ROBUSTNESS_MODULE, "write_robustness_report")
    report_dir = ctx.output_dir / "robustness"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = write_robustness_report(
        ctx.model_name, ctx.allocator, ctx.config, ctx.returns,
        pit=ctx.pit, output_dir=str(report_dir),
        n_bootstrap=ctx.n_bootstrap, block_length_weeks=ctx.block_length_weeks,
    )
    store["robustness_report"] = report if isinstance(report, dict) else {}
    ci = None
    if isinstance(report, dict):
        ci = report.get("bootstrap_sharpe_ci") or report.get("sharpe_confidence_interval")
    status = "PASS"
    headline = "Robustness report generated"
    if ci is not None:
        headline += f"; bootstrap Sharpe CI {ci}"
        if isinstance(ci, (list, tuple)) and len(ci) == 2 and ci[0] < 0 < ci[1]:
            status = "FLAG"
            headline += " -- CI straddles zero"
    return CheckResult(
        name="block-bootstrap", status=status, headline=headline,
        detail=f"Block bootstrap, stratified-block variant, CUSUM + vol-threshold regime "
               f"detection, and regime_hac contrasts all written to {report_dir}.",
        artifacts=[str(report_dir)],
    )


def check_cost_stress(ctx: Context, store: dict) -> CheckResult:
    report = store.get("robustness_report")
    if not report:
        return CheckResult(name="cost-stress", status="SKIPPED",
                            headline="No robustness report available (block-bootstrap must succeed first)")
    row = report.get("cost_stress") or report.get("cost_stress_repricing")
    if row is None:
        return CheckResult(name="cost-stress", status="SKIPPED",
                            headline="Robustness report has no cost_stress field under a recognized name")
    status, notes = "PASS", []
    try:
        s1 = row.get(1.0, row.get("1.0x", row.get("1x")))
        s2 = row.get(2.0, row.get("2.0x", row.get("2x")))
        if s1 is not None and s2 is not None and s1 > 0 and s2 < 0.3 * s1:
            status = "FLAG"
            notes.append(f"Sharpe at 2x costs ({s2:.2f}) is under 30% of 1x costs ({s1:.2f}) -- "
                         "edge may be thin relative to trading friction.")
    except Exception:
        pass
    return CheckResult(
        name="cost-stress", status=status,
        headline="Cost stress (1x/1.5x/2x) read from robustness report" + (": " + "; ".join(notes) if notes else ""),
        metrics=dict(row) if isinstance(row, dict) else {},
    )


def check_repair_frequency(ctx: Context, store: dict) -> CheckResult:
    report = store.get("robustness_report")
    if not report:
        return CheckResult(name="repair-frequency", status="SKIPPED",
                            headline="No robustness report available (block-bootstrap must succeed first)")
    repaired = report.get("mean_optimizer_repaired")
    relaxed = report.get("mean_risk_limit_relaxed")
    status, notes = "PASS", []
    if repaired is not None and repaired > 0.15:
        status = "FLAG"
        notes.append(f"optimizer repaired on {repaired:.0%} of rebalances")
    if relaxed is not None and relaxed > 0.10:
        status = "FLAG"
        notes.append(f"CVaR/risk limit relaxed on {relaxed:.0%} of rebalances")
    return CheckResult(
        name="repair-frequency", status=status,
        headline="; ".join(notes) if notes else f"repaired={repaired}, relaxed={relaxed}",
        detail="High repair/relaxation frequency means some backtest performance may come "
               "from constraints being patched over rather than genuinely satisfied -- related "
               "to the infeasibility corner in retail_alpha_mpc_latest_target_linear_infeasibility.md.",
        metrics={"mean_optimizer_repaired": repaired, "mean_risk_limit_relaxed": relaxed},
    )


def check_placebo(ctx: Context, store: dict) -> CheckResult:
    if "walk_forward_returns" not in store:
        return CheckResult(name="placebo", status="SKIPPED",
                            headline="walk-forward-oos did not complete; nothing to compare against")
    placebo_config = _clone_config_with(ctx.config, ml_shuffle_labels=True, ml_placebo_seed=ctx.placebo_seed)
    placebo_allocator = _build_allocator(placebo_config)
    run_walk_forward = getattr(ENGINE_MODULE, "run_walk_forward")
    result = run_walk_forward(
        ctx.returns, placebo_allocator, placebo_config, pit=ctx.pit,
        rebalance_every_weeks=ctx.rebalance_every_weeks,
        evaluation_start=ctx.evaluation_start, evaluation_end=ctx.evaluation_end,
        data_start=ctx.data_start,
    )
    placebo_r = _extract_returns_series(result)
    placebo_sharpe = _sharpe(placebo_r)
    real_sharpe = store["walk_forward_sharpe"]

    status = "PASS"
    detail = ("Labels shuffled via ml_shuffle_labels=True (the mechanism documented in "
              "shared_chat_ideas_implementation_2026-09-09.md). A model with genuine "
              "predictive content should see this Sharpe collapse toward zero.")
    if not (np.isnan(real_sharpe) or np.isnan(placebo_sharpe)):
        if abs(real_sharpe) > 0.2 and abs(placebo_sharpe) > 0.3 * abs(real_sharpe):
            status = "FLAG"
            detail += (f" Here it did not: placebo Sharpe {placebo_sharpe:.3f} is not small "
                       f"relative to real Sharpe {real_sharpe:.3f} -- some of the real result "
                       f"may be coming from backtest mechanics rather than signal.")
    return CheckResult(
        name="placebo", status=status,
        headline=f"real Sharpe {real_sharpe:.3f} vs shuffled-label Sharpe {placebo_sharpe:.3f}",
        detail=detail,
        metrics={"real_sharpe": real_sharpe, "placebo_sharpe": placebo_sharpe},
    )


def check_significance(ctx: Context, store: dict, trial_counts) -> CheckResult:
    if "walk_forward_returns" not in store:
        return CheckResult(name="significance", status="SKIPPED",
                            headline="walk-forward-oos did not complete; nothing to test")
    r = store["walk_forward_returns"]

    module = None
    used_path = None
    for candidate in SIGNIFICANCE_MODULE_CANDIDATES:
        module = _try_import(candidate)
        if module is not None:
            used_path = candidate
            break

    psr_fn = dsr_fn = None
    source = None
    if module is not None:
        psr_fn = getattr(module, "psr", None) or getattr(module, "probabilistic_sharpe_ratio", None)
        dsr_fn = getattr(module, "dsr_eo", None) or getattr(module, "deflated_sharpe_ratio_eo", None)
        if psr_fn and dsr_fn:
            source = f"repo module ({used_path})"

    if source is None:
        if not ctx.allow_approximate_significance:
            return CheckResult(
                name="significance", status="SKIPPED",
                headline="Significance module not found; skipped (pass --allow-approximate-significance for a fallback)",
                detail="Tried: " + ", ".join(SIGNIFICANCE_MODULE_CANDIDATES) + ". "
                       "This is expected if your checkout uses a different path -- edit "
                       "SIGNIFICANCE_MODULE_CANDIDATES at the top of this script.",
            )
        psr_fn = lambda series, sr_benchmark=0.0: _approx_psr(series, sr_benchmark)
        dsr_fn = lambda series, K=1, sr_benchmark=0.0: _approx_dsr_eo(series, K, sr_benchmark)
        source = "APPROXIMATE fallback in this script -- NOT your repo's validated implementation"

    rows = {}
    for k in trial_counts:
        try:
            p = psr_fn(r, sr_benchmark=0.0)
            d = dsr_fn(r, K=k, sr_benchmark=0.0)
        except TypeError:
            # signature fallback in case the real module doesn't take these exact kwargs
            p = psr_fn(r)
            d = dsr_fn(r, k)
        rows[k] = {"psr": p, "dsr_eo": d}

    smallest_k = trial_counts[0]
    largest_k = trial_counts[-1]
    status = "PASS"
    detail_bits = [f"source: {source}."]
    d_small = rows[smallest_k]["dsr_eo"]
    d_large = rows[largest_k]["dsr_eo"]
    if d_large is not None and not np.isnan(d_large) and d_large < 0.5:
        status = "FLAG"
        detail_bits.append(
            f"DSR-EO at K={largest_k} is {d_large:.3f} -- below the honest-trial-count bar this "
            f"project's own pre-registrations use. Per kelly_factor_sizing_findings.md Sec 8, "
            f"small K is not credible for a codebase iterated across many sessions; K={largest_k} "
            f"is the more honest column, not K={smallest_k}'s {d_small if d_small is None else f'{d_small:.3f}'}."
        )
    return CheckResult(
        name="significance", status=status,
        headline=f"DSR-EO across K={trial_counts}: " + ", ".join(f"{k}:{rows[k]['dsr_eo']:.3f}" for k in trial_counts),
        detail=" ".join(detail_bits),
        metrics={f"dsr_eo_K{k}": rows[k]["dsr_eo"] for k in trial_counts},
    )


def check_crowding_stress_window(ctx: Context, store: dict) -> CheckResult:
    if not ctx.crowding_enabled:
        return CheckResult(name="crowding-stress-window", status="SKIPPED",
                            headline="Crowding kappa overlay not enabled for this run")
    diagnostics = store.get("walk_forward_diagnostics")
    if diagnostics is None or "crowding_z" not in getattr(diagnostics, "columns", []):
        return CheckResult(
            name="crowding-stress-window", status="SKIPPED",
            headline="No crowding_z column found on the walk-forward diagnostics",
            detail="Expected the per-rebalance diagnostics fields named in "
                   "crowding_kappa_preregistration.md Sec 7 (crowding_active, crowding_rho_bar, "
                   "crowding_z, crowding_kappa_multiplier, kappa_used). Edit _extract_diagnostics() "
                   "if your result object stores this under a different name/shape.",
        )
    r = store.get("walk_forward_returns")
    if r is None:
        return CheckResult(name="crowding-stress-window", status="SKIPPED",
                            headline="No walk-forward returns available to score")
    z = diagnostics["crowding_z"].abs()
    threshold = z.quantile(0.9)
    stress_dates = z.index[z >= threshold]
    stress_r = r.reindex(stress_dates).dropna()
    rest_r = r.drop(index=stress_dates, errors="ignore").dropna()
    stress_dd, rest_dd = _max_drawdown(stress_r), _max_drawdown(rest_r)
    stress_vol, rest_vol = _annualized_vol(stress_r), _annualized_vol(rest_r)

    ablation_note = ""
    if "kappa_used" in diagnostics.columns:
        kappa_series = diagnostics["kappa_used"]
        ablation_note = (
            f" kappa_used ranged {kappa_series.min():.3f}-{kappa_series.max():.3f} over the window; "
            f"the redundancy-with-gamma ablation from crowding_kappa_preregistration.md Sec 5 "
            f"(lagging kappa_t against tr(Omega_g)) needs an Omega_g trace this script doesn't "
            f"have a confirmed field name for -- add it to _extract_diagnostics() if your "
            f"diagnostics table carries it, to complete this ablation automatically."
        )
    return CheckResult(
        name="crowding-stress-window", status="PASS",
        headline=f"top-decile |z| weeks: max DD {stress_dd:.1%} vol {stress_vol:.1%} "
                 f"vs rest: max DD {rest_dd:.1%} vol {rest_vol:.1%}",
        detail="Per Sec 3's honest prior, a real effect should show up here (smaller stress-window "
               "drawdown) rather than in the headline Sharpe." + ablation_note,
        metrics={"stress_max_dd": stress_dd, "rest_max_dd": rest_dd,
                 "stress_vol": stress_vol, "rest_vol": rest_vol, "n_stress_weeks": len(stress_r)},
    )


def check_effective_breadth(ctx: Context, store: dict) -> CheckResult:
    if ALLOCATOR_MODULE is None:
        return CheckResult(name="effective-breadth", status="SKIPPED", headline="Allocator module not importable")
    fn = getattr(ALLOCATOR_MODULE, "effective_breadth", None)
    if fn is None:
        return CheckResult(name="effective-breadth", status="SKIPPED",
                            headline="effective_breadth() not found on the allocator module")
    diagnostics = store.get("walk_forward_diagnostics")
    factor_data = None
    if diagnostics is not None:
        for attr in ("factor_returns", "factor_return_panel"):
            if hasattr(diagnostics, attr):
                factor_data = getattr(diagnostics, attr)
    if factor_data is None:
        return CheckResult(
            name="effective-breadth", status="SKIPPED",
            headline="No factor-return panel found on the walk-forward diagnostics to score",
            detail="effective_breadth() needs the k-factor return panel used for Kelly sizing; "
                   "this script doesn't have a confirmed attribute name for it on your result "
                   "object -- wire it in if you have one handy.",
        )
    breadth = fn(factor_data)
    return CheckResult(
        name="effective-breadth", status="PASS",
        headline=f"effective breadth {breadth:.2f}",
        metrics={"effective_breadth": breadth},
    )


def check_data_caveats(ctx: Context, store: dict) -> CheckResult:
    notes = []
    status = "PASS"
    cutover = pd.Timestamp("2026-01-02")
    end = pd.Timestamp(ctx.evaluation_end) if ctx.evaluation_end else None
    if end is None or end >= cutover:
        status = "FLAG"
        notes.append(
            "Evaluation window extends into 2026: those weekly returns are Yahoo Finance "
            "adjusted-close (not WRDS CRSP DlyRet), the universe is frozen at the 2025-12-26 "
            "PIT-eligible set (no re-ranking, no delisting), and data/pit_universe.csv is "
            "forward-filled rather than freshly determined for those weeks (see "
            "weekly_returns_2026_yfinance_extension.md and pit_universe_2026_forward_fill_extension.md). "
            "Any figure above whose window includes this tail is not directly comparable to one "
            "computed entirely on pre-2026 WRDS data."
        )
    if ctx.rebalance_every_weeks != 1 and (ctx.kelly_mix_enabled or ctx.kelly_scale_enabled or ctx.crowding_enabled):
        status = "FLAG"
        notes.append(
            f"--rebalance-every-weeks={ctx.rebalance_every_weeks} with Kelly/crowding enabled: "
            "both are specified in calendar weeks, not rebalance calls, so their effective "
            "timescales stretch by this factor. Use 1 for the numbers above to mean what their "
            "pre-registrations say (see kelly-buffer-check)."
        )
    return CheckResult(
        name="data-caveats", status=status,
        headline="; ".join(notes) if notes else "No known data-provenance caveats apply to this window/config",
        detail="Pulled from this project's own documented data-extension and cadence notes, not "
               "computed from the data itself.",
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

CHECK_ORDER = [
    ("walk-forward-oos", check_walk_forward_oos, "skip_walk_forward"),
    ("kelly-buffer-check", check_kelly_buffer_filled, "skip_kelly_buffer_check"),
    ("cpcv", check_cpcv, "skip_cpcv"),
    ("block-bootstrap", check_block_bootstrap, "skip_bootstrap"),
    ("cost-stress", check_cost_stress, "skip_bootstrap"),
    ("repair-frequency", check_repair_frequency, "skip_bootstrap"),
    ("placebo", check_placebo, "skip_placebo"),
    ("significance", None, "skip_significance"),  # wired up specially, needs trial_counts
    ("crowding-stress-window", check_crowding_stress_window, "skip_crowding"),
    ("effective-breadth", check_effective_breadth, "skip_effective_breadth"),
    ("data-caveats", check_data_caveats, None),
]


def _run_check(name, fn, results, ctx, store, skip=False):
    if skip:
        results.append(CheckResult(name=name, status="SKIPPED", headline="Skipped by flag"))
        print(f"[{name}] SKIPPED (by flag)")
        return
    print(f"[{name}] running...", flush=True)
    try:
        result = fn(ctx, store)
        results.append(result)
        print(f"[{name}] {result.status}: {result.headline}")
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: one check's crash must not kill the rest
        tb = traceback.format_exc()
        results.append(CheckResult(name=name, status="ERROR", headline=str(exc), detail=tb))
        print(f"[{name}] ERROR: {exc}")
        print("  (isolated -- the rest of the battery continues; full traceback is in the report)")


def write_report(results, ctx: Context, output_dir: Path):
    status_order = {"FAIL": 0, "ERROR": 1, "FLAG": 2, "PASS": 3, "SKIPPED": 4}
    ordered = sorted(results, key=lambda r: status_order.get(r.status, 5))
    lines = [
        f"# Overfitting / bias diagnostics -- {ctx.model_name}", "",
        f"Run at {datetime.now().isoformat()}",
        f"Evaluation window: {ctx.evaluation_start} to {ctx.evaluation_end}, "
        f"rebalance every {ctx.rebalance_every_weeks} week(s)",
        "",
    ]
    for r in ordered:
        lines.append(f"## [{r.status}] {r.name}")
        lines.append("")
        lines.append(r.headline)
        if r.detail:
            lines.append("")
            lines.append(r.detail)
        if r.metrics:
            lines.append("")
            for k, v in r.metrics.items():
                lines.append(f"- **{k}**: {v}")
        if r.artifacts:
            lines.append("")
            for a in r.artifacts:
                lines.append(f"- artifact: `{a}`")
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines))
    (output_dir / "report.json").write_text(json.dumps([asdict(r) for r in results], indent=2, default=str))


def print_summary(results):
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for r in results:
        print(f"  [{r.status:8s}] {r.name:24s} {r.headline}")
    n_fail = sum(1 for r in results if r.status in ("FAIL", "ERROR"))
    n_flag = sum(1 for r in results if r.status == "FLAG")
    print("=" * 78)
    if n_fail:
        print(f"{n_fail} check(s) FAILED/ERRORED -- do not trust this model's numbers until resolved.")
    elif n_flag:
        print(f"{n_flag} check(s) raised a FLAG -- review before trusting the headline performance.")
    else:
        print("No fails or flags. This is NOT proof the model is good -- it means this particular "
              "battery didn't find a problem. See report.md for what each check does and doesn't cover.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--returns", default="data/weekly_returns.csv")
    p.add_argument("--pit", default="data/pit_universe.csv")
    p.add_argument("--model", choices=["retail_alpha_ml_mpc", "retail_alpha_ml_mpc_crowding"],
                    default="retail_alpha_ml_mpc_crowding")
    p.add_argument("--evaluation-start", default=None)
    p.add_argument("--evaluation-end", default=None)
    p.add_argument("--data-start", default=None)
    p.add_argument("--rebalance-every-weeks", type=int, default=1)
    p.add_argument("--max-total-assets", type=int, default=200)
    p.add_argument("--kelly-mix-enabled", dest="kelly_mix_enabled", action="store_true", default=True)
    p.add_argument("--no-kelly-mix", dest="kelly_mix_enabled", action="store_false")
    p.add_argument("--kelly-scale-enabled", dest="kelly_scale_enabled", action="store_true", default=True)
    p.add_argument("--no-kelly-scale", dest="kelly_scale_enabled", action="store_false")
    p.add_argument("--allow-cvar-floor-relaxation", action="store_true", default=True,
                    help="Safety net for an unattended multi-path/bootstrap run (default on for this script).")
    p.add_argument("--cpcv-splits", type=int, default=6)
    p.add_argument("--cpcv-embargo-weeks", type=int, default=4)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--block-length-weeks", type=int, default=8)
    p.add_argument("--placebo-seed", type=int, default=13)
    p.add_argument("--dsr-trial-counts", type=int, nargs="+", default=[4, 10, 50, 200, 1000])
    p.add_argument("--allow-approximate-significance", action="store_true", default=False)
    p.add_argument("--output-dir", default="outputs/overfitting_diagnostics")
    p.add_argument("--quick", action="store_true",
                    help="Trimmed splits/bootstrap draws to smoke-test THIS SCRIPT -- not a real evaluation.")
    p.add_argument("--skip-walk-forward", action="store_true", default=False)
    p.add_argument("--skip-kelly-buffer-check", action="store_true", default=False)
    p.add_argument("--skip-cpcv", action="store_true", default=False)
    p.add_argument("--skip-bootstrap", action="store_true", default=False)
    p.add_argument("--skip-placebo", action="store_true", default=False)
    p.add_argument("--skip-significance", action="store_true", default=False)
    p.add_argument("--skip-crowding", action="store_true", default=False)
    p.add_argument("--skip-effective-breadth", action="store_true", default=False)
    args = p.parse_args(argv)
    if args.quick:
        args.cpcv_splits = min(args.cpcv_splits, 3)
        args.n_bootstrap = min(args.n_bootstrap, 50)
        args.max_total_assets = min(args.max_total_assets, 40)
    return args


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    returns = pd.read_csv(args.returns, index_col=0, parse_dates=True)
    pit = pd.read_csv(args.pit, index_col=0, parse_dates=True) if args.pit else None

    crowding = args.model == "retail_alpha_ml_mpc_crowding"
    config = _build_config(args, crowding=crowding)
    allocator = _build_allocator(config)

    ctx = Context(
        returns=returns, pit=pit, config=config, allocator=allocator,
        evaluation_start=args.evaluation_start, evaluation_end=args.evaluation_end,
        data_start=args.data_start, rebalance_every_weeks=args.rebalance_every_weeks,
        model_name=args.model, kelly_mix_enabled=args.kelly_mix_enabled,
        kelly_scale_enabled=args.kelly_scale_enabled, crowding_enabled=crowding,
        n_bootstrap=args.n_bootstrap, block_length_weeks=args.block_length_weeks,
        cpcv_splits=args.cpcv_splits, cpcv_embargo_weeks=args.cpcv_embargo_weeks,
        placebo_seed=args.placebo_seed, output_dir=output_dir,
        allow_approximate_significance=args.allow_approximate_significance,
    )

    store: dict = {}
    results = []

    for name, fn, skip_attr in CHECK_ORDER:
        skip = bool(getattr(args, skip_attr, False)) if skip_attr else False
        if name == "significance":
            if skip:
                results.append(CheckResult(name=name, status="SKIPPED", headline="Skipped by flag"))
                print(f"[{name}] SKIPPED (by flag)")
                continue
            _run_check(name, lambda c, s: check_significance(c, s, args.dsr_trial_counts), results, ctx, store)
            continue
        if name == "crowding-stress-window" and not crowding:
            results.append(CheckResult(name=name, status="SKIPPED", headline="Crowding overlay not enabled for this run"))
            print(f"[{name}] SKIPPED (crowding not enabled)")
            continue
        _run_check(name, fn, results, ctx, store, skip=skip)

    write_report(results, ctx, output_dir)
    print_summary(results)
    print(f"\nFull report: {output_dir / 'report.md'}  /  {output_dir / 'report.json'}")

    n_fail = sum(1 for r in results if r.status in ("FAIL", "ERROR"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
