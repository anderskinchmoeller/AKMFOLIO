from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from akm_hrp.allocators.barra_factor_hrp import (
    BarraFactorHRPAllocator,
    BarraFactorHRPConfig,
)
from akm_hrp.allocators.dynamic_barra_alpha import (
    DynamicBarraAlphaAllocator,
    DynamicBarraAlphaConfig,
)
from akm_hrp.allocators.hrp_alpha_v1 import (
    HRPAlphaV1Allocator,
    HRPAlphaV1Config,
)
from akm_hrp.allocators.hrp_alpha_v2 import (
    HRPAlphaV2Allocator,
    HRPAlphaV2Config,
    combine_structural_feature_tables,
)
from akm_hrp.allocators.hrp_overlay_allocator import (
    AllocatorConfig,
    HRPOverlayAllocator,
)
from akm_hrp.allocators.low_overfit_hrp import LowOverfitHRPAllocator
from akm_hrp.allocators.mapper_factor_nco import (
    MapperFactorNCOAllocator,
    MapperFactorNCOConfig,
)
from akm_hrp.allocators.ra_hrp_allocator import RAHRPAllocator, RAHRPConfig
from akm_hrp.allocators.ra_hrp_v2_allocator import RAHRPV2Allocator, RAHRPV2Config
from akm_hrp.allocators.regularized_minimum_variance import (
    RegularizedMinimumVarianceAllocator,
    RegularizedMinimumVarianceConfig,
)
from akm_hrp.allocators.retail_alpha_ml_mpc import (
    RetailAlphaMLMPCAllocator,
    RetailAlphaMLMPCConfig,
)
from akm_hrp.allocators.retail_alpha_mpc import (
    RetailAlphaMPCAllocator,
    RetailAlphaMPCConfig,
)
from akm_hrp.allocators.retail_edge_mpc import (
    RetailEdgeMPCAllocator,
    RetailEdgeMPCConfig,
)
from akm_hrp.allocators.retail_edge_ml_mpc import (
    RetailEdgeMLMPCAllocator,
    RetailEdgeMLMPCConfig,
)
from akm_hrp.backtest.engine import (
    _compute_metrics,
    latest_target_weights,
    run_walk_forward,
)
from akm_hrp.config import HRPConfig
from akm_hrp.data.returns import load_and_clean_returns
from akm_hrp.diagnostics.dashboard import export_backtest_dashboard, _export_dashboard_period
from akm_hrp.diagnostics.robustness import write_robustness_report
from akm_hrp.diagnostics.significance import newey_west_mean_test
from akm_hrp.overlay.bounds import apply_bounds


@dataclass(frozen=True)
class _BenchmarkConfig:
    min_weight: float = 0.0
    max_weight: float = 0.10


class EqualWeightAllocator:
    def __init__(self, max_weight: float):
        self.config = _BenchmarkConfig(max_weight=max_weight)

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        raw = pd.Series(1.0 / returns.shape[1], index=returns.columns)
        return apply_bounds(raw, 0.0, self.config.max_weight)


class InverseVolatilityAllocator:
    def __init__(self, max_weight: float):
        self.config = _BenchmarkConfig(max_weight=max_weight)

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        volatility = returns.std(ddof=1).clip(lower=1e-12)
        raw = (1.0 / volatility).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw /= raw.sum()
        return apply_bounds(raw, 0.0, self.config.max_weight)


def _parse_turnover_cap(value: str):
    if str(value).strip().lower() == "none":
        return None
    if value == "default":
        return "default"
    cap = float(value)
    if not (cap >= 0.0 and cap < float("inf")):
        raise argparse.ArgumentTypeError(
            "--max-rebalance-turnover must be a non-negative number or 'none'."
        )
    return cap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare robust HRP with institutional benchmark allocators."
    )
    parser.add_argument("--returns", default="weekly_returns.csv")
    parser.add_argument("--dashboard-every-year", action="store_true",
                        help="Save and announce annual dashboards during the focus model run.")
    parser.add_argument(
        "--factor-returns",
        default=None,
        help=(
            "Optional date-indexed wide CSV of weekly factor returns in decimal "
            "units. mapper_factor_nco learns PCA factors when omitted."
        ),
    )
    parser.add_argument(
        "--dynamic-balanced-pit",
        default=None,
        help=(
            "Balanced-universe PIT CSV used as the mandatory core by "
            "dynamic_barra_alpha, retail_alpha_mpc, retail_alpha_ml_mpc, "
            "retail_alpha_ml_mpc_crowding, retail_edge_mpc, and "
            "retail_edge_ml_mpc. "
            "Defaults to "
            "balanced_hrp/pit_universe.csv "
            "beside the broad returns bundle."
        ),
    )
    parser.add_argument(
        "--dynamic-balanced-returns",
        default=None,
        help=(
            "Balanced weekly returns whose balanced-only sleeves are merged "
            "into the broad input for retail MPC models. Defaults to "
            "balanced_hrp/weekly_returns.csv beside the broad bundle."
        ),
    )
    parser.add_argument(
        "--dynamic-features",
        nargs="+",
        default=None,
        help=(
            "PIT market/fundamental feature CSV(s) used for top-2500 candidate "
            "admission and factor exposures."
        ),
    )
    parser.add_argument(
        "--dynamic-sector-history",
        default=None,
        help="Point-in-time CRSP sector history for dynamic universe models.",
    )
    parser.add_argument(
        "--dynamic-max-added-assets",
        type=int,
        default=50,
        help="Maximum top-2500 additions beyond the balanced core.",
    )
    parser.add_argument(
        "--dynamic-portfolio-value",
        type=float,
        default=1_000_000.0,
        help="Portfolio value used by dynamic candidate capacity constraints.",
    )
    parser.add_argument(
        "--retail-mpc-horizon",
        type=int,
        default=3,
        help="Planning periods used by retail_alpha_mpc (default: 3).",
    )
    parser.add_argument(
        "--retail-max-added-assets",
        type=int,
        default=30,
        help="Maximum top-2500 additions for retail_alpha_mpc (default: 30).",
    )
    parser.add_argument(
        "--retail-optimizer-max-iterations",
        type=int,
        default=450,
        help="Maximum SLSQP iterations for retail_alpha_mpc (default: 450).",
    )
    parser.add_argument(
        "--retail-mpc-allow-cvar-floor-relaxation",
        action="store_true",
        help=(
            "Let retail_alpha_mpc's optimizer log a small, controlled CVaR "
            "overshoot instead of raising when constraint repair can't reach "
            "exact feasibility. Off by default, matching "
            "RetailAlphaMPCConfig's own risk-policy default of raising "
            "rather than silently breaching the CVaR limit -- pass this if "
            "an unattended walk-forward run should keep going through that "
            "corner instead. Mirrors "
            "--retail-alpha-ml-allow-cvar-floor-relaxation, which only "
            "wires into retail_alpha_ml_mpc, not this (plain) model."
        ),
    )
    parser.add_argument(
        "--retail-alpha-ml-max-total-assets",
        type=int,
        default=40,
        help=(
            "Hard cap on retail_alpha_ml_mpc's total book size (core + "
            "additions); the derived per-name weight floor is "
            "1/this value (default: 40)."
        ),
    )
    parser.add_argument(
        "--retail-alpha-ml-min-weight",
        type=float,
        default=None,
        help=(
            "Per-name weight floor for names retail_alpha_ml_mpc holds "
            "(e.g. 0.01 = 1%%). Default: 1/--retail-alpha-ml-max-total-assets."
        ),
    )
    parser.add_argument(
        "--retail-alpha-ml-max-weight",
        type=float,
        default=None,
        help=(
            "Override RetailAlphaMPCConfig.max_weight (default 0.03), the "
            "per-name position cap the ML sleeve inherits. Pass 1.0 to remove "
            "the cap entirely -- concentration is then limited only by the "
            "sector and +/-0.25 style-exposure caps and by the max-total-assets "
            "trim, and the optimizer's feasible region grows a lot, so solves "
            "get slower and repair passes more likely."
        ),
    )
    parser.add_argument(
        "--retail-alpha-ml-max-training-cross-sections",
        type=int,
        default=None,
        help=(
            "Override RetailAlphaMLMPCConfig.ml_max_training_cross_sections "
            "(default 252). The ML ensemble refit cost grows with this "
            "deque's current length as it fills, so lowering it bounds "
            "per-rebalance compute cost for long weekly-cadence runs at the "
            "expense of a shorter effective ML training lookback."
        ),
    )
    parser.add_argument(
        "--retail-alpha-ml-min-training-cross-sections",
        type=int,
        default=None,
        help=(
            "Override ml_minimum_training_cross_sections (default 12): how "
            "many buffered cross-sections must exist before the ML ensemble "
            "trains at all. A sample-size gate, counted in cross-sections, "
            "so it does NOT need rescaling when the cadence changes."
        ),
    )
    parser.add_argument(
        "--retail-alpha-ml-fast-halflife",
        type=float,
        default=None,
        help=(
            "Override ml_fast_halflife_rebalances (default 6). Counted in "
            "REBALANCES, not weeks, so its calendar meaning moves with "
            "--rebalance-every-weeks: 6 is ~6 weeks at weekly cadence and "
            "~18 months at 13w. See --retail-alpha-ml-slow-halflife."
        ),
    )
    parser.add_argument(
        "--retail-alpha-ml-slow-halflife",
        type=float,
        default=None,
        help=(
            "Override ml_slow_halflife_rebalances (default 24). Also counted "
            "in rebalances. The fast/slow pair is the ensemble's only source "
            "of diversity between its two gradient-boosted votes -- they are "
            "otherwise the same estimator on the same buffer -- so the gap "
            "between them is what makes the third vote's job meaningful. "
            "Must be strictly greater than the fast halflife."
        ),
    )
    parser.add_argument(
        "--retail-alpha-ml-retrain-every",
        type=int,
        default=None,
        help=(
            "Override ml_retrain_every_n_rebalances (default 5). In "
            "rebalances: ~5 weeks at weekly cadence, ~15 months at 13w."
        ),
    )
    parser.add_argument(
        "--retail-alpha-ml-ic-halflife",
        type=float,
        default=None,
        help=(
            "Override ic_halflife_rebalances (default 12), the decay on the "
            "rolling rank-IC estimates that weight each signal. In "
            "rebalances: ~12 weeks at weekly cadence, ~3 years at 13w."
        ),
    )
    parser.add_argument(
        "--retail-alpha-ml-allow-exposure-limit-relaxation",
        action="store_true",
        help=("Allow the minimum common additive increase in sector/style caps "
              "when the ML MPC affine constraints are infeasible; logs the "
              "increase and records exposure_limit_relaxation in diagnostics."),
    )
    parser.add_argument(
        "--retail-alpha-ml-allow-cvar-floor-relaxation",
        action="store_true",
        help=(
            "Let retail_alpha_ml_mpc's optimizer log a small, controlled CVaR "
            "overshoot instead of raising when constraint repair can't reach "
            "exact feasibility. Off by default, matching retail_alpha_mpc's "
            "own risk-policy default of raising rather than silently "
            "breaching the CVaR limit -- pass this if an unattended "
            "walk-forward run should keep going through that corner instead."
        ),
    )
    parser.add_argument(
        "--retail-alpha-ml-kelly-mix-enabled",
        action="store_true",
        help=(
            "Blend retail_alpha_ml_mpc's signals by growth-optimal factor "
            "covariance (KNS ridge) instead of the parent's cross-sectional "
            "signal-score correlation. Off by default -- applies to both "
            "retail_alpha_ml_mpc and retail_alpha_ml_mpc_crowding. See "
            "kelly_factor_sizing_preregistration.md."
        ),
    )
    parser.add_argument(
        "--retail-alpha-ml-kelly-scale-enabled",
        action="store_true",
        help=(
            "Size retail_alpha_ml_mpc's total factor-sleeve exposure by the "
            "derived growth-optimal fraction c* instead of the "
            "alpha_strength/risk_aversion knobs. Off by default -- applies to "
            "both retail_alpha_ml_mpc and retail_alpha_ml_mpc_crowding. See "
            "kelly_factor_sizing_preregistration.md."
        ),
    )
    parser.add_argument(
        "--retail-edge-mpc-horizon",
        type=int,
        default=1,
        help="Planning periods used by retail_edge_mpc (default: 1).",
    )
    parser.add_argument(
        "--retail-edge-max-added-assets",
        type=int,
        default=12,
        help="Maximum capacity-edge additions for retail_edge_mpc (default: 12).",
    )
    parser.add_argument(
        "--retail-edge-optimizer-max-iterations",
        type=int,
        default=30,
        help="Maximum primary SLSQP iterations for retail_edge_mpc (default: 30).",
    )
    parser.add_argument(
        "--deflated-sharpe-trials",
        type=int,
        default=1,
        help=(
            "Honest total number of model/parameter variants searched, including "
            "discarded attempts. The effective count is at least the number of "
            "models in this run (default: 1)."
        ),
    )
    parser.add_argument(
        "--significance-benchmark",
        default="equal_weight",
        help="Model used for HAC active-return significance diagnostics.",
    )
    parser.add_argument(
        "--barra-max-weight",
        type=float,
        default=0.03,
        help="Maximum Barra-style Factor HRP position (default: 0.03).",
    )
    parser.add_argument(
        "--barra-risk-contribution-cap",
        type=float,
        default=0.08,
        help="HRP risk-contribution regularization target (default: 0.08).",
    )
    parser.add_argument(
        "--barra-specific-variance-shrinkage",
        type=float,
        default=0.35,
        help="Specific-risk shrinkage toward the cross-sectional median.",
    )
    parser.add_argument(
        "--barra-specific-variance-floor-fraction",
        type=float,
        default=0.05,
        help="Minimum specific variance as a fraction of asset variance.",
    )
    parser.add_argument(
        "--hrp-alpha-max-weight",
        type=float,
        default=0.20,
        help="Maximum HRP Alpha v1 risky-asset weight (default: 0.20).",
    )
    parser.add_argument(
        "--hrp-alpha-no-trade-band",
        type=float,
        default=0.02,
        help="Per-position HRP Alpha v1 target freeze band (default: 0.02).",
    )
    parser.add_argument(
        "--hrp-alpha-target-vol",
        type=float,
        default=None,
        help=(
            "Optional annual volatility target. Requires --hrp-alpha-cash-asset "
            "to name a real cash/T-bill column in the returns file."
        ),
    )
    parser.add_argument(
        "--hrp-alpha-cash-asset",
        default=None,
        help="Returns-column name used as the cash/T-bill sleeve.",
    )
    parser.add_argument(
        "--hrp-alpha-max-cash-weight",
        type=float,
        default=1.0,
        help="Maximum cash/T-bill weight under volatility targeting.",
    )
    parser.add_argument(
        "--hrp-alpha-structural-features",
        nargs="+",
        default=None,
        help=(
            "Optional PIT long feature CSV(s) for hrp_alpha_v2. Tables may "
            "contain market/liquidity, fundamental, or live event fields."
        ),
    )
    parser.add_argument(
        "--hrp-alpha-portfolio-value",
        type=float,
        default=100_000.0,
        help="Portfolio value used for v2 dollar-volume capacity caps.",
    )
    parser.add_argument(
        "--hrp-alpha-max-adv-participation",
        type=float,
        default=0.01,
        help="Maximum position value as a fraction of 20-day dollar volume.",
    )
    parser.add_argument("--pit", default=None)
    parser.add_argument("--as-of", default=None)
    parser.add_argument(
        "--data-start",
        type=pd.Timestamp,
        default=None,
        help="Discard rows before this date, including estimator warm-up rows.",
    )
    parser.add_argument(
        "--lookback-weeks",
        type=int,
        default=None,
        help=(
            "Override the trailing estimation window (HRPConfig.lookback_weeks, "
            "default 260) used by every allocator, including the ML/MPC models. "
            "Useful to bound per-rebalance compute cost for weekly-cadence runs, "
            "since the window grows with available history up to this cap."
        ),
    )
    parser.add_argument(
        "--evaluation-start",
        type=pd.Timestamp,
        default=None,
        help=(
            "First date included in reported metrics; earlier rows remain "
            "available for estimator warm-up."
        ),
    )
    parser.add_argument("--tc-bps", type=float, default=10.0)
    parser.add_argument(
        "--max-rebalance-turnover",
        type=_parse_turnover_cap,
        default="default",
        help=(
            "Per-rebalance L1 turnover cap applied by the backtest engine to "
            "every model (2.0 = the whole book can be replaced; 'none' = no "
            "cap). Default: HRPConfig's 0.10, which is sized for weekly "
            "rebalancing and freezes the book at slow cadences."
        ),
    )
    parser.add_argument(
        "--rebalance-every-weeks",
        type=int,
        default=4,
        help="Minimum weeks between allocations (default: 4).",
    )
    parser.add_argument(
        "--progress-every-rebalances",
        type=int,
        default=1,
        help="Print elapsed time and ETA every N allocations; use 0 to disable.",
    )
    parser.add_argument("--output", default="outputs/model_comparison.csv")
    parser.add_argument(
        "--weights-output",
        default=None,
        help=(
            "Optional CSV for latest post-close target weights. The file uses "
            "date, model, asset, permno, and weight columns when CRSP metadata "
            "is available."
        ),
    )
    parser.add_argument(
        "--asset-metadata",
        default=None,
        help=(
            "Optional CSV containing permno and ticker columns. By default, "
            "crsp_security_metadata.csv beside the returns file is used when present."
        ),
    )
    parser.add_argument(
        "--weights-png",
        default=None,
        help="Optional PNG chart of the latest target weights.",
    )
    parser.add_argument(
        "--weights-png-top",
        type=int,
        default=25,
        help="Number of largest individual positions shown in the PNG (default: 25).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model names to run; omitted means all registered models.",
    )
    parser.add_argument(
        "--diagnostics-output",
        default=None,
        help="Optional CSV containing latest scalar allocator diagnostics.",
    )
    parser.add_argument(
        "--robustness-output",
        default=None,
        help=(
            "Optional directory for an offline robustness report: paired "
            "moving-block and regime-stratified bootstrap confidence "
            "intervals, causal volatility-regime and CUSUM structural-break "
            "HAC significance, fixed-trade cost stress at 1x/1.5x/2x, and "
            "optimizer-repair/CVaR-relaxation frequency, per model against "
            "--significance-benchmark. Never affects backtest results -- "
            "purely a post-hoc report over what already ran."
        ),
    )
    parser.add_argument(
        "--robustness-samples",
        type=int,
        default=1000,
        help="Bootstrap draws per model for the robustness report (default: 1000).",
    )
    parser.add_argument(
        "--robustness-block-weeks",
        type=int,
        default=8,
        help="Moving block length in weeks for the robustness bootstrap (default: 8).",
    )
    parser.add_argument(
        "--robustness-seed",
        type=int,
        default=0,
        help="Random seed for the robustness bootstrap (default: 0).",
    )
    parser.add_argument(
        "--dashboard-pdf",
        default=None,
        help="Optional one-page PDF dashboard built from these backtest results.",
    )
    parser.add_argument(
        "--dashboard-png",
        default=None,
        help="Optional PNG copy of the one-page research dashboard.",
    )
    parser.add_argument(
        "--dashboard-focus-model",
        default=None,
        help=(
            "Model used for transaction-cost and historical-weight panels. "
            "Default: the first name passed to --models."
        ),
    )
    parser.add_argument(
        "--dashboard-title",
        default=None,
        help="Optional dashboard title; defaults to the focus model name.",
    )
    parser.add_argument(
        "--dashboard-rolling-sharpe-years",
        type=int,
        default=3,
        help="Rolling Sharpe window shown in the dashboard (default: 3 years).",
    )
    parser.add_argument(
        "--dashboard-heatmap-assets",
        type=int,
        default=15,
        help="Number of top average weights shown in the heatmap (default: 15).",
    )
    return parser.parse_args()


def _load_ticker_map(path: Path) -> dict[str, str]:
    """Load a one-to-one PERMNO-to-ticker lookup for readable exports."""
    metadata = pd.read_csv(path, dtype=str)
    required = {"permno", "ticker"}
    if not required.issubset(metadata.columns):
        raise ValueError(
            f"Asset metadata must contain {sorted(required)} columns: {path}"
        )

    clean = metadata.loc[:, ["permno", "ticker"]].copy()
    clean["permno"] = clean["permno"].str.strip()
    clean["ticker"] = clean["ticker"].str.strip()
    clean = clean.loc[(clean["permno"] != "") & (clean["ticker"] != "")]

    conflicting = clean.groupby("permno")["ticker"].nunique().gt(1)
    if conflicting.any():
        examples = conflicting[conflicting].index[:5].tolist()
        raise ValueError(f"Conflicting ticker mappings for PERMNOs: {examples}")

    clean = clean.drop_duplicates("permno", keep="last")
    return dict(zip(clean["permno"], clean["ticker"], strict=True))


def _resolve_asset_metadata_path(args: argparse.Namespace) -> Path | None:
    """Locate CRSP permno/ticker metadata for readable asset labels.

    An explicit --asset-metadata always wins. Otherwise we look beside the
    returns file and beside the PIT file (their usual home), then fall back
    to the repo-wide canonical data/crsp_security_metadata.csv. That last
    fallback matters for trimmed/scratch inputs -- e.g. the smoke test's
    outputs/smoke_test/weekly_returns_trimmed_*.csv -- which don't carry
    their own metadata file and previously left every weights/dashboard
    export showing raw PERMNOs with no way to recover the ticker.
    """
    if args.asset_metadata:
        explicit = Path(args.asset_metadata)
        if explicit.exists():
            return explicit
        print(f"warning: --asset-metadata path not found: {explicit}", flush=True)
        return None

    candidates = [Path(args.returns).resolve().parent / "crsp_security_metadata.csv"]
    pit_path = getattr(args, "pit", None)
    if pit_path:
        candidates.append(Path(pit_path).resolve().parent / "crsp_security_metadata.csv")
    candidates.append(
        Path(__file__).resolve().parents[2] / "data" / "crsp_security_metadata.csv"
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    print(
        "warning: no asset metadata found in any of "
        f"{[str(c) for c in candidates]}; asset values remain source identifiers",
        flush=True,
    )
    return None


def _merge_balanced_only_returns(
    broad: pd.DataFrame, balanced: pd.DataFrame
) -> pd.DataFrame:
    """Add balanced-only sleeves without replacing broad CRSP stock returns."""

    extra = balanced.columns.difference(broad.columns, sort=False)
    if not len(extra):
        return broad
    supplement = balanced.loc[:, extra].reindex(broad.index)
    return broad.join(supplement, how="left", validate="one_to_one")


def _label_weight_frame(
    frame: pd.DataFrame,
    ticker_map: dict[str, str],
) -> pd.DataFrame:
    """Use ticker as the display asset while retaining PERMNO for traceability."""
    labelled = frame.copy()
    permno = labelled["asset"].astype(str)
    ticker = permno.map(ticker_map)
    labelled.insert(labelled.columns.get_loc("asset") + 1, "permno", permno)
    labelled["asset"] = ticker.fillna(permno)
    return labelled


def _save_weights_png(frame: pd.DataFrame, path: Path, top_n: int) -> None:
    """Plot the largest positions and aggregate the remainder for legibility."""
    if top_n < 1:
        raise ValueError("--weights-png-top must be at least 1")

    import matplotlib

    # File-only rendering must work in CI, SSH sessions, and other headless runs.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    models = list(frame["model"].drop_duplicates())
    figure_height = max(5.0, 0.34 * (min(top_n, len(frame)) + 1) * len(models))
    figure, axes = plt.subplots(
        len(models),
        1,
        figsize=(11, figure_height),
        squeeze=False,
    )

    for axis, model in zip(axes.flat, models, strict=True):
        model_weights = frame.loc[frame["model"] == model].sort_values(
            "weight", ascending=False
        )
        displayed = model_weights.head(top_n).loc[:, ["asset", "weight"]].copy()
        remaining = model_weights.iloc[top_n:]
        if not remaining.empty:
            displayed.loc[len(displayed)] = {
                "asset": f"Other ({len(remaining)} assets)",
                "weight": float(remaining["weight"].sum()),
            }
        displayed = displayed.iloc[::-1]

        bars = axis.barh(displayed["asset"], displayed["weight"], color="#3568a8")
        axis.bar_label(
            bars,
            labels=[f"{value:.2%}" for value in displayed["weight"]],
            padding=4,
        )
        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_xlabel("Portfolio weight")
        as_of = str(model_weights["date"].iloc[0])
        axis.set_title(f"{model} target weights — {as_of}")
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        axis.margins(x=0.15)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _ml_config_overrides(args) -> dict:
    """Explicitly-set ML memory overrides, validated against each other.

    Five of these settings are counted in REBALANCES, not weeks, so changing
    --rebalance-every-weeks silently rescales what the model remembers: the
    "fast" halflife of 6 means ~6 weeks at weekly cadence and ~18 months at
    13-week cadence, and the fast/slow pair (6 vs 24) collapses from 6-vs-24
    weeks to 18-months-vs-6-years. That is why they are exposed here rather
    than left to the dataclass defaults.

    There is no automatic cadence rescaling, deliberately. Preserving the
    weekly *calendar* horizons at 13w would put both tree halflives below a
    single rebalance, which is degenerate; preserving them in rebalances is
    what already happens. Neither is right, so the choice is the operator's,
    and the validation below only rules out the combinations that are
    incoherent whatever the cadence.

    The two cross-section settings are different in kind: they are sample-size
    gates on the training buffer, not memory horizons, so they are counted in
    cross-sections and do not track the cadence at all.
    """

    defaults = RetailAlphaMLMPCConfig()
    overrides: dict = {}

    def take(attribute, config_field):
        value = getattr(args, attribute, None)
        if value is not None:
            overrides[config_field] = value
        return overrides.get(config_field, getattr(defaults, config_field))

    maximum = take(
        "retail_alpha_ml_max_training_cross_sections",
        "ml_max_training_cross_sections",
    )
    minimum = take(
        "retail_alpha_ml_min_training_cross_sections",
        "ml_minimum_training_cross_sections",
    )
    fast = take("retail_alpha_ml_fast_halflife", "ml_fast_halflife_rebalances")
    slow = take("retail_alpha_ml_slow_halflife", "ml_slow_halflife_rebalances")
    retrain = take("retail_alpha_ml_retrain_every", "ml_retrain_every_n_rebalances")
    ic_halflife = take("retail_alpha_ml_ic_halflife", "ic_halflife_rebalances")

    if fast <= 0 or slow <= 0 or ic_halflife <= 0:
        raise ValueError(
            "ML halflives must be positive; got fast="
            f"{fast}, slow={slow}, ic={ic_halflife}."
        )
    if fast >= slow:
        # Equal halflives make the two GBMs the same estimator fit twice on the
        # same rows with the same weights: an ensemble of one, at twice the
        # cost, with a consensus fraction that is meaninglessly 1.0.
        raise ValueError(
            f"ML fast halflife ({fast}) must be strictly less than the slow "
            f"halflife ({slow}); equal or inverted values make the two "
            "gradient-boosted votes identical."
        )
    if retrain < 1:
        raise ValueError(f"ML retrain interval must be >= 1 rebalance; got {retrain}.")
    if minimum < 1:
        raise ValueError(
            f"ML minimum training cross-sections must be >= 1; got {minimum}."
        )
    if minimum > maximum:
        raise ValueError(
            f"ML minimum training cross-sections ({minimum}) exceeds the "
            f"maximum ({maximum}); the ensemble could never train."
        )

    # Per-name position cap. Validated against the floor and the book size so a
    # combination that can never be fully invested is rejected at parse time
    # rather than as an opaque solver failure some hours into a walk-forward.
    max_weight = take("retail_alpha_ml_max_weight", "max_weight")
    if not 0.0 < max_weight <= 1.0:
        raise ValueError(f"ML max weight must be in (0, 1]; got {max_weight}.")
    budget = getattr(args, "retail_alpha_ml_max_total_assets", None) or (
        defaults.ml_max_total_assets
    )
    if budget * max_weight < 1.0 - 1e-9:
        raise ValueError(
            f"ML max weight {max_weight} caps a {budget}-name book at "
            f"{budget * max_weight:.4f} of capital; it can never be fully "
            "invested. Raise --retail-alpha-ml-max-weight or the asset budget."
        )
    floor = getattr(args, "retail_alpha_ml_min_weight", None)
    if floor is not None and floor > max_weight:
        raise ValueError(
            f"ML min weight ({floor}) exceeds max weight ({max_weight})."
        )
    if floor is not None and budget * floor > 1.0 + 1e-9:
        # The floor is a post-solve prune-and-reproject, not a solver bound, so
        # this does not crash -- it silently caps the book at 1/floor names and
        # the asset budget above that is inert. Refuse rather than run a
        # walk-forward whose headline setting does nothing.
        raise ValueError(
            f"ML min weight {floor} x asset budget {budget} = "
            f"{budget * floor:.3f} > 1.0, so the floor is unreachable at full "
            f"occupancy: the book is capped at {int(1.0 / floor)} names and the "
            f"budget of {budget} is inert. Lower --retail-alpha-ml-min-weight "
            f"to <= {1.0 / budget:.4f}, or lower the asset budget."
        )
    return overrides


def main() -> None:
    args = parse_args()
    if args.rebalance_every_weeks < 1:
        raise ValueError("--rebalance-every-weeks must be at least 1.")
    if args.retail_optimizer_max_iterations < 1:
        raise ValueError("--retail-optimizer-max-iterations must be at least 1.")
    if args.retail_alpha_ml_max_total_assets < 2:
        raise ValueError("--retail-alpha-ml-max-total-assets must be at least 2.")
    if args.retail_alpha_ml_min_weight is not None and not (
        0.0 < args.retail_alpha_ml_min_weight
        <= 1.0 / args.retail_alpha_ml_max_total_assets
    ):
        raise ValueError(
            "--retail-alpha-ml-min-weight must be in (0, "
            "1/--retail-alpha-ml-max-total-assets]."
        )
    if args.retail_edge_mpc_horizon < 1:
        raise ValueError("--retail-edge-mpc-horizon must be at least 1.")
    if args.retail_edge_max_added_assets < 0:
        raise ValueError("--retail-edge-max-added-assets cannot be negative.")
    if args.retail_edge_optimizer_max_iterations < 1:
        raise ValueError(
            "--retail-edge-optimizer-max-iterations must be at least 1."
        )
    if args.deflated_sharpe_trials < 1:
        raise ValueError("--deflated-sharpe-trials must be at least 1.")
    pit_path = args.pit
    dynamic_names = {
        "dynamic_barra_alpha",
        "retail_alpha_mpc",
        "retail_alpha_ml_mpc",
        "retail_alpha_ml_mpc_equal_weight",
        "retail_alpha_ml_mpc_crowding",
        "retail_edge_mpc",
        "retail_edge_ml_mpc",
    }
    requested_dynamic_names = (
        dynamic_names.intersection(args.models) if args.models is not None else set()
    )
    dynamic_requested = bool(requested_dynamic_names)
    returns_path = Path(args.returns)
    bundle_root = returns_path.parent
    if dynamic_requested and bundle_root.name == "balanced_hrp":
        raise ValueError(
            "Dynamic universe models must receive the broad top-2500 returns "
            "file, not balanced_hrp/weekly_returns.csv."
        )
    balanced_path = (
        Path(args.dynamic_balanced_pit)
        if args.dynamic_balanced_pit
        else bundle_root / "balanced_hrp" / "pit_universe.csv"
    )
    if pit_path is None and args.models is not None:
        inferred_pit = returns_path.resolve().parent / "pit_universe.csv"
        if dynamic_names.intersection(args.models) and inferred_pit.exists():
            pit_path = str(inferred_pit)
    config = replace(
        HRPConfig(),
        pit_universe_file=pit_path,
        strict_pit_universe=bool(pit_path),
        tc_bps=args.tc_bps,
        min_holding_weeks=args.rebalance_every_weeks,
        **(
            {}
            if args.max_rebalance_turnover == "default"
            else {"max_rebalance_turnover_l1": args.max_rebalance_turnover}
        ),
        **(
            {"lookback_weeks": args.lookback_weeks}
            if args.lookback_weeks is not None
            else {}
        ),
    )
    returns = load_and_clean_returns(args.returns, config.min_history_weeks)
    if {
        "retail_alpha_mpc",
        "retail_alpha_ml_mpc",
        "retail_alpha_ml_mpc_equal_weight",
        "retail_alpha_ml_mpc_crowding",
        "retail_edge_mpc",
        "retail_edge_ml_mpc",
    }.intersection(requested_dynamic_names):
        balanced_returns_path = (
            Path(args.dynamic_balanced_returns)
            if args.dynamic_balanced_returns
            else balanced_path.with_name("weekly_returns.csv")
        )
        if not balanced_returns_path.exists():
            raise FileNotFoundError(
                "Retail Alpha MPC balanced returns are missing: "
                f"{balanced_returns_path}"
            )
        balanced_returns = load_and_clean_returns(
            str(balanced_returns_path), config.min_history_weeks
        )
        returns = _merge_balanced_only_returns(returns, balanced_returns)
    if args.data_start is not None:
        returns = returns.loc[returns.index >= args.data_start]
    if args.as_of:
        returns = returns.loc[returns.index <= pd.Timestamp(args.as_of)]

    factor_returns = None
    if args.factor_returns:
        factor_returns = pd.read_csv(
            args.factor_returns,
            index_col=0,
            parse_dates=True,
        ).sort_index()
        factor_returns = factor_returns.apply(pd.to_numeric, errors="coerce")
        factor_returns = factor_returns.loc[factor_returns.index <= returns.index[-1]]
        if factor_returns.empty:
            raise ValueError("--factor-returns contains no usable rows.")

    structural_features = None
    if args.hrp_alpha_structural_features:
        structural_tables = [
            pd.read_csv(path, low_memory=False)
            for path in args.hrp_alpha_structural_features
        ]
        structural_features = combine_structural_feature_tables(structural_tables)

    # Validated once, shared by every RetailAlphaMLMPCConfig built below.
    ml_config_kwargs = _ml_config_overrides(args)

    models = {
        "equal_weight": EqualWeightAllocator(config.max_weight),
        "inverse_volatility": InverseVolatilityAllocator(config.max_weight),
        "regularized_minimum_variance": RegularizedMinimumVarianceAllocator(
            RegularizedMinimumVarianceConfig(max_weight=config.max_weight)
        ),
        "ra_hrp": RAHRPAllocator(
            RAHRPConfig(
                max_weight=config.max_weight,
                annual_risk_free_rate=config.risk_free_rate,
            )
        ),
        "ra_hrp_v2": RAHRPV2Allocator(
            RAHRPV2Config(
                max_weight=config.max_weight,
                annual_risk_free_rate=config.risk_free_rate,
            )
        ),
        "hrp_alpha_v1": HRPAlphaV1Allocator(
            HRPAlphaV1Config(
                max_weight=args.hrp_alpha_max_weight,
                no_trade_band=args.hrp_alpha_no_trade_band,
                annual_target_volatility=args.hrp_alpha_target_vol,
                cash_asset=args.hrp_alpha_cash_asset,
                max_cash_weight=args.hrp_alpha_max_cash_weight,
            )
        ),
        "hrp_alpha_v2": HRPAlphaV2Allocator(
            HRPAlphaV2Config(
                max_weight=args.hrp_alpha_max_weight,
                no_trade_band=args.hrp_alpha_no_trade_band,
                portfolio_value=args.hrp_alpha_portfolio_value,
                maximum_adv_participation=args.hrp_alpha_max_adv_participation,
            ),
            structural_features=structural_features,
        ),
        "mapper_factor_nco": MapperFactorNCOAllocator(
            MapperFactorNCOConfig(max_weight=config.max_weight),
            factor_returns=factor_returns,
        ),
        "barra_factor_hrp": BarraFactorHRPAllocator(
            BarraFactorHRPConfig(
                max_weight=args.barra_max_weight,
                risk_contribution_cap=args.barra_risk_contribution_cap,
                specific_variance_shrinkage=(args.barra_specific_variance_shrinkage),
                specific_variance_floor_fraction=(
                    args.barra_specific_variance_floor_fraction
                ),
            ),
            factor_returns=factor_returns,
        ),
        "low_overfit_hrp": LowOverfitHRPAllocator(),
        "legacy_ensemble_hrp": HRPOverlayAllocator(
            AllocatorConfig(
                use_regret_aware_hrp=False,
                use_signal_budget=False,
                use_cov_inverse=False,
            )
        ),
        "regret_aware_core": HRPOverlayAllocator(
            AllocatorConfig(
                use_regret_aware_hrp=True,
                use_signal_budget=False,
                use_cov_inverse=False,
            )
        ),
        "regret_aware_with_overlay": HRPOverlayAllocator(AllocatorConfig()),
    }
    if dynamic_requested:
        feature_paths = (
            [Path(path) for path in args.dynamic_features]
            if args.dynamic_features
            else [
                bundle_root / "structural_alpha_features.csv.gz",
                bundle_root / "compustat_pit_features_long.csv.gz",
            ]
        )
        sector_path = (
            Path(args.dynamic_sector_history)
            if args.dynamic_sector_history
            else bundle_root / "crsp_sector_history.csv.gz"
        )
        required_paths = [balanced_path, sector_path, *feature_paths]
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "Dynamic universe model inputs are missing: "
                + ", ".join(missing_paths)
            )
        balanced_pit = pd.read_csv(
            balanced_path,
            index_col=0,
            parse_dates=True,
        )
        dynamic_tables = [pd.read_csv(path, low_memory=False) for path in feature_paths]
        dynamic_features = combine_structural_feature_tables(dynamic_tables)
        sector_history = pd.read_csv(sector_path, low_memory=False)
        if "dynamic_barra_alpha" in requested_dynamic_names:
            models["dynamic_barra_alpha"] = DynamicBarraAlphaAllocator(
                balanced_pit,
                structural_features=dynamic_features,
                sector_history=sector_history,
                config=DynamicBarraAlphaConfig(
                    maximum_added_assets=args.dynamic_max_added_assets,
                    portfolio_value=args.dynamic_portfolio_value,
                ),
            )
        if "retail_alpha_mpc" in requested_dynamic_names:
            models["retail_alpha_mpc"] = RetailAlphaMPCAllocator(
                balanced_pit,
                structural_features=dynamic_features,
                sector_history=sector_history,
                config=RetailAlphaMPCConfig(
                    maximum_added_assets=args.retail_max_added_assets,
                    portfolio_value=args.dynamic_portfolio_value,
                    planning_horizon=args.retail_mpc_horizon,
                    planning_step_weeks=args.rebalance_every_weeks,
                    optimizer_max_iterations=args.retail_optimizer_max_iterations,
                    allow_cvar_floor_relaxation=(
                        args.retail_mpc_allow_cvar_floor_relaxation
                    ),
                ),
            )
        if "retail_alpha_ml_mpc" in requested_dynamic_names:
            models["retail_alpha_ml_mpc"] = RetailAlphaMLMPCAllocator(
                balanced_pit,
                structural_features=dynamic_features,
                sector_history=sector_history,
                config=RetailAlphaMLMPCConfig(
                    maximum_added_assets=args.retail_max_added_assets,
                    portfolio_value=args.dynamic_portfolio_value,
                    planning_horizon=args.retail_mpc_horizon,
                    planning_step_weeks=args.rebalance_every_weeks,
                    optimizer_max_iterations=args.retail_optimizer_max_iterations,
                    ml_max_total_assets=args.retail_alpha_ml_max_total_assets,
                    ml_min_weight=args.retail_alpha_ml_min_weight,
                    **ml_config_kwargs,
                    allow_cvar_floor_relaxation=(
                        args.retail_alpha_ml_allow_cvar_floor_relaxation
                    ),
                    allow_exposure_limit_relaxation=(
                        args.retail_alpha_ml_allow_exposure_limit_relaxation
                    ),
                    ml_kelly_mix_enabled=args.retail_alpha_ml_kelly_mix_enabled,
                    ml_kelly_scale_enabled=args.retail_alpha_ml_kelly_scale_enabled,
                    # The short overlay produces negative weights, which this
                    # engine's _sanitize_target_weights rejects outright (see
                    # backtest/engine.py) -- every other model registered here
                    # is long-only, so the overlay stays off for CLI runs
                    # through this engine until that's addressed deliberately.
                    short_overlay_enabled=False,
                ),
            )
        if "retail_alpha_ml_mpc_equal_weight" in requested_dynamic_names:
            # The sizing control for retail_alpha_ml_mpc: identical selection
            # stack, equal weights instead of the optimizer. Distinct from the
            # `equal_weight` benchmark above, which is 1/N over the whole
            # eligible universe. Running all three decomposes performance:
            #   selection value = this        minus equal_weight
            #   sizing value    = the model   minus this
            models["retail_alpha_ml_mpc_equal_weight"] = RetailAlphaMLMPCAllocator(
                balanced_pit,
                structural_features=dynamic_features,
                sector_history=sector_history,
                config=RetailAlphaMLMPCConfig(
                    maximum_added_assets=args.retail_max_added_assets,
                    portfolio_value=args.dynamic_portfolio_value,
                    planning_horizon=args.retail_mpc_horizon,
                    planning_step_weeks=args.rebalance_every_weeks,
                    optimizer_max_iterations=args.retail_optimizer_max_iterations,
                    ml_max_total_assets=args.retail_alpha_ml_max_total_assets,
                    ml_min_weight=args.retail_alpha_ml_min_weight,
                    **ml_config_kwargs,
                    allow_cvar_floor_relaxation=(
                        args.retail_alpha_ml_allow_cvar_floor_relaxation
                    ),
                    allow_exposure_limit_relaxation=(
                        args.retail_alpha_ml_allow_exposure_limit_relaxation
                    ),
                    ml_kelly_mix_enabled=args.retail_alpha_ml_kelly_mix_enabled,
                    ml_kelly_scale_enabled=args.retail_alpha_ml_kelly_scale_enabled,
                    short_overlay_enabled=False,
                    ml_equal_weight_benchmark=True,
                ),
            )
        if "retail_alpha_ml_mpc_crowding" in requested_dynamic_names:
            # Configuration 5 of crowding_kappa_preregistration.md: identical
            # to retail_alpha_ml_mpc (same mix/scale flags) plus the
            # crowding-regime-conditioned kappa layered on top. Registered as
            # its own name -- not a third flag on retail_alpha_ml_mpc -- so a
            # single run can put "configuration 4" (retail_alpha_ml_mpc),
            # "configuration 5" (this), and equal_weight on the same
            # dashboard for a direct, apples-to-apples comparison.
            models["retail_alpha_ml_mpc_crowding"] = RetailAlphaMLMPCAllocator(
                balanced_pit,
                structural_features=dynamic_features,
                sector_history=sector_history,
                config=RetailAlphaMLMPCConfig(
                    maximum_added_assets=args.retail_max_added_assets,
                    portfolio_value=args.dynamic_portfolio_value,
                    planning_horizon=args.retail_mpc_horizon,
                    planning_step_weeks=args.rebalance_every_weeks,
                    optimizer_max_iterations=args.retail_optimizer_max_iterations,
                    ml_max_total_assets=args.retail_alpha_ml_max_total_assets,
                    ml_min_weight=args.retail_alpha_ml_min_weight,
                    **ml_config_kwargs,
                    allow_cvar_floor_relaxation=(
                        args.retail_alpha_ml_allow_cvar_floor_relaxation
                    ),
                    allow_exposure_limit_relaxation=(
                        args.retail_alpha_ml_allow_exposure_limit_relaxation
                    ),
                    ml_kelly_mix_enabled=args.retail_alpha_ml_kelly_mix_enabled,
                    ml_kelly_scale_enabled=args.retail_alpha_ml_kelly_scale_enabled,
                    ml_kelly_crowding_kappa_enabled=True,
                    short_overlay_enabled=False,
                ),
            )
        if "retail_edge_mpc" in requested_dynamic_names:
            models["retail_edge_mpc"] = RetailEdgeMPCAllocator(
                balanced_pit,
                structural_features=dynamic_features,
                sector_history=sector_history,
                config=RetailEdgeMPCConfig(
                    maximum_added_assets=args.retail_edge_max_added_assets,
                    portfolio_value=args.dynamic_portfolio_value,
                    planning_horizon=args.retail_edge_mpc_horizon,
                    planning_step_weeks=args.rebalance_every_weeks,
                    optimizer_max_iterations=(
                        args.retail_edge_optimizer_max_iterations
                    ),
                ),
            )
        if "retail_edge_ml_mpc" in requested_dynamic_names:
            models["retail_edge_ml_mpc"] = RetailEdgeMLMPCAllocator(
                balanced_pit,
                structural_features=dynamic_features,
                sector_history=sector_history,
                config=RetailEdgeMLMPCConfig(
                    maximum_added_assets=args.retail_edge_max_added_assets,
                    portfolio_value=args.dynamic_portfolio_value,
                    planning_horizon=args.retail_edge_mpc_horizon,
                    planning_step_weeks=args.rebalance_every_weeks,
                    optimizer_max_iterations=(
                        args.retail_edge_optimizer_max_iterations
                    ),
                ),
            )
    if args.models is not None:
        unknown = sorted(set(args.models) - set(models))
        if unknown:
            raise ValueError(
                f"Unknown model(s): {unknown}. Available models: {sorted(models)}"
            )
        models = {name: models[name] for name in args.models}

    rows = []
    weight_frames = []
    diagnostic_rows = []
    backtest_results = {}
    significance_trials = max(args.deflated_sharpe_trials, len(models))
    live_focus = args.dashboard_focus_model or next(iter(models))
    metadata_path = _resolve_asset_metadata_path(args)
    live_tickers = _load_ticker_map(metadata_path) if metadata_path is not None else None

    def save_live_year(year, annual_result):
        start = pd.Timestamp(args.evaluation_start) if args.evaluation_start else None
        valid = annual_result.portfolio_returns.dropna()
        if start is not None:
            valid = valid.loc[valid.index >= start]
        if valid.empty:
            return
        from dataclasses import replace
        annual_results = {}
        for other_name, other in backtest_results.items():
            mask = other.portfolio_returns.index.year == year
            annual_results[other_name] = replace(
                other, portfolio_returns=other.portfolio_returns.loc[mask],
                weights=other.weights.loc[mask], turnover=other.turnover.loc[mask],
                transaction_costs=(other.transaction_costs.loc[mask]
                                   if other.transaction_costs is not None else None))
        annual_results[live_focus] = annual_result
        def year_path(path):
            path = Path(path) if path else None
            return path.with_name(f"{path.stem}_{year}{path.suffix}") if path else None
        paths = _export_dashboard_period(
            annual_results, focus_model=live_focus,
            output_pdf=year_path(args.dashboard_pdf), output_png=year_path(args.dashboard_png),
            title=f"{args.dashboard_title or live_focus} - {year}",
            tc_bps=args.tc_bps, lookback_weeks=config.lookback_weeks,
            rolling_sharpe_years=args.dashboard_rolling_sharpe_years,
            heatmap_assets=args.dashboard_heatmap_assets,
            evaluation_start=args.evaluation_start, ticker_map=live_tickers)
        for kind, path in paths.items():
            print(f"Year {year}: saved dashboard {kind}: {path.resolve()}", flush=True)

    failed_models: list[str] = []
    checkpoint_directory = Path(args.output).parent / "partial"

    def checkpoint(name, result, rows_so_far):
        """Persist one finished model's series and the metric rows so far.

        A walk-forward that dies late is expensive: `comparison.csv` and
        friends are only written after every model finishes, so a failure in
        the last model of a 17-hour run discarded the *other* models'
        completed results too. These checkpoints are written as each model
        finishes, so a later crash costs only the model that crashed.
        """
        try:
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            series = {
                "portfolio_return": result.portfolio_returns,
                "turnover": result.turnover,
            }
            if result.transaction_costs is not None:
                series["transaction_cost"] = result.transaction_costs
            pd.DataFrame(series).to_csv(
                checkpoint_directory / f"{name}_series.csv", index_label="date"
            )
            pd.DataFrame(rows_so_far).set_index("model").to_csv(
                checkpoint_directory / "comparison_partial.csv"
            )
            print(f"checkpoint: saved partial results for {name}", flush=True)
        except Exception as exc:  # noqa: BLE001 - never fail a run over a checkpoint
            print(f"warning: checkpoint for {name} failed: {exc}", flush=True)

    for name, allocator in models.items():
        try:
            result = run_walk_forward(
                returns,
                allocator,
                config,
                progress_every_rebalances=args.progress_every_rebalances,
                progress_label=name,
                year_end_callback=(save_live_year if args.dashboard_every_year
                                   and name == live_focus
                                   and (args.dashboard_pdf or args.dashboard_png) else None),
            )
        except (RuntimeError, ValueError) as exc:
            # One model's walk-forward blowing up should not destroy the models
            # that already finished, nor the ones still queued behind it. Record
            # it, keep going, and let the outputs below reflect what succeeded.
            print(
                f"warning: {name} walk-forward failed, skipping this model: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            failed_models.append(name)
            continue
        backtest_results[name] = result
        score_mask = pd.Series(True, index=result.portfolio_returns.index)
        if args.evaluation_start is not None:
            score_mask = result.portfolio_returns.index >= args.evaluation_start
        metrics = _compute_metrics(
            result.portfolio_returns.loc[score_mask],
            result.turnover.loc[score_mask],
            risk_free_rate=config.risk_free_rate,
            number_of_trials=significance_trials,
        )
        rows.append({"model": name, **metrics})
        checkpoint(name, result, rows)

        if args.weights_output or args.weights_png or args.diagnostics_output:
            # latest_target_weights is a one-shot, single-week transition
            # from result.ending_weights to a fully-constraint-compliant
            # target. That can be genuinely, correctly infeasible (e.g. the
            # live book has drifted enough since the last rebalance that no
            # portfolio satisfies weight/sector/style caps *and* a week's
            # ADV participation budget at once) even when the walk-forward
            # backtest above -- which only ever rebalances from an
            # already-recent, already-compliant starting point -- ran
            # cleanly the whole way through. Letting that raise here would
            # discard every model's already-computed backtest row (rows is
            # only written to comparison.csv after this whole loop ends)
            # and skip every model still queued behind this one, over what
            # is really just an optional bonus output. So: keep this
            # model's backtest metrics (already appended above), skip only
            # its weights/diagnostics row, and move on.
            try:
                target = latest_target_weights(
                    returns,
                    allocator,
                    config,
                    current_weights=result.ending_weights,
                )
            except (RuntimeError, ValueError) as exc:
                print(
                    f"warning: {name} latest_target_weights failed, "
                    f"skipping its weights/diagnostics row: {exc}",
                    flush=True,
                )
                target = None
            if target is not None:
                if args.weights_output or args.weights_png:
                    target = target[target > 1e-12]
                    weight_frames.append(
                        pd.DataFrame(
                            {
                                "date": pd.Timestamp(returns.index[-1]).date().isoformat(),
                                "model": name,
                                "asset": target.index.astype(str),
                                "weight": target.to_numpy(dtype=float),
                            }
                        )
                    )
                diagnostics = getattr(allocator, "last_diagnostics", None)
                if args.diagnostics_output and hasattr(diagnostics, "as_dict"):
                    diagnostic_rows.append(
                        {
                            "date": pd.Timestamp(returns.index[-1]).date().isoformat(),
                            "model": name,
                            **diagnostics.as_dict(),
                        }
                    )
        print(f"completed: {name}", flush=True)

    benchmark_name = args.significance_benchmark
    if benchmark_name in backtest_results:
        benchmark_returns = backtest_results[benchmark_name].portfolio_returns
        if args.evaluation_start is not None:
            benchmark_returns = benchmark_returns.loc[
                benchmark_returns.index >= args.evaluation_start
            ]
        for row in rows:
            model_returns = backtest_results[row["model"]].portfolio_returns
            if args.evaluation_start is not None:
                model_returns = model_returns.loc[
                    model_returns.index >= args.evaluation_start
                ]
            active = model_returns.subtract(benchmark_returns).dropna()
            test = newey_west_mean_test(active)
            row[f"{benchmark_name}_active_return"] = test["annualized_mean"]
            row[f"{benchmark_name}_alpha_hac_t_stat"] = test["hac_t_stat"]
            row[f"{benchmark_name}_alpha_hac_p_value"] = test["hac_p_value"]

    if failed_models:
        print(
            "warning: these models failed and are absent from the outputs: "
            + ", ".join(failed_models)
        )
    if not rows:
        # main() is called bare at module scope, so `return` here would exit 0
        # and a caller scripting this would read the run as a success.
        raise SystemExit("error: every model failed; no comparison to write.")
    comparison = pd.DataFrame(rows).set_index("model")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(destination)
    print(comparison.to_string())
    print(f"saved: {destination.resolve()}")

    if (args.weights_output or args.weights_png) and not weight_frames:
        print(
            "warning: no model produced latest target weights; "
            "skipping weights outputs."
        )
    elif args.weights_output or args.weights_png:
        exported_weights = pd.concat(weight_frames, ignore_index=True)

        metadata_path = _resolve_asset_metadata_path(args)
        if metadata_path is not None:
            exported_weights = _label_weight_frame(
                exported_weights,
                _load_ticker_map(metadata_path),
            )

    if args.weights_output and weight_frames:
        weights_destination = Path(args.weights_output)
        weights_destination.parent.mkdir(parents=True, exist_ok=True)
        exported_weights.to_csv(weights_destination, index=False)
        print(f"saved weights: {weights_destination.resolve()}")

    if args.weights_png and weight_frames:
        png_destination = Path(args.weights_png)
        _save_weights_png(exported_weights, png_destination, args.weights_png_top)
        print(f"saved weights chart: {png_destination.resolve()}")

    if args.diagnostics_output:
        diagnostics_destination = Path(args.diagnostics_output)
        diagnostics_destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diagnostic_rows).to_csv(diagnostics_destination, index=False)
        print(f"saved diagnostics: {diagnostics_destination.resolve()}")

    if args.robustness_output and args.significance_benchmark not in backtest_results:
        # Everything above is already on disk; a missing benchmark should not
        # abort the run before the dashboard is written.
        print(
            "warning: skipping robustness report -- benchmark "
            f"{args.significance_benchmark!r} is not in --models."
        )
    elif args.robustness_output:
        write_robustness_report(
            backtest_results,
            Path(args.robustness_output),
            benchmark=args.significance_benchmark,
            evaluation_start=args.evaluation_start,
            samples=args.robustness_samples,
            block_weeks=args.robustness_block_weeks,
            seed=args.robustness_seed,
        )
        print(f"saved robustness report: {Path(args.robustness_output).resolve()}")

    if args.dashboard_pdf or args.dashboard_png:
        if args.dashboard_focus_model is not None:
            focus_model = args.dashboard_focus_model
        elif args.models:
            focus_model = args.models[0]
        elif "regret_aware_with_overlay" in backtest_results:
            focus_model = "regret_aware_with_overlay"
        else:
            focus_model = next(iter(backtest_results))

        if focus_model not in backtest_results:
            raise ValueError(
                f"--dashboard-focus-model must be one of "
                f"{sorted(backtest_results)}; received {focus_model!r}."
            )

        metadata_path = _resolve_asset_metadata_path(args)
        ticker_map = _load_ticker_map(metadata_path) if metadata_path is not None else None

        dashboard_paths = export_backtest_dashboard(
            backtest_results,
            focus_model=focus_model,
            output_pdf=args.dashboard_pdf,
            output_png=args.dashboard_png,
            title=args.dashboard_title,
            tc_bps=args.tc_bps,
            lookback_weeks=config.lookback_weeks,
            rolling_sharpe_years=args.dashboard_rolling_sharpe_years,
            heatmap_assets=args.dashboard_heatmap_assets,
            evaluation_start=args.evaluation_start,
            ticker_map=ticker_map,
        )
        for format_name, path in dashboard_paths.items():
            print(f"saved dashboard {format_name}: {path.resolve()}")


if __name__ == "__main__":
    main()
