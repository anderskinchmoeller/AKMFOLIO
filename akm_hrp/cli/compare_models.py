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
from akm_hrp.diagnostics.dashboard import export_backtest_dashboard
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare robust HRP with institutional benchmark allocators."
    )
    parser.add_argument("--returns", default="weekly_returns.csv")
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


def main() -> None:
    args = parse_args()
    if args.rebalance_every_weeks < 1:
        raise ValueError("--rebalance-every-weeks must be at least 1.")
    if args.retail_optimizer_max_iterations < 1:
        raise ValueError("--retail-optimizer-max-iterations must be at least 1.")
    if args.retail_alpha_ml_max_total_assets < 2:
        raise ValueError("--retail-alpha-ml-max-total-assets must be at least 2.")
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
                    **(
                        {
                            "ml_max_training_cross_sections": (
                                args.retail_alpha_ml_max_training_cross_sections
                            )
                        }
                        if args.retail_alpha_ml_max_training_cross_sections
                        is not None
                        else {}
                    ),
                    allow_cvar_floor_relaxation=(
                        args.retail_alpha_ml_allow_cvar_floor_relaxation
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
                    **(
                        {
                            "ml_max_training_cross_sections": (
                                args.retail_alpha_ml_max_training_cross_sections
                            )
                        }
                        if args.retail_alpha_ml_max_training_cross_sections
                        is not None
                        else {}
                    ),
                    allow_cvar_floor_relaxation=(
                        args.retail_alpha_ml_allow_cvar_floor_relaxation
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
                    **(
                        {
                            "ml_max_training_cross_sections": (
                                args.retail_alpha_ml_max_training_cross_sections
                            )
                        }
                        if args.retail_alpha_ml_max_training_cross_sections
                        is not None
                        else {}
                    ),
                    allow_cvar_floor_relaxation=(
                        args.retail_alpha_ml_allow_cvar_floor_relaxation
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
    for name, allocator in models.items():
        result = run_walk_forward(
            returns,
            allocator,
            config,
            progress_every_rebalances=args.progress_every_rebalances,
            progress_label=name,
        )
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

        if args.weights_output or args.weights_png or args.diagnostics_output:
            target = latest_target_weights(
                returns,
                allocator,
                config,
                current_weights=result.ending_weights,
            )
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

    comparison = pd.DataFrame(rows).set_index("model")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(destination)
    print(comparison.to_string())
    print(f"saved: {destination.resolve()}")

    if args.weights_output or args.weights_png:
        exported_weights = pd.concat(weight_frames, ignore_index=True)

        metadata_path = (
            Path(args.asset_metadata)
            if args.asset_metadata
            else Path(args.returns).resolve().parent / "crsp_security_metadata.csv"
        )
        if metadata_path.exists():
            exported_weights = _label_weight_frame(
                exported_weights,
                _load_ticker_map(metadata_path),
            )
        else:
            print(
                f"warning: no asset metadata found at {metadata_path}; "
                "asset values remain source identifiers",
                flush=True,
            )

    if args.weights_output:
        weights_destination = Path(args.weights_output)
        weights_destination.parent.mkdir(parents=True, exist_ok=True)
        exported_weights.to_csv(weights_destination, index=False)
        print(f"saved weights: {weights_destination.resolve()}")

    if args.weights_png:
        png_destination = Path(args.weights_png)
        _save_weights_png(exported_weights, png_destination, args.weights_png_top)
        print(f"saved weights chart: {png_destination.resolve()}")

    if args.diagnostics_output:
        diagnostics_destination = Path(args.diagnostics_output)
        diagnostics_destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diagnostic_rows).to_csv(diagnostics_destination, index=False)
        print(f"saved diagnostics: {diagnostics_destination.resolve()}")

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

        metadata_path = (
            Path(args.asset_metadata)
            if args.asset_metadata
            else Path(args.returns).resolve().parent / "crsp_security_metadata.csv"
        )
        ticker_map = _load_ticker_map(metadata_path) if metadata_path.exists() else None
        if ticker_map is None:
            print(
                f"warning: no asset metadata found at {metadata_path}; "
                "dashboard heatmap labels remain source identifiers",
                flush=True,
            )

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
