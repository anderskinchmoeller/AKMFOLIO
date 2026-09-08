"""The selected-book equal-weight control.

Two different 1/N benchmarks answer two different questions, and conflating
them is how a model gets credited for work its optimizer did not do:

  `equal_weight` in compare_models.py  = 1/N over the whole eligible universe
      -> "does the model beat naive diversification?"
  `ml_equal_weight_benchmark=True`     = 1/N over the model's OWN selected book
      -> "does the optimizer beat naive weighting of the model's own picks?"

The second holds selection fixed, so the difference isolates what the risk
model, alpha and Kelly sizing are worth. DeMiguel-Garlappi-Uppal is the reason
it matters: 1/N is a punishing benchmark, and a model that cannot clear its own
selected-book 1/N is not being helped by its optimizer.
"""

import numpy as np
import pandas as pd

from akm_hrp.allocators.retail_alpha_ml_mpc import (
    RetailAlphaMLMPCAllocator,
    RetailAlphaMLMPCConfig,
)

MAX_WEIGHT = 0.06


def build_universe():
    """Build a deterministic, sufficiently broad universe for the benchmark."""

    rng = np.random.default_rng(744)
    dates = pd.date_range("2023-01-06", periods=82, freq="W-FRI")
    assets = pd.Index([str(20_000 + number) for number in range(30)])
    market = rng.normal(0.001, 0.011, size=(len(dates), 1))
    returns = pd.DataFrame(
        market + rng.normal(0.0, 0.012, size=(len(dates), len(assets))),
        index=dates,
        columns=assets,
    )
    returns.iloc[-52:, 15:24] += np.linspace(0.0002, 0.003, 9)

    pit = pd.DataFrame(0, index=dates, columns=assets, dtype=np.int8)
    pit.loc[:, assets[:15]] = 1
    features = pd.DataFrame(
        {
            "formation_date": dates[-2],
            "asset": assets,
            "market_cap_usd": np.geomspace(2e8, 2e10, len(assets)),
            "price": 30.0,
            "dollar_volume_20d": np.geomspace(5e6, 80e6, len(assets)),
            "amihud_20d": np.geomspace(0.02, 0.0002, len(assets)),
            "zero_return_fraction_20d": np.linspace(0.08, 0.0, len(assets)),
            "book_to_market": np.linspace(0.2, 1.1, len(assets)),
            "gross_profitability": np.linspace(0.08, 0.45, len(assets)),
            "return_on_assets": np.linspace(0.01, 0.16, len(assets)),
            "accruals_to_assets": np.linspace(0.08, -0.05, len(assets)),
            "standardized_unexpected_earnings": np.linspace(-1.5, 1.5, len(assets)),
            "fundamental_momentum": np.linspace(-0.08, 0.10, len(assets)),
            "shareholder_carry": np.linspace(-0.02, 0.06, len(assets)),
        }
    )
    sectors = pd.DataFrame(
        {
            "permno": assets,
            "sec_info_start": "2000-01-01",
            "sec_info_end": "2030-01-01",
            "sector": [f"SECTOR_{number % 5}" for number in range(len(assets))],
        }
    )
    return returns, pit, features, sectors, assets


def _run(equal_weight: bool):
    returns, pit, features, sectors, assets = build_universe()
    config = RetailAlphaMLMPCConfig(
        max_weight=MAX_WEIGHT,
        minimum_history_weeks=52,
        risk_lookback_weeks=78,
        ml_max_total_assets=60,
        maximum_added_assets=30,
        max_added_per_sector=6,
        max_sector_weight=0.40,
        max_absolute_style_exposure=1.0,
        weekly_cvar_95_limit=0.30,
        portfolio_value=100_000.0,
        planning_horizon=2,
        optimizer_max_iterations=150,
        short_overlay_enabled=False,
        allow_cvar_floor_relaxation=True,
        ml_equal_weight_benchmark=equal_weight,
    )
    allocator = RetailAlphaMLMPCAllocator(
        pit, structural_features=features, sector_history=sectors, config=config
    )
    return allocator, allocator.allocate(returns)


def test_equal_weight_benchmark_is_a_valid_portfolio():
    _, weights = _run(True)
    held = weights[weights > 1e-9]
    assert abs(float(weights.sum()) - 1.0) < 1e-6, "must be fully invested"
    assert (weights >= -1e-12).all(), "must be long-only"
    assert float(weights.max()) <= MAX_WEIGHT + 1e-9, "must respect max_weight"
    assert not weights.isna().any(), "no NaN weights"
    print(f"EW book: {len(held)} names, sum={weights.sum():.6f}, "
          f"max={held.max():.4f}, min={held.min():.4f}")
    print("equal-weight benchmark produces a valid portfolio: OK")


def test_weights_are_actually_equal():
    """If the cap and exit schedule do not bind, every held name is 1/N."""
    _, weights = _run(True)
    held = weights[weights > 1e-9]
    distinct = held.round(8).nunique()
    print(f"distinct weight levels among {len(held)} held names: {distinct}")
    assert distinct == 1, (
        f"expected genuinely equal weights, found {distinct} distinct levels"
    )
    assert abs(float(held.iloc[0]) - 1.0 / len(held)) < 1e-9, "each name must be 1/N"
    print("every held name carries exactly 1/N: OK")


def test_benchmark_reuses_selection_but_differs_in_sizing():
    """Same picks, different weights -- that is what makes it a clean control."""
    opt_allocator, opt = _run(False)
    ew_allocator, ew = _run(True)

    opt_held = set(opt[opt > 1e-9].index)
    ew_held = set(ew[ew > 1e-9].index)
    overlap = len(opt_held & ew_held) / max(len(opt_held | ew_held), 1)
    turnover_between = float((ew - opt.reindex(ew.index).fillna(0.0)).abs().sum())

    print(f"selection overlap={overlap:.1%}, L1 weight difference={turnover_between:.4f}")
    assert overlap > 0.8, (
        "the control must reuse the model's selection, not pick different names"
    )
    assert turnover_between > 0.05, (
        "the control must differ materially in sizing, else it is not a control"
    )
    print("benchmark holds selection fixed and varies only sizing: OK")


def test_benchmark_is_at_least_as_diversified():
    _, opt = _run(False)
    _, ew = _run(True)
    eff_opt = 1.0 / float(np.sum(opt.to_numpy() ** 2))
    eff_ew = 1.0 / float(np.sum(ew.to_numpy() ** 2))
    print(f"effective N: optimizer={eff_opt:.1f}, equal weight={eff_ew:.1f}")
    assert eff_ew >= eff_opt - 1e-9, (
        "equal weighting cannot be less diversified than the optimized book"
    )
    print("equal weighting is the more diversified book, as it must be: OK")


def test_default_is_off():
    """Nothing changes unless the benchmark is deliberately switched on."""
    assert RetailAlphaMLMPCConfig().ml_equal_weight_benchmark is False
    print("benchmark defaults to off: OK")


if __name__ == "__main__":
    test_default_is_off()
    test_equal_weight_benchmark_is_a_valid_portfolio()
    test_weights_are_actually_equal()
    test_benchmark_reuses_selection_but_differs_in_sizing()
    test_benchmark_is_at_least_as_diversified()
    print("\nALL EQUAL-WEIGHT BENCHMARK TESTS PASSED")
