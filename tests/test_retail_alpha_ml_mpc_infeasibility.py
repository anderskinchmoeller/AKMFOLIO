"""Regression test for an opaque RuntimeError users hit at runtime:

    RuntimeError: Retail Alpha MPC constraints are linearly infeasible;
    solver message=The problem is infeasible. (HiGHS Status 8: ...)

Root cause of the *opacity* (not the infeasibility itself, which can be a
genuine config problem): ``_run_long_engine`` in
``akm_hrp/allocators/retail_alpha_ml_mpc.py`` duplicates the parent
``RetailAlphaMPCAllocator.allocate()`` body (see
``retail_alpha_ml_mpc_phase3_notes.md``, ranked action #2), including its
``linprog``-based feasibility check on the affine constraints. When that
check fails, the original code raised only the raw HiGHS solver message with
no indication of which cap was responsible. The fix names the binding cap
(``max_weight`` x asset count, or ``max_sector_weight`` x represented sector
count) when either alone explains the infeasibility. The identical fix was
also applied to the shared parent ``akm_hrp/allocators/retail_alpha_mpc.py``
(covered by ``tests/test_model_ml_ensemble.py`` via the top-level
``model.py`` allocator).
"""

import numpy as np
import pandas as pd
import pytest

from akm_hrp.allocators.retail_alpha_ml_mpc import (
    RetailAlphaMLMPCAllocator,
    RetailAlphaMLMPCConfig,
)


def test_sector_cap_infeasibility_raises_actionable_message():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-03", periods=120, freq="W-FRI")
    assets = pd.Index([str(30_000 + n) for n in range(25)])
    returns = pd.DataFrame(
        rng.normal(0.0008, 0.012, size=(len(dates), len(assets))),
        index=dates,
        columns=assets,
    )
    core = assets[:15]
    balanced_pit = pd.DataFrame(0, index=dates, columns=assets, dtype=np.int8)
    balanced_pit.loc[:, core] = 1
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
    # Only 3 sectors represented among the selected names; 3 * 25% = 75% < 100%.
    sectors = pd.DataFrame(
        {
            "permno": assets,
            "sec_info_start": "2000-01-01",
            "sec_info_end": "2030-01-01",
            "sector": [f"SECTOR_{n % 3}" for n in range(len(assets))],
        }
    )
    allocator = RetailAlphaMLMPCAllocator(
        balanced_pit,
        structural_features=features,
        sector_history=sectors,
        config=RetailAlphaMLMPCConfig(
            minimum_history_weeks=52,
            risk_lookback_weeks=78,
            ml_max_total_assets=60,
            maximum_added_assets=8,
            max_added_per_sector=3,
            max_weight=0.10,
            max_sector_weight=0.25,
            max_absolute_style_exposure=1.0,
            weekly_cvar_95_limit=0.30,
            portfolio_value=100_000.0,
            planning_horizon=2,
            optimizer_max_iterations=150,
            short_overlay_enabled=False,
            allow_cvar_floor_relaxation=True,
            ml_minimum_training_cross_sections=8,
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        allocator.allocate(returns.iloc[:80])

    message = str(excinfo.value)
    assert "max_sector_weight" in message
    assert "0.7500" in message or "0.75" in message
