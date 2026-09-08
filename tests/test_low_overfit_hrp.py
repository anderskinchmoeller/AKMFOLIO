import numpy as np
import pandas as pd
import pytest

from akm_hrp.allocators.low_overfit_hrp import (
    LowOverfitHRPAllocator,
    LowOverfitHRPConfig,
)


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(101)
    dates = pd.date_range("2019-01-04", periods=130, freq="W-FRI")
    values = rng.normal(0.001, 0.02, size=(len(dates), 12))
    # Give the first asset an economically visible, but not fitted, trend.
    values[-52:, 0] += 0.006
    return pd.DataFrame(
        values,
        index=dates,
        columns=[f"A{asset:02d}" for asset in range(12)],
    )


def test_low_overfit_hrp_is_long_only_bounded_and_tilted() -> None:
    returns = _returns()
    tilted = LowOverfitHRPAllocator().allocate(returns)
    untilted = LowOverfitHRPAllocator(LowOverfitHRPConfig(momentum_tilt=0.0)).allocate(
        returns
    )

    assert tilted.sum() == pytest.approx(1.0)
    assert (tilted >= 0.0).all()
    assert tilted.max() <= 0.10 + 1e-12
    assert not np.allclose(tilted.sort_index(), untilted.sort_index())


def test_low_overfit_hrp_freezes_target_within_calendar_month() -> None:
    returns = _returns()
    adjacent = next(
        position
        for position in range(105, len(returns))
        if returns.index[position - 1].to_period("M")
        == returns.index[position].to_period("M")
    )
    allocator = LowOverfitHRPAllocator()
    first = allocator.allocate(returns.iloc[:adjacent])
    second = allocator.allocate(returns.iloc[: adjacent + 1])

    pd.testing.assert_series_equal(first.sort_index(), second.sort_index())
    assert allocator.last_diagnostics.frozen_by_monthly_schedule
