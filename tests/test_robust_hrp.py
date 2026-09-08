import numpy as np
import pandas as pd

from akm_hrp.allocators.hrp_overlay_allocator import (
    AllocatorConfig,
    HRPOverlayAllocator,
)


def test_regret_aware_allocator_is_bounded_and_diagnostic():
    rng = np.random.default_rng(17)
    market = rng.normal(0.0, 0.01, 180)
    values = np.column_stack(
        [0.55 * market + rng.normal(0.0, 0.012, 180) for _ in range(12)]
    )
    returns = pd.DataFrame(
        values,
        index=pd.date_range("2020-01-03", periods=180, freq="W-FRI"),
        columns=[str(10000 + i) for i in range(12)],
    )
    allocator = HRPOverlayAllocator(
        AllocatorConfig(
            max_weight=0.15,
            use_signal_budget=False,
            use_cov_inverse=False,
        )
    )

    weights = allocator.allocate(returns)

    assert np.isclose(weights.sum(), 1.0)
    assert (weights >= 0.0).all()
    assert weights.max() <= 0.15 + 1e-9
    assert allocator.last_robust_diagnostics is not None
    assert allocator.last_robust_diagnostics.scenario_count == 5
    assert allocator.last_robust_diagnostics.split_count == len(weights) - 1
