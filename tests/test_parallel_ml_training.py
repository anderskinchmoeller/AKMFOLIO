"""Parallel fits retain serial predictions and publish only complete ensembles."""
from collections import deque
from threading import Barrier

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from akm_hrp.allocators.retail_alpha_ml_mpc import (
    RetailAlphaMLMPCAllocator, RetailAlphaMLMPCConfig, _ML_FEATURES,
)


def training_allocator():
    allocator = object.__new__(RetailAlphaMLMPCAllocator)
    allocator.config = RetailAlphaMLMPCConfig(ml_gbm_n_estimators=12)
    rng = np.random.default_rng(48)
    allocator._ml_history = deque(
        (rng.normal(size=(80, len(_ML_FEATURES))), rng.normal(size=80))
        for _ in range(5)
    )
    return allocator


def test_parallel_predictions_match_serial():
    allocator = training_allocator()
    allocator._refit_models()
    x = np.concatenate([x for x, _ in allocator._ml_history])
    y = np.concatenate([y for _, y in allocator._ml_history])
    cfg = allocator.config
    for model, halflife in (
        (allocator._ml_fast_model, cfg.ml_fast_halflife_rebalances),
        (allocator._ml_slow_model, cfg.ml_slow_halflife_rebalances),
    ):
        serial = GradientBoostingRegressor(**model.get_params()).fit(
            x, y, sample_weight=allocator._recency_weights(np.exp(np.log(.5) / halflife)))
        np.testing.assert_array_equal(model.predict(x), serial.predict(x))
        np.testing.assert_array_equal(model.feature_importances_, serial.feature_importances_)
    serial_linear = Ridge(alpha=cfg.ml_linear_ridge_alpha).fit(x, y)
    np.testing.assert_array_equal(allocator._ml_linear_model.predict(x), serial_linear.predict(x))


def test_fits_overlap_and_failure_preserves_previous_ensemble(monkeypatch):
    allocator = training_allocator()
    allocator._refit_models()
    previous = (allocator._ml_fast_model, allocator._ml_slow_model, allocator._ml_linear_model)
    barrier = Barrier(2)

    def failing_fit(self, x, y, sample_weight):
        # Both fits must be active at once for either worker to proceed.
        barrier.wait(timeout=5)
        raise ValueError("training failed")

    monkeypatch.setattr(GradientBoostingRegressor, "fit", failing_fit)
    with pytest.raises(ValueError, match="training failed"):
        allocator._refit_models()
    assert previous == (allocator._ml_fast_model, allocator._ml_slow_model, allocator._ml_linear_model)
