"""The fast GBM's sample weights must not span an unbounded dynamic range.

Recency weights decay geometrically with cross-section age. Left untruncated
over the default 252-cross-section buffer, the fast halflife of 6 rebalances
drives the oldest rows to ~2.5e-13 of the newest. Rows that small contribute
nothing to the fit but stretch `sample_weight` far enough that sklearn's
weighted impurity accumulation underflows in the later boosting stages: every
tree reports zero total importance, `feature_importances_` divides 0 by 0, and
the published diagnostic silently becomes all-NaN. Observed live at rebalance
317 of a full-scale walk-forward, once the buffer first saturated.

`ml_recency_weight_floor` bounds that ratio by construction. These tests pin
the bound rather than the underflow itself: reproducing the NaN needs a full
100-estimator fit over ~200k rows, since it only surfaces once boosting
residuals shrink -- far too slow for a unit test, and sensitive to the sklearn
version's tree internals. The invariant is what prevents it, so the invariant
is what is asserted.
"""
from collections import deque

import numpy as np

from akm_hrp.allocators.retail_alpha_ml_mpc import (
    RetailAlphaMLMPCAllocator,
    RetailAlphaMLMPCConfig,
    _ML_FEATURES,
)

ROWS = 40


def allocator_with_buffer(cross_sections, **config_kwargs):
    allocator = object.__new__(RetailAlphaMLMPCAllocator)
    allocator.config = RetailAlphaMLMPCConfig(
        ml_gbm_n_estimators=12, **config_kwargs
    )
    rng = np.random.default_rng(11)
    allocator._ml_history = deque(
        (
            rng.normal(size=(ROWS, len(_ML_FEATURES))),
            rng.normal(size=ROWS),
        )
        for _ in range(cross_sections)
    )
    return allocator


def decay_for(halflife):
    return float(np.exp(np.log(0.5) / halflife))


def test_saturated_buffer_bounds_the_fast_weight_ratio():
    allocator = allocator_with_buffer(252)
    _, _, weights = allocator._training_view(
        decay_for(allocator.config.ml_fast_halflife_rebalances)
    )
    assert weights.max() == 1.0
    assert weights.min() >= allocator.config.ml_recency_weight_floor


def test_untruncated_weights_would_have_been_far_below_the_floor():
    """Guards the premise: without truncation the ratio really is extreme."""
    allocator = allocator_with_buffer(252)
    full = allocator._recency_weights(
        decay_for(allocator.config.ml_fast_halflife_rebalances)
    )
    assert full.min() < 1e-12
    assert full.min() < allocator.config.ml_recency_weight_floor


def test_fast_model_trains_on_a_truncated_tail_of_the_buffer():
    allocator = allocator_with_buffer(252)
    horizon = allocator._recency_horizon(
        decay_for(allocator.config.ml_fast_halflife_rebalances)
    )
    assert horizon == 120
    x, y, weights = allocator._training_view(
        decay_for(allocator.config.ml_fast_halflife_rebalances)
    )
    assert len(x) == len(y) == len(weights) == horizon * ROWS
    # The tail kept is the most recent one, not an arbitrary slice.
    newest = list(allocator._ml_history)[-1][0]
    np.testing.assert_array_equal(x[-ROWS:], newest)


def test_slow_model_still_sees_the_whole_buffer():
    """Halflife 24 bottoms out at ~7e-4 over 252 cross-sections -- no
    truncation, so the slow vote's behaviour is unchanged by the fix."""
    allocator = allocator_with_buffer(252)
    decay = decay_for(allocator.config.ml_slow_halflife_rebalances)
    assert allocator._recency_horizon(decay) == 252
    _, _, weights = allocator._training_view(decay)
    assert weights.min() > allocator.config.ml_recency_weight_floor


def test_short_buffer_is_never_truncated():
    allocator = allocator_with_buffer(5)
    for halflife in (
        allocator.config.ml_fast_halflife_rebalances,
        allocator.config.ml_slow_halflife_rebalances,
    ):
        decay = decay_for(halflife)
        assert allocator._recency_horizon(decay) == 5
        np.testing.assert_array_equal(
            allocator._training_view(decay)[2],
            allocator._recency_weights(decay),
        )


def test_floor_is_configurable_and_disengageable():
    tight = allocator_with_buffer(252, ml_recency_weight_floor=1e-3)
    loose = allocator_with_buffer(252, ml_recency_weight_floor=1e-12)
    decay = decay_for(tight.config.ml_fast_halflife_rebalances)
    assert tight._recency_horizon(decay) < loose._recency_horizon(decay) <= 252
    # A floor outside (0, 1) is treated as "no truncation" rather than an
    # error, so an operator cannot accidentally empty the training set.
    off = allocator_with_buffer(252, ml_recency_weight_floor=0.0)
    assert off._recency_horizon(decay) == 252


def test_refit_publishes_finite_feature_importances():
    allocator = allocator_with_buffer(252)
    allocator._refit_models()
    importances = allocator.last_ml_feature_importances
    assert np.isfinite(importances.to_numpy()).all()
    assert list(importances.columns) == ["fast", "slow"]
    assert list(importances.index) == list(_ML_FEATURES)
