"""A signal with no inputs must say so, out loud, once per run.

Zero coverage produces weight 0.0 -- which is exactly what a signal the IC
machinery has decided against also gets. Because the two are indistinguishable
in every output the model writes, four of nine signals sat dead for this
model's entire history and surfaced only when someone asked why the book
tracked its benchmark so closely. See
`four_signals_zero_coverage_missing_compustat_features.md`.
"""
import logging

import pandas as pd
import pytest

from akm_hrp.allocators.retail_alpha_ml_mpc import (
    RetailAlphaMLMPCAllocator,
    RetailAlphaMLMPCConfig,
)

AS_OF = pd.Timestamp("2015-07-03")


def allocator(**config_kwargs):
    a = object.__new__(RetailAlphaMLMPCAllocator)
    a.config = RetailAlphaMLMPCConfig(**config_kwargs)
    a._reported_unusable_signals = set()
    return a


def coverage(**values):
    return pd.Series(values, dtype=float)


def test_zero_coverage_is_reported_as_missing_data(caplog):
    a = allocator()
    with caplog.at_level(logging.WARNING):
        a._report_unusable_signals(
            coverage(momentum_12_1=0.99, carry_quality_tilt=0.0), AS_OF
        )
    text = caplog.text
    assert "carry_quality_tilt" in text
    assert "ZERO coverage" in text
    assert "missing data, not a rejected signal" in text
    # a healthy signal is not mentioned
    assert "momentum_12_1" not in text


def test_each_signal_is_named_once_per_run(caplog):
    a = allocator()
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            a._report_unusable_signals(coverage(carry_quality_tilt=0.0), AS_OF)
    assert caplog.text.count("carry_quality_tilt") == 1


def test_low_but_nonzero_coverage_is_reported_differently(caplog):
    """Below the floor is a deliberate configuration choice, not missing data."""
    a = allocator(minimum_signal_coverage=0.30)
    with caplog.at_level(logging.WARNING):
        a._report_unusable_signals(coverage(quality_value_carry=0.12), AS_OF)
    text = caplog.text
    assert "quality_value_carry" in text
    assert "12.0%" in text and "30.0%" in text
    assert "ZERO coverage" not in text


def test_coverage_at_or_above_the_floor_is_silent(caplog):
    a = allocator(minimum_signal_coverage=0.30)
    with caplog.at_level(logging.WARNING):
        a._report_unusable_signals(
            coverage(a=0.30, b=0.31, c=1.0), AS_OF
        )
    assert caplog.text == ""


def test_the_real_dead_set_is_all_reported(caplog):
    """The four signals that were actually dead, plus the one still dead."""
    a = allocator()
    dead = ["post_earnings_drift", "quality_value_carry",
            "carry_quality_tilt", "robust_fundamental_carry"]
    with caplog.at_level(logging.WARNING):
        a._report_unusable_signals(
            coverage(**{s: 0.0 for s in dead}, momentum_12_1=0.999), AS_OF
        )
    for s in dead:
        assert s in caplog.text
    assert caplog.text.count("ZERO coverage") == 4


@pytest.mark.parametrize("empty", [pd.Series(dtype=float), None])
def test_empty_coverage_is_not_an_error(empty, caplog):
    with caplog.at_level(logging.WARNING):
        allocator()._report_unusable_signals(empty, AS_OF)
    assert caplog.text == ""
