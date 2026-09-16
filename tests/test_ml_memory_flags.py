"""ML memory settings are counted in rebalances, so the cadence rescales them.

`ml_fast_halflife_rebalances=6` means ~6 weeks at weekly cadence and ~18 months
at 13-week cadence; the fast/slow pair (6 vs 24) goes from 6-vs-24 weeks to
18-months-vs-6-years. Before these flags existed the only way to change that was
to edit the dataclass, so a cadence change silently redefined what the model
remembers.

No automatic rescaling is offered, deliberately: preserving the weekly calendar
horizons at 13w puts both tree halflives below one rebalance, which is
degenerate. The validation here only rejects combinations that are incoherent at
any cadence.
"""
import argparse

import pytest

from akm_hrp.allocators.retail_alpha_ml_mpc import RetailAlphaMLMPCConfig
from akm_hrp.cli.compare_models import _ml_config_overrides

FLAGS = {
    "retail_alpha_ml_max_weight": None,
    "retail_alpha_ml_min_weight": None,
    "retail_alpha_ml_max_total_assets": None,
    "retail_alpha_ml_max_training_cross_sections": None,
    "retail_alpha_ml_min_training_cross_sections": None,
    "retail_alpha_ml_fast_halflife": None,
    "retail_alpha_ml_slow_halflife": None,
    "retail_alpha_ml_retrain_every": None,
    "retail_alpha_ml_ic_halflife": None,
}


def args(**overrides):
    unknown = set(overrides) - set(FLAGS)
    assert not unknown, f"test typo: {unknown}"
    return argparse.Namespace(**{**FLAGS, **overrides})


def test_nothing_set_overrides_nothing():
    assert _ml_config_overrides(args()) == {}


def test_only_explicit_settings_are_returned():
    out = _ml_config_overrides(args(retail_alpha_ml_fast_halflife=2.0))
    assert out == {"ml_fast_halflife_rebalances": 2.0}


def test_a_quarterly_cadence_profile_is_accepted():
    """fast 2 / slow 8 keeps the 1:4 ratio at ~6 months vs ~2 years at 13w."""
    out = _ml_config_overrides(args(
        retail_alpha_ml_fast_halflife=2.0,
        retail_alpha_ml_slow_halflife=8.0,
        retail_alpha_ml_retrain_every=1,
        retail_alpha_ml_ic_halflife=4.0,
    ))
    assert out == {
        "ml_fast_halflife_rebalances": 2.0,
        "ml_slow_halflife_rebalances": 8.0,
        "ml_retrain_every_n_rebalances": 1,
        "ic_halflife_rebalances": 4.0,
    }


def test_every_key_matches_a_real_config_field():
    out = _ml_config_overrides(args(
        retail_alpha_ml_max_training_cross_sections=80,
        retail_alpha_ml_min_training_cross_sections=6,
        retail_alpha_ml_fast_halflife=2.0,
        retail_alpha_ml_slow_halflife=8.0,
        retail_alpha_ml_retrain_every=2,
        retail_alpha_ml_ic_halflife=4.0,
    ))
    config = RetailAlphaMLMPCConfig(**out)
    for field, value in out.items():
        assert getattr(config, field) == value


@pytest.mark.parametrize("fast,slow", [(24.0, 24.0), (30.0, 24.0)])
def test_fast_must_be_faster_than_slow(fast, slow):
    with pytest.raises(ValueError, match="strictly less"):
        _ml_config_overrides(args(
            retail_alpha_ml_fast_halflife=fast,
            retail_alpha_ml_slow_halflife=slow,
        ))


def test_override_is_checked_against_the_other_sides_default():
    """Setting only fast=30 is invalid because slow defaults to 24."""
    with pytest.raises(ValueError, match="strictly less"):
        _ml_config_overrides(args(retail_alpha_ml_fast_halflife=30.0))
    # ...and setting only slow=4 is invalid because fast defaults to 6.
    with pytest.raises(ValueError, match="strictly less"):
        _ml_config_overrides(args(retail_alpha_ml_slow_halflife=4.0))


@pytest.mark.parametrize("field", [
    "retail_alpha_ml_fast_halflife",
    "retail_alpha_ml_ic_halflife",
])
def test_halflives_must_be_positive(field):
    with pytest.raises(ValueError, match="positive"):
        _ml_config_overrides(args(**{field: 0.0}))


def test_retrain_interval_must_be_at_least_one_rebalance():
    with pytest.raises(ValueError, match="retrain interval"):
        _ml_config_overrides(args(retail_alpha_ml_retrain_every=0))


def test_minimum_above_maximum_would_never_train():
    with pytest.raises(ValueError, match="exceeds the"):
        _ml_config_overrides(args(
            retail_alpha_ml_min_training_cross_sections=300,
            retail_alpha_ml_max_training_cross_sections=252,
        ))


def test_minimum_is_checked_against_a_lowered_maximum():
    """The pair is validated jointly, not each against its own default."""
    with pytest.raises(ValueError, match="exceeds the"):
        _ml_config_overrides(args(retail_alpha_ml_max_training_cross_sections=6))


def test_minimum_must_be_positive():
    with pytest.raises(ValueError, match="must be >= 1"):
        _ml_config_overrides(args(retail_alpha_ml_min_training_cross_sections=0))


# --- per-name position cap -------------------------------------------------

def test_max_weight_defaults_to_the_config_and_is_not_emitted():
    assert RetailAlphaMLMPCConfig().max_weight == 0.03
    assert "max_weight" not in _ml_config_overrides(args())


def test_removing_the_cap_is_expressible():
    out = _ml_config_overrides(args(retail_alpha_ml_max_weight=1.0))
    assert out == {"max_weight": 1.0}
    assert RetailAlphaMLMPCConfig(**out).max_weight == 1.0


@pytest.mark.parametrize("value", [0.0, -0.01, 1.5])
def test_max_weight_must_be_a_valid_fraction(value):
    with pytest.raises(ValueError, match="max weight must be in"):
        _ml_config_overrides(args(retail_alpha_ml_max_weight=value))


def test_cap_too_tight_for_the_asset_budget_is_rejected():
    """60 names x 1% caps the book at 60% of capital -- never fully invested."""
    with pytest.raises(ValueError, match="never be fully invested"):
        _ml_config_overrides(args(
            retail_alpha_ml_max_weight=0.01,
            retail_alpha_ml_max_total_assets=60,
        ))


def test_cap_is_checked_against_the_supplied_budget_not_just_the_default():
    # 0.02 x 60 = 1.2 -> fine; 0.02 x 40 = 0.8 -> not.
    _ml_config_overrides(args(retail_alpha_ml_max_weight=0.02,
                              retail_alpha_ml_max_total_assets=60))
    with pytest.raises(ValueError, match="never be fully invested"):
        _ml_config_overrides(args(retail_alpha_ml_max_weight=0.02,
                                  retail_alpha_ml_max_total_assets=40))


def test_floor_above_cap_is_rejected():
    with pytest.raises(ValueError, match="exceeds max weight"):
        _ml_config_overrides(args(retail_alpha_ml_max_weight=0.03,
                                  retail_alpha_ml_min_weight=0.05))


def test_floor_unreachable_at_the_asset_budget_is_rejected():
    """120 names x a 1% floor needs 120% of capital; the budget would be inert."""
    with pytest.raises(ValueError, match="inert"):
        _ml_config_overrides(args(
            retail_alpha_ml_min_weight=0.01,
            retail_alpha_ml_max_total_assets=120,
            retail_alpha_ml_max_weight=1.0,
        ))


def test_the_matching_floor_for_a_120_name_budget_is_accepted():
    out = _ml_config_overrides(args(
        retail_alpha_ml_min_weight=1.0 / 120,
        retail_alpha_ml_max_total_assets=120,
        retail_alpha_ml_max_weight=1.0,
    ))
    assert out == {"max_weight": 1.0}


def test_the_current_60_name_book_with_a_1pct_floor_still_passes():
    _ml_config_overrides(args(
        retail_alpha_ml_min_weight=0.01,
        retail_alpha_ml_max_total_assets=60,
        retail_alpha_ml_max_weight=1.0,
    ))
