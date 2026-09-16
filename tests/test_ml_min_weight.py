"""Configurable per-name floor for retail_alpha_ml_mpc (`ml_min_weight`)."""

import pandas as pd
import pytest

from akm_hrp.allocators.retail_alpha_ml_mpc import (
    RetailAlphaMLMPCConfig,
    _apply_min_weight_floor,
)
from akm_hrp.cli import compare_models


def test_default_keeps_coupled_floor():
    assert RetailAlphaMLMPCConfig().ml_min_weight is None


def test_one_percent_floor_prunes_dust_and_stays_invested():
    weights = pd.Series([0.005, 0.012] + [0.983 / 40] * 40)
    upper = pd.Series(0.03, index=weights.index)
    out = _apply_min_weight_floor(weights, 0.01, upper, pd.Index([]))
    assert out.iloc[0] == 0.0
    held = out[out > 0]
    assert held.min() >= 0.01 - 1e-12
    assert out.iloc[1] >= 0.012 - 1e-12
    assert out.sum() == pytest.approx(1.0)


def test_protected_name_below_floor_is_kept():
    weights = pd.Series([0.005] + [0.995 / 40] * 40)
    upper = pd.Series(0.03, index=weights.index)
    out = _apply_min_weight_floor(weights, 0.01, upper, pd.Index([0]))
    assert out.iloc[0] == pytest.approx(0.005)


@pytest.mark.parametrize("value", ["0", "-0.01", "0.02"])
def test_cli_rejects_out_of_range_floor(monkeypatch, value):
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_models",
            "--retail-alpha-ml-max-total-assets",
            "60",
            "--retail-alpha-ml-min-weight",
            value,
        ],
    )
    with pytest.raises(ValueError, match="min-weight"):
        compare_models.main()
