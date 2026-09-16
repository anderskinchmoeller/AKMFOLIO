"""--max-rebalance-turnover: per-rebalance L1 cap exposed on the CLI."""

import argparse

import pandas as pd
import pytest

from akm_hrp.backtest.engine import _cap_l1_turnover
from akm_hrp.cli.compare_models import _parse_turnover_cap
from akm_hrp.config import HRPConfig


def test_parse_values():
    assert _parse_turnover_cap("default") == "default"
    assert _parse_turnover_cap("none") is None
    assert _parse_turnover_cap("None") is None
    assert _parse_turnover_cap("2.0") == 2.0
    assert _parse_turnover_cap("0") == 0.0


@pytest.mark.parametrize("value", ["-0.1", "inf", "nan"])
def test_parse_rejects_bad_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_turnover_cap(value)


def test_default_cap_unchanged():
    assert HRPConfig().max_rebalance_turnover_l1 == 0.10


def test_full_book_replacement_passes_at_cap_two():
    current = pd.Series([0.5, 0.5, 0.0, 0.0], index=list("abcd"))
    target = pd.Series([0.0, 0.0, 0.5, 0.5], index=list("abcd"))
    assert _cap_l1_turnover(current, target, 2.0).equals(target)
    capped = _cap_l1_turnover(current, target, 0.10)
    assert float((capped - current).abs().sum()) == pytest.approx(0.10)
