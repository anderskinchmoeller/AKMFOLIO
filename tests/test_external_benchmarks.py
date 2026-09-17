import numpy as np
import pandas as pd
import pytest

from akm_hrp.backtest.engine import _compute_metrics
from akm_hrp.data.external_benchmarks import (
    benchmark_backtest_result,
    load_benchmark_returns,
    weekly_returns_from_prices,
)


def test_weekly_returns_compound_daily_closes_to_friday():
    days = pd.bdate_range("2024-01-01", "2024-01-19")
    prices = pd.Series(np.linspace(100, 114, len(days)), index=days)
    weekly = weekly_returns_from_prices(prices)
    assert list(weekly.index) == [pd.Timestamp("2024-01-12"), pd.Timestamp("2024-01-19")]
    assert weekly.iloc[0] == pytest.approx(prices["2024-01-12"] / prices["2024-01-05"] - 1)
    # Holiday Friday: the week is labelled Friday and uses Thursday's close.
    prices = prices.drop(pd.Timestamp("2024-01-19"))
    assert weekly_returns_from_prices(prices).index[-1] == pd.Timestamp("2024-01-19")


def test_benchmark_result_is_nan_before_inception_and_buy_and_hold(tmp_path):
    index = pd.date_range("2010-01-01", periods=200, freq="W-FRI")
    series = pd.Series(0.002, index=index[60:190]).drop(index[100])
    path = tmp_path / "URTH_weekly_returns.csv"
    series.to_frame("return").to_csv(path, index_label="date")
    loaded = load_benchmark_returns("urth", tmp_path, fetch_if_missing=False)
    result = benchmark_backtest_result("URTH", loaded, index)
    r = result.portfolio_returns
    assert r.iloc[:60].isna().all()
    assert r.iloc[190:].isna().all()          # after the cache ends: unknown
    assert r.loc[index[100]] == 0.0            # closed week inside life: flat
    assert (result.turnover == 0).all() and (result.transaction_costs == 0).all()
    assert result.weights["URTH"].iloc[60] == 1.0
    metrics = _compute_metrics(r, result.turnover)
    assert metrics["n_obs"] == 130
    with pytest.raises(FileNotFoundError):
        load_benchmark_returns("ACWI", tmp_path, fetch_if_missing=False)


def test_benchmark_without_overlap_is_rejected():
    index = pd.date_range("2010-01-01", periods=20, freq="W-FRI")
    series = pd.Series(0.01, index=pd.date_range("2020-01-03", periods=5, freq="W-FRI"))
    with pytest.raises(ValueError):
        benchmark_backtest_result("URTH", series, index)
