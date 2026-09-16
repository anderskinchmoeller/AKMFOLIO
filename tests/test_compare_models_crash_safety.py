"""A model that blows up mid-run must not take the other models down with it.

`comparison.csv`, the weights and the diagnostics are all written after every
model has finished, so before this a single late failure discarded results that
had already been computed. That is not hypothetical: a 17-hour full-scale
walk-forward died on its second model at rebalance 1,776 of 1,809 and took the
already-complete equal_weight metrics with it, leaving nothing but the log.

So: a failing model is reported and skipped, the survivors still reach disk,
and each model's series is checkpointed as it finishes.
"""
import numpy as np
import pandas as pd
import pytest

from akm_hrp.backtest.engine import BacktestResult
from akm_hrp.cli import compare_models

ASSETS = [f"A{i:02d}" for i in range(12)]


@pytest.fixture
def returns_csv(tmp_path):
    index = pd.date_range("2015-01-02", periods=160, freq="W-FRI")
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        rng.normal(0.001, 0.02, size=(len(index), len(ASSETS))),
        index=index,
        columns=ASSETS,
    )
    path = tmp_path / "weekly_returns.csv"
    frame.to_csv(path, index_label="date")
    return path, index


def fake_result(index):
    rng = np.random.default_rng(7)
    weights = pd.DataFrame(1.0 / len(ASSETS), index=index, columns=ASSETS)
    return BacktestResult(
        weights=weights,
        portfolio_returns=pd.Series(rng.normal(0.001, 0.01, len(index)), index=index),
        turnover=pd.Series(0.05, index=index),
        metrics={},
        transaction_costs=pd.Series(0.0001, index=index),
        ending_weights=weights.iloc[-1],
    )


def run_cli(monkeypatch, tmp_path, returns_path, failing_model, exception):
    """Drive main() with run_walk_forward stubbed: one model raises."""
    index = pd.read_csv(returns_path, index_col=0, parse_dates=True).index
    seen = []

    def stub(returns, allocator, config, **kwargs):
        name = kwargs.get("progress_label")
        seen.append(name)
        if name == failing_model:
            raise exception
        return fake_result(index)

    monkeypatch.setattr(compare_models, "run_walk_forward", stub)
    monkeypatch.setattr(
        compare_models, "latest_target_weights",
        lambda *a, **k: pd.Series(1.0 / len(ASSETS), index=ASSETS),
    )
    output = tmp_path / "comparison.csv"
    monkeypatch.setattr("sys.argv", [
        "compare_models",
        "--returns", str(returns_path),
        "--models", "equal_weight", "inverse_volatility",
        "--output", str(output),
        "--significance-benchmark", "equal_weight",
    ])
    compare_models.main()
    return output, seen


def test_failing_model_is_skipped_and_survivors_are_written(
    monkeypatch, tmp_path, returns_csv, capsys
):
    returns_path, _ = returns_csv
    output, seen = run_cli(
        monkeypatch, tmp_path, returns_path,
        failing_model="inverse_volatility",
        exception=ValueError("Liquidity/participation bounds are infeasible"),
    )

    # The failure did not abort the loop, and it did not abort the outputs.
    assert seen == ["equal_weight", "inverse_volatility"]
    assert output.exists()
    written = pd.read_csv(output, index_col="model")
    assert list(written.index) == ["equal_weight"]

    message = capsys.readouterr().out
    assert "inverse_volatility walk-forward failed" in message
    assert "these models failed" in message


def test_each_finished_model_is_checkpointed(monkeypatch, tmp_path, returns_csv):
    returns_path, _ = returns_csv
    output, _ = run_cli(
        monkeypatch, tmp_path, returns_path,
        failing_model="inverse_volatility",
        exception=ValueError("boom"),
    )
    partial = output.parent / "partial"
    series = pd.read_csv(partial / "equal_weight_series.csv", index_col="date")
    assert list(series.columns) == [
        "portfolio_return", "turnover", "transaction_cost"
    ]
    assert not series.empty
    # The crashed model leaves no checkpoint, by design.
    assert not (partial / "inverse_volatility_series.csv").exists()
    # The partial comparison is readable on its own.
    assert list(
        pd.read_csv(partial / "comparison_partial.csv", index_col="model").index
    ) == ["equal_weight"]


def test_a_crash_in_the_first_model_still_runs_the_rest(
    monkeypatch, tmp_path, returns_csv
):
    returns_path, _ = returns_csv
    output, seen = run_cli(
        monkeypatch, tmp_path, returns_path,
        failing_model="equal_weight",
        exception=RuntimeError("HiGHS Status 8"),
    )
    assert seen == ["equal_weight", "inverse_volatility"]
    assert list(pd.read_csv(output, index_col="model").index) == ["inverse_volatility"]


def test_every_model_failing_exits_nonzero(monkeypatch, tmp_path, returns_csv):
    returns_path, index = returns_csv

    def stub(returns, allocator, config, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(compare_models, "run_walk_forward", stub)
    monkeypatch.setattr("sys.argv", [
        "compare_models",
        "--returns", str(returns_path),
        "--models", "equal_weight", "inverse_volatility",
        "--output", str(tmp_path / "comparison.csv"),
    ])
    with pytest.raises(SystemExit) as excinfo:
        compare_models.main()
    assert "every model failed" in str(excinfo.value)
