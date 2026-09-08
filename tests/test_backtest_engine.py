from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from akm_hrp.backtest.cpcv import CPCVConfig
from akm_hrp.backtest.engine import (
    latest_target_weights,
    run_cpcv_backtest,
    run_walk_forward,
)
from akm_hrp.cli.compare_models import (
    _label_weight_frame,
    _load_ticker_map,
    _merge_balanced_only_returns,
    _save_weights_png,
)
from akm_hrp.data.pit_universe import load_pit_universe_mask
from akm_hrp.diagnostics.dashboard import export_backtest_dashboard


class EqualWeightAllocator:
    def allocate(self, returns):
        return pd.Series(1.0 / returns.shape[1], index=returns.columns)


class RecordingEqualWeightAllocator(EqualWeightAllocator):
    def __init__(self):
        self.last_window = None

    def allocate(self, returns):
        self.last_window = returns.copy()
        return super().allocate(returns)


class HoldingsRecordingAllocator(RecordingEqualWeightAllocator):
    def set_current_weights(self, weights):
        self.current_weights = weights.copy()


def _config(**overrides):
    values = {
        "lookback_weeks": 8,
        "min_window_obs": 2,
        "cov_max_interior_missing_fraction": 1.0,
        "risk_free_rate": 0.0,
        "min_weight": 0.0,
        "max_weight": 1.0,
        "pit_universe_file": None,
        "strict_pit_universe": False,
        "drift_threshold": 0.0,
        "min_holding_weeks": 10,
        "max_rebalance_turnover_l1": None,
        "tc_bps": 0.0,
        "cpcv_refit_per_split": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_weights_drift_between_rebalances():
    returns = pd.DataFrame(
        [[0.01, -0.01], [-0.01, 0.01], [1.0, 0.0], [0.0, 0.0]],
        index=pd.date_range("2020-01-03", periods=4, freq="W-FRI"),
        columns=["A", "B"],
    )
    result = run_walk_forward(returns, EqualWeightAllocator(), _config())

    assert np.allclose(result.weights.iloc[2].to_numpy(), [0.5, 0.5])
    assert np.allclose(result.weights.iloc[3].to_numpy(), [2.0 / 3.0, 1.0 / 3.0])


def test_cpcv_scores_test_rows_with_full_chronological_warmup():
    rng = np.random.default_rng(5)
    returns = pd.DataFrame(
        rng.normal(0.001, 0.01, size=(18, 3)),
        index=pd.date_range("2020-01-03", periods=18, freq="W-FRI"),
        columns=["A", "B", "C"],
    )
    cfg = _config(
        min_holding_weeks=0,
        cpcv=CPCVConfig(n_groups=3, test_groups=1, group_weeks=4),
    )
    results = run_cpcv_backtest(
        returns,
        lambda train: EqualWeightAllocator(),
        cfg,
    )

    assert len(results) == 3
    assert all(result.metrics["n_obs"] > 0 for result in results.values())


def test_latest_target_uses_final_row_and_configured_lookback():
    returns = pd.DataFrame(
        [
            [0.01, -0.01],
            [-0.01, 0.02],
            [0.02, 0.01],
            [-0.02, -0.01],
            [0.03, 0.02],
        ],
        index=pd.date_range("2020-01-03", periods=5, freq="W-FRI"),
        columns=["A", "B"],
    )
    allocator = RecordingEqualWeightAllocator()

    target = latest_target_weights(
        returns,
        allocator,
        _config(lookback_weeks=3, min_holding_weeks=0),
    )

    assert allocator.last_window.index.equals(returns.index[-3:])
    assert np.allclose(target.to_numpy(), [0.5, 0.5])


def test_latest_target_receives_ending_drifted_holdings():
    returns = pd.DataFrame(
        [[0.01, -0.01], [-0.01, 0.02], [0.02, 0.01]],
        index=pd.date_range("2020-01-03", periods=3, freq="W-FRI"),
        columns=["A", "B"],
    )
    allocator = HoldingsRecordingAllocator()
    live = pd.Series([0.7, 0.3], index=returns.columns)

    latest_target_weights(
        returns,
        allocator,
        _config(lookback_weeks=3, min_holding_weeks=0),
        current_weights=live,
    )

    pd.testing.assert_series_equal(allocator.current_weights, live)


def test_balanced_merge_adds_only_missing_columns() -> None:
    dates = pd.date_range("2025-01-03", periods=3, freq="W-FRI")
    broad = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=dates)
    balanced = pd.DataFrame(
        {"A": [9.0, 9.0, 9.0], "GOLD": [0.0, 0.01, -0.01]}, index=dates
    )

    merged = _merge_balanced_only_returns(broad, balanced)

    pd.testing.assert_series_equal(merged["A"], broad["A"])
    pd.testing.assert_series_equal(merged["GOLD"], balanced["GOLD"])


def test_walk_forward_progress_reports_elapsed_and_eta(capsys):
    returns = pd.DataFrame(
        [[0.01, -0.01], [-0.01, 0.02], [0.02, 0.01], [-0.02, -0.01]],
        index=pd.date_range("2020-01-03", periods=4, freq="W-FRI"),
        columns=["A", "B"],
    )

    run_walk_forward(
        returns,
        EqualWeightAllocator(),
        _config(min_holding_weeks=0),
        progress_every_rebalances=1,
        progress_label="test_model",
    )

    output = capsys.readouterr().out
    assert "test_model started" in output
    assert "elapsed=" in output
    assert "eta~" in output
    assert "finish~" in output
    assert "test_model finished" in output
    assert "progress=100.0%" in output


def test_weight_export_uses_tickers_and_keeps_permno(tmp_path):
    metadata_path = tmp_path / "crsp_security_metadata.csv"
    metadata_path.write_text("permno,ticker\n11995,AAPL\n19849,MSFT\n")
    weights = pd.DataFrame(
        {
            "date": ["2025-12-26", "2025-12-26", "2025-12-26"],
            "model": ["ra_hrp", "ra_hrp", "ra_hrp"],
            "asset": ["11995", "19849", "99999"],
            "weight": [0.4, 0.35, 0.25],
        }
    )

    labelled = _label_weight_frame(weights, _load_ticker_map(metadata_path))

    assert labelled["asset"].tolist() == ["AAPL", "MSFT", "99999"]
    assert labelled["permno"].tolist() == ["11995", "19849", "99999"]


def test_weight_png_is_rendered(tmp_path):
    weights = pd.DataFrame(
        {
            "date": ["2025-12-26"] * 3,
            "model": ["ra_hrp"] * 3,
            "asset": ["AAPL", "MSFT", "NVDA"],
            "weight": [0.4, 0.35, 0.25],
        }
    )
    destination = tmp_path / "weights.png"

    _save_weights_png(weights, destination, top_n=2)

    assert destination.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_backtest_dashboard_renders_pdf_and_png(tmp_path):
    dates = pd.date_range("2020-01-03", periods=120, freq="W-FRI")
    focus_returns = pd.Series(
        0.001 + 0.01 * np.sin(np.arange(len(dates)) / 7.0),
        index=dates,
    )
    benchmark_returns = pd.Series(
        0.0008 + 0.008 * np.cos(np.arange(len(dates)) / 9.0),
        index=dates,
    )
    focus_weights = pd.DataFrame(
        {
            "11995": 0.50 + 0.05 * np.sin(np.arange(len(dates)) / 8.0),
            "19849": 0.30 - 0.03 * np.sin(np.arange(len(dates)) / 8.0),
            "99999": 0.20 - 0.02 * np.sin(np.arange(len(dates)) / 8.0),
        },
        index=dates,
    )
    benchmark_weights = pd.DataFrame(
        np.full((len(dates), 3), 1.0 / 3.0),
        index=dates,
        columns=focus_weights.columns,
    )
    focus_result = SimpleNamespace(
        portfolio_returns=focus_returns,
        turnover=focus_weights.diff().abs().sum(axis=1).fillna(0.0),
        weights=focus_weights,
    )
    benchmark_result = SimpleNamespace(
        portfolio_returns=benchmark_returns,
        turnover=benchmark_weights.diff().abs().sum(axis=1).fillna(0.0),
        weights=benchmark_weights,
    )
    pdf_path = tmp_path / "dashboard.pdf"
    png_path = tmp_path / "dashboard.png"

    artifacts = export_backtest_dashboard(
        {"hrp_alpha_v2": focus_result, "equal_weight": benchmark_result},
        focus_model="hrp_alpha_v2",
        output_pdf=pdf_path,
        output_png=png_path,
        rolling_sharpe_years=1,
        ticker_map={"11995": "AAPL", "19849": "MSFT"},
    )

    assert artifacts == {"pdf": pdf_path, "png": png_path}
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_pit_loader_rejects_mismatched_asset_identifiers(tmp_path):
    pit_path = tmp_path / "pit.csv"
    pit_path.write_text("date,AAPL,MSFT\n2025-01-03,1,1\n")

    with pytest.raises(ValueError, match="0 asset columns in common"):
        load_pit_universe_mask(
            str(pit_path),
            dates=pd.DatetimeIndex(["2025-01-03"]),
            assets=["11995", "19849"],
        )
