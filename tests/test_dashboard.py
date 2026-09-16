from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from akm_hrp.diagnostics import dashboard
from akm_hrp.diagnostics.dashboard import export_backtest_dashboard


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

    expected = {
        "pdf": pdf_path,
        "png": png_path,
        "pdf_combined": tmp_path / "dashboard_combined.pdf",
    }
    for year in (2020, 2021, 2022):
        for kind in ("pdf", "png"):
            path = tmp_path / f"dashboard_{year}.{kind}"
            expected[f"{kind}_{year}"] = path
            magic = b"%PDF" if kind == "pdf" else b"\x89PNG\r\n\x1a\n"
            assert path.read_bytes().startswith(magic)
    assert artifacts == expected
    assert expected["pdf_combined"].read_bytes().startswith(b"%PDF")
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("kind", ["pdf", "png"])
def test_annual_dashboard_filters_periods_and_preserves_legacy_costs(tmp_path, monkeypatch, kind):
    dates = pd.to_datetime(["2019-12-27", "2020-01-03", "2020-07-03", "2022-01-07"])
    weights = pd.DataFrame({"A": [1., 0.8, 0.6, 0.4], "B": [0., 0.2, 0.4, 0.6]}, index=dates)
    result = SimpleNamespace(port_ret=pd.Series(0.01, index=dates), weights=weights)
    old_benchmark = SimpleNamespace(port_ret=pd.Series(0.02, index=dates[:3]), weights=weights.iloc[:3])
    calls = []

    def render(results, **kwargs):
        calls.append((results, kwargs))
        return {kind: kwargs[f"output_{kind}"]}

    monkeypatch.setattr(dashboard, "_export_dashboard_period", render)
    path = tmp_path / f"custom.dashboard.{kind}"
    artifacts = export_backtest_dashboard(
        {"focus": result, "benchmark": old_benchmark}, focus_model="focus",
        evaluation_start="2020-06-01", **{f"output_{kind}": path},
    )
    expected = {
        kind: path,
        f"{kind}_2020": tmp_path / f"custom.dashboard_2020.{kind}",
        f"{kind}_2022": tmp_path / f"custom.dashboard_2022.{kind}",
    }
    if kind == "pdf":
        expected["pdf_combined"] = tmp_path / "custom.dashboard_combined.pdf"
    assert artifacts == expected
    assert len(calls) == 3
    for (results, options), year in zip(calls[1:], [2020, 2022]):
        assert set(results) == ({"focus", "benchmark"} if year == 2020 else {"focus"})
        annual = results["focus"]
        assert list(annual.portfolio_returns.index.year) == [year]
        assert list(annual.weights.index.year) == [year]
        assert annual.transaction_costs.iloc[0] == pytest.approx(0.0004)
        assert options[f"output_{'png' if kind == 'pdf' else 'pdf'}"] is None
