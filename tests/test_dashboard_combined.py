"""export_backtest_dashboard also writes one combined multi-page PDF."""

import numpy as np
import pandas as pd
from types import SimpleNamespace

from akm_hrp.diagnostics.dashboard import export_backtest_dashboard


def _result(seed):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-01-04", periods=150, freq="W-FRI")
    weights = pd.DataFrame(
        rng.dirichlet(np.ones(5), size=len(index)), index=index, columns=list("abcde")
    )
    return SimpleNamespace(
        portfolio_returns=pd.Series(rng.normal(0.002, 0.02, len(index)), index=index),
        weights=weights,
        turnover=pd.Series(0.1, index=index),
        transaction_costs=pd.Series(0.0001, index=index),
    )


def test_combined_pdf_has_full_period_plus_each_year(tmp_path):
    results = {"model": _result(0), "equal_weight": _result(1)}
    paths = export_backtest_dashboard(
        results, focus_model="model", output_pdf=tmp_path / "dashboard.pdf"
    )
    combined = paths["pdf_combined"]
    assert combined == tmp_path / "dashboard_combined.pdf"
    years = sorted(results["model"].portfolio_returns.index.year.unique())
    raw = combined.read_bytes()
    assert raw.count(b"/Type /Page\n") + raw.count(b"/Type /Page ") >= 1
    # PdfPages writes one /Page object per figure.
    pages = len([m for m in raw.split(b"/Type") if m.lstrip().startswith(b"/Page") and not m.lstrip().startswith(b"/Pages")])
    assert pages == 1 + len(years)
    assert (tmp_path / "dashboard.pdf").exists()
    for year in years:
        assert (tmp_path / f"dashboard_{year}.pdf").exists()


def test_png_only_writes_no_combined(tmp_path):
    paths = export_backtest_dashboard(
        {"model": _result(0)}, focus_model="model", output_png=tmp_path / "d.png"
    )
    assert "pdf_combined" not in paths
    assert not list(tmp_path.glob("*_combined.pdf"))
