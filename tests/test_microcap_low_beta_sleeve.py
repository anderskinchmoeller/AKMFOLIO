import numpy as np
import pandas as pd
import pytest

from akm_hrp.cli.add_microcap_low_beta_sleeve import build_monthly_sleeve


def test_monthly_sleeve_uses_prior_pit_and_moves_missing_holding_to_cash() -> None:
    dates = pd.date_range("2025-01-03", periods=10, freq="W-FRI")
    returns = pd.DataFrame(
        {
            "A": [0.0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.02, 0.02, 0.02],
            # B resumes trading later. The missing observation must be handled
            # without consulting that future return.
            "B": [0.0, 0.00, 0.00, 0.00, 0.00, 0.00, np.nan, 0.03, 0.03, 0.03],
        },
        index=dates,
    )
    pit = pd.DataFrame(False, index=dates, columns=returns.columns)
    pit.loc[:, ["A", "B"]] = True
    pit.loc[dates[6]:, "B"] = False

    sleeve, audit = build_monthly_sleeve(
        returns,
        pit,
        transaction_cost_bps=0.0,
        terminal_missing_return=0.0,
    )

    missing_week = dates[6]
    expected = 0.5 * returns.loc[missing_week, "A"]
    assert sleeve.loc[missing_week] == pytest.approx(expected)
    row = audit.set_index("date").loc[missing_week]
    assert row["missing_to_cash_count"] == 1
    assert row["missing_to_cash_assets"] == "B"
    assert "B" not in audit.set_index("date").loc[dates[7], "missing_to_cash_assets"]
    assert sleeve.dropna().index.min() == dates[1]


def test_missing_return_handling_is_invariant_to_future_data() -> None:
    dates = pd.date_range("2025-01-03", periods=8, freq="W-FRI")
    returns = pd.DataFrame(
        {"A": [0.0, 0.01, np.nan, 0.02, 0.01, 0.01, 0.01, 0.01]},
        index=dates,
    )
    pit = pd.DataFrame(True, index=dates, columns=["A"])

    full, _ = build_monthly_sleeve(
        returns, pit, transaction_cost_bps=0.0, terminal_missing_return=0.0
    )
    truncated, _ = build_monthly_sleeve(
        returns.iloc[:3],
        pit.iloc[:3],
        transaction_cost_bps=0.0,
        terminal_missing_return=0.0,
    )

    pd.testing.assert_series_equal(full.iloc[:3], truncated)
    assert full.loc[dates[3]] == pytest.approx(0.0)


def test_additional_terminal_loss_is_rejected() -> None:
    dates = pd.date_range("2025-01-03", periods=3, freq="W-FRI")
    returns = pd.DataFrame({"A": [0.0, 0.01, np.nan]}, index=dates)
    pit = pd.DataFrame(True, index=dates, columns=["A"])

    with pytest.raises(ValueError, match="already includes delisting returns"):
        build_monthly_sleeve(
            returns,
            pit,
            transaction_cost_bps=0.0,
            terminal_missing_return=-1.0,
        )


def test_monthly_rebalance_charges_l1_cost() -> None:
    dates = pd.date_range("2025-01-03", periods=9, freq="W-FRI")
    returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    pit = pd.DataFrame(False, index=dates, columns=returns.columns)
    pit.loc[:, "A"] = True
    pit.loc[dates[4]:, "B"] = True

    sleeve, audit = build_monthly_sleeve(
        returns,
        pit,
        transaction_cost_bps=25.0,
        terminal_missing_return=0.0,
    )

    charged = audit.loc[audit["trading_cost"] > 0.0].iloc[0]
    assert charged["turnover_l1"] == pytest.approx(1.0)
    assert charged["trading_cost"] == pytest.approx(0.0025)
    assert sleeve.loc[charged["date"]] == pytest.approx(-0.0025)
