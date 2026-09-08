import pandas as pd

from audit_missing_held_returns import audit_missing_held_returns


def test_missing_held_returns_are_classified_and_saved(tmp_path):
    dates = pd.date_range("2020-01-03", periods=4, freq="W-FRI")
    returns = pd.DataFrame(
        {
            "A": [0.01, None, 0.02, 0.03],
            "B": [0.01, 0.02, None, 0.03],
            "C": [0.01, 0.02, 0.03, None],
        },
        index=dates,
    )
    pit = pd.DataFrame(
        {
            "A": [1, 1, 1, 1],
            "B": [1, 1, 0, 0],
            "C": [1, 1, 1, 1],
        },
        index=dates,
    )
    returns_path = tmp_path / "returns.csv"
    pit_path = tmp_path / "pit.csv"
    output_path = tmp_path / "reports" / "audit.csv"
    returns.to_csv(returns_path)
    pit.to_csv(pit_path)

    report = audit_missing_held_returns(returns_path, pit_path, output_path)

    classifications = report.set_index("permno")["classification"].to_dict()
    assert classifications == {
        "A": "interior_gap",
        "B": "universe_exit_or_possible_delisting",
        "C": "terminal_missing",
    }
    saved = pd.read_csv(output_path, parse_dates=["date"])
    pd.testing.assert_frame_equal(saved, report.reset_index(drop=True))


def test_clean_history_produces_an_empty_report_with_stable_columns(tmp_path):
    dates = pd.date_range("2020-01-03", periods=2, freq="W-FRI")
    returns = pd.DataFrame({"A": [0.01, 0.02]}, index=dates)
    pit = pd.DataFrame({"A": [1, 1]}, index=dates)
    returns_path = tmp_path / "returns.csv"
    pit_path = tmp_path / "pit.csv"
    output_path = tmp_path / "audit.csv"
    returns.to_csv(returns_path)
    pit.to_csv(pit_path)

    report = audit_missing_held_returns(returns_path, pit_path, output_path)

    assert report.empty
    assert list(pd.read_csv(output_path).columns) == list(report.columns)
