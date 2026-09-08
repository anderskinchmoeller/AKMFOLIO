
import pandas as pd

def generate_pit_mask(audit_csv: str, returns_csv: str, output_csv: str):
    audit = pd.read_csv(audit_csv)

    returns = pd.read_csv(returns_csv, parse_dates=["Date"]).set_index("Date")
    dates = returns.index
    assets = returns.columns

    pit = pd.DataFrame(0, index=dates, columns=assets)

    for _, row in audit.iterrows():
        asset = row["Asset"]

        if asset not in pit.columns:
            continue

        if not bool(row["EverPITEligible"]):
            continue

        start = pd.to_datetime(row["FirstPITEligible"])
        end = pd.to_datetime(row["LastPITEligible"])

        mask = (dates >= start) & (dates <= end)
        pit.loc[mask, asset] = 1

    pit.to_csv(output_csv)
    print(f"PIT mask saved to {output_csv}")


if __name__ == "__main__":
    generate_pit_mask(
        audit_csv="~/hrp/akm_hrp/data/point_in_time_universe_audit.csv",
        returns_csv="~/hrp/weekly_returns.csv",
        output_csv="~/hrp/akm_hrp/data/pit_universe.csv"
    )

