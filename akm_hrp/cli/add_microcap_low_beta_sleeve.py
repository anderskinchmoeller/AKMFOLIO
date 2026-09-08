from __future__ import annotations

"""Build a causal micro-cap/low-beta sleeve and append it to a balanced bundle."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ASSET = "MICROCAP_LOW_BETA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="wrds_microcap_low_beta")
    parser.add_argument("--target-dir", default="wrds_top2500/balanced_hrp")
    parser.add_argument("--minimum-history-weeks", type=int, default=52)
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=25.0,
        help="Cost per unit of L1 turnover (default: 25 bps).",
    )
    parser.add_argument(
        "--terminal-missing-return",
        type=float,
        default=0.0,
        help=(
            "Deprecated compatibility option. It must be 0 because CRSP CIZ "
            "already includes delisting returns (default: 0)."
        ),
    )
    return parser.parse_args()


def _load_wide(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    frame.columns = frame.columns.astype(str)
    if frame.index.has_duplicates:
        raise ValueError(f"{path} contains duplicate dates.")
    return frame.apply(pd.to_numeric, errors="coerce")


def build_monthly_sleeve(
    returns: pd.DataFrame,
    pit: pd.DataFrame,
    *,
    transaction_cost_bps: float,
    terminal_missing_return: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Hold the prior month-end PIT set and drift weights between formations."""

    if not returns.index.equals(pit.index):
        raise ValueError("Source returns and PIT masks use different dates.")
    if transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps cannot be negative.")
    if not np.isclose(terminal_missing_return, 0.0):
        raise ValueError(
            "terminal_missing_return must be 0. CRSP CIZ DlyRet already includes "
            "delisting returns, so an additional loss would double count them."
        )

    pit = pit.reindex_like(returns).fillna(0).astype(bool)
    output = pd.Series(np.nan, index=returns.index, name=ASSET, dtype=float)
    audit_rows: list[dict[str, object]] = []
    weights = pd.Series(dtype=float)
    cash_weight = 0.0
    has_formed = False

    for position in range(1, len(returns)):
        date = returns.index[position]
        previous_date = returns.index[position - 1]
        new_month = date.to_period("M") != previous_date.to_period("M")
        turnover = 0.0

        if (not has_formed and weights.empty) or new_month:
            eligible = pit.columns[pit.loc[previous_date]]
            if len(eligible):
                target = pd.Series(1.0 / len(eligible), index=eligible)
                if not weights.empty:
                    union = weights.index.union(target.index)
                    turnover = float(
                        (
                            weights.reindex(union).fillna(0.0)
                            - target.reindex(union).fillna(0.0)
                        )
                        .abs()
                        .sum()
                    )
                weights = target
                cash_weight = 0.0
                has_formed = True

        if weights.empty:
            if has_formed:
                output.loc[date] = 0.0
                audit_rows.append(
                    {
                        "date": date,
                        "constituent_count": 0,
                        "turnover_l1": turnover,
                        "trading_cost": 0.0,
                        "gross_return": 0.0,
                        "net_return": 0.0,
                        "missing_to_cash_count": 0,
                        "missing_to_cash_assets": "",
                    }
                )
            continue
        realized = returns.loc[date].reindex(weights.index)
        missing_assets = realized.index[realized.isna()]
        # CRSP CIZ DlyRet already contains the delisting return. Once a held
        # PERMNO has no current observation, preserve its proceeds as cash
        # rather than fabricating another loss. This decision uses only data
        # observable on ``date``; it never scans later returns.
        realized.loc[missing_assets] = 0.0
        gross_return = float(weights @ realized)
        trading_cost = turnover * transaction_cost_bps / 10_000.0
        output.loc[date] = gross_return - trading_cost
        audit_rows.append(
            {
                "date": date,
                "constituent_count": len(weights),
                "turnover_l1": turnover,
                "trading_cost": trading_cost,
                "gross_return": gross_return,
                "net_return": output.loc[date],
                "missing_to_cash_count": len(missing_assets),
                "missing_to_cash_assets": "|".join(missing_assets),
            }
        )

        ending_value = weights * (1.0 + realized)
        ending_value = ending_value.clip(lower=0.0)
        missing_proceeds = float(ending_value.reindex(missing_assets).sum())
        cash_value = cash_weight + missing_proceeds
        ending_value = ending_value.drop(index=missing_assets)
        total_value = float(ending_value.sum()) + cash_value
        if total_value > 0.0:
            weights = ending_value[ending_value > 0.0] / total_value
            cash_weight = cash_value / total_value
        else:
            weights = pd.Series(dtype=float)
            cash_weight = 1.0

    return output, pd.DataFrame(audit_rows)


def _backup_once(path: Path) -> Path:
    backup = path.with_name(f"{path.stem}.before_microcap_low_beta{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def _atomic_csv(frame: pd.DataFrame, path: Path, *, index: bool = True) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=index)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    source = Path(args.source_dir)
    target = Path(args.target_dir)
    source_returns = _load_wide(source / "weekly_returns.csv")
    source_pit = _load_wide(source / "pit_universe.csv")
    source_manifest_path = source / "wrds_data_manifest.json"
    if not source_manifest_path.exists():
        raise FileNotFoundError(
            "Source bundle must include wrds_data_manifest.json so delisting "
            "return treatment can be verified."
        )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("delisting_return_already_included") is not True:
        raise ValueError(
            "Source manifest must confirm delisting_return_already_included=true."
        )
    sleeve, audit = build_monthly_sleeve(
        source_returns,
        source_pit,
        transaction_cost_bps=args.transaction_cost_bps,
        terminal_missing_return=args.terminal_missing_return,
    )

    target_returns_path = target / "weekly_returns.csv"
    target_pit_path = target / "pit_universe.csv"
    metadata_path = target / "crsp_security_metadata.csv"
    manifest_path = target / "universe_manifest.json"
    required = [target_returns_path, target_pit_path, metadata_path, manifest_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Target bundle is missing: " + ", ".join(missing))

    target_returns = _load_wide(target_returns_path)
    target_pit = _load_wide(target_pit_path).fillna(0).astype("int8")
    sleeve = sleeve.reindex(target_returns.index)
    valid = sleeve.notna()
    sleeve_pit = (
        valid & (valid.cumsum() >= int(args.minimum_history_weeks))
    ).astype("int8")
    target_returns[ASSET] = sleeve
    target_pit[ASSET] = sleeve_pit

    metadata = pd.read_csv(metadata_path, dtype=str)
    metadata = metadata.loc[metadata["asset"] != ASSET].copy()
    metadata.loc[len(metadata)] = {
        "asset": ASSET,
        "permno": "",
        "ticker": ASSET,
        "cluster": "equity:microcap_low_beta",
        "role": "monthly long-only PIT micro-cap low-beta sleeve",
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["macro_columns"] = list(
        dict.fromkeys([*manifest.get("macro_columns", []), ASSET])
    )
    manifest["n_assets"] = int(target_returns.shape[1])
    manifest["microcap_low_beta_sleeve"] = {
        "source_dir": str(source.resolve()),
        "rebalance": "monthly",
        "weighting": "equal weight at formation, drift between formations",
        "transaction_cost_bps_per_l1": float(args.transaction_cost_bps),
        "missing_return_policy": (
            "CRSP CIZ delisting return already included; move missing holdings "
            "to zero-return cash without an additional loss"
        ),
        "minimum_history_weeks": int(args.minimum_history_weeks),
    }

    backups = [_backup_once(path) for path in required]
    _atomic_csv(target_returns, target_returns_path)
    _atomic_csv(target_pit, target_pit_path)
    _atomic_csv(metadata, metadata_path, index=False)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    _atomic_csv(audit.set_index("date"), target / "microcap_low_beta_audit.csv")

    active_dates = target_pit.index[target_pit[ASSET].astype(bool)]
    print(f"saved: {target_returns_path.resolve()}")
    print(f"added: {ASSET}")
    print(f"observed weekly returns: {int(sleeve.notna().sum())}")
    print(f"PIT active from: {active_dates.min().date().isoformat()}")
    print(f"missing-to-cash events: {int(audit['missing_to_cash_count'].sum())}")
    print("backups: " + ", ".join(str(path.resolve()) for path in backups))


if __name__ == "__main__":
    main()
