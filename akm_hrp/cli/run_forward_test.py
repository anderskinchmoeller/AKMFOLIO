
import argparse
import pandas as pd

from akm_hrp.config import HRPConfig
from akm_hrp.data.returns import load_and_clean_returns
from akm_hrp.data.pit_universe import load_pit_universe_mask, pit_fingerprint
from akm_hrp.allocators.hrp_overlay_allocator import HRPOverlayAllocator, AllocatorConfig
from akm_hrp.backtest.engine import run_walk_forward
from akm_hrp.freeze.signature import validate_forward_run, FreezeConfig
from akm_hrp.freeze.manifest import load_manifest
from akm_hrp.diagnostics.reports import generate_diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--returns", default="weekly_returns.csv")
    parser.add_argument("--pit", default="")
    args = parser.parse_args()

    returns = load_and_clean_returns(args.returns, min_history_weeks=104)
    alloc_cfg = AllocatorConfig()
    freeze_cfg = FreezeConfig()

    manifest = load_manifest(freeze_cfg.manifest_path)
    if manifest is None:
        raise RuntimeError("No frozen manifest found; freeze the specification first.")

    pit_mask = None
    pit_fp = None
    if args.pit:
        pit_mask = load_pit_universe_mask(args.pit, returns.index, returns.columns)
        frozen_pit = pit_mask.loc[pit_mask.index <= pd.Timestamp(manifest.last_data_date)]
        pit_fp = pit_fingerprint(frozen_pit)

    # Validate frozen spec
    manifest = validate_forward_run(returns, alloc_cfg, pit_fp, freeze_cfg)

    # Forward-test region
    forward_index = returns.index[returns.index > pd.Timestamp(manifest.last_data_date)]

    allocator = HRPOverlayAllocator(alloc_cfg)
    bt_cfg = HRPConfig(pit_universe_file=args.pit or None)

    # Retain the frozen historical prefix for estimator warm-up, then report
    # only the untouched forward region.
    full_result = run_walk_forward(returns, allocator, bt_cfg)
    forward_weights = full_result.weights.reindex(forward_index)
    forward_returns = full_result.portfolio_returns.reindex(forward_index)

    # Diagnostics
    forward_asset_returns = returns.reindex(forward_index)
    corr = forward_asset_returns.corr()
    cov = forward_asset_returns.cov()
    alpha = forward_asset_returns.mean()

    report = generate_diagnostics(forward_weights, forward_returns, alpha, cov, corr)

    print("Forward-test complete.")
    print("Sharpe:", report.sharpe)
    print("Max Drawdown:", report.max_drawdown)
    print("Turnover:", report.ann_turnover)


if __name__ == "__main__":
    main()
