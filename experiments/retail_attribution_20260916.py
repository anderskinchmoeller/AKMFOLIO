"""Isolated attribution experiment; does not change production allocators."""
from pathlib import Path
import sys
import json
import hashlib
import types
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from akm_hrp.cli import compare_models as cli

OUT = ROOT / 'runs/retail_attribution_20260916'
OUT.mkdir(parents=True, exist_ok=True)
original_run = cli.run_walk_forward

def save(name, result):
    pd.DataFrame({'portfolio_return': result.portfolio_returns,
                  'turnover': result.turnover,
                  'transaction_cost': result.transaction_costs}).to_csv(OUT / f'{name}_series.csv', index_label='date')
    result.diagnostics.to_csv(OUT / f'{name}_diagnostics.csv', index=False)
    dates = pd.DatetimeIndex(result.diagnostics.date)
    result.weights.loc[dates].to_csv(OUT / f'{name}_rebalance_weights.csv.gz', index_label='date')

def replay(returns, result, equal):
    """Replay exact realized baseline holdings sets, with weekly weight drift."""
    data = returns.reindex(index=result.weights.index, columns=result.weights.columns).replace([np.inf, -np.inf], np.nan).fillna(0.)
    dates = set(pd.DatetimeIndex(result.diagnostics.date))
    current = np.zeros(len(data.columns))
    rows = []
    for date, realized in zip(data.index, data.to_numpy(), strict=True):
        turnover = 0.
        if date in dates:
            target = result.weights.loc[date].to_numpy().copy()
            if equal:
                target = (target > 1e-10).astype(float)
                target /= target.sum()
            turnover = float(np.abs(target-current).sum()) if current.sum() > 1e-12 else 0.
            current = target
        gross = float(current @ realized) if current.sum() > 1e-12 else np.nan
        rows.append((date, gross, turnover, turnover * .001))
        if np.isfinite(gross):
            current = np.maximum(current * (1 + realized), 0.)
            if current.sum() > 1e-12:
                current /= current.sum()
    frame = pd.DataFrame(rows, columns=['date','gross_return','turnover','transaction_cost']).set_index('date')
    frame['portfolio_return'] = frame.gross_return-frame.transaction_cost
    return frame

def disabled_ml(self, design, eligible):
    self.last_ml_consensus_fraction = 0.
    return pd.Series(0., index=design.index), pd.Series(False, index=design.index)

def experiment(returns, allocator, config, **kwargs):
    (OUT/'configuration.json').write_text(json.dumps({'engine':vars(config),'allocator':vars(allocator.config),'arguments':sys.argv},default=str,indent=2))
    base = original_run(returns, allocator, config, **kwargs)
    save('full_model', base)
    weighted = replay(returns, base, False)
    expected = base.portfolio_returns + base.transaction_costs
    error = float((weighted.gross_return-expected).abs().max())
    assert error < 1e-10, f'Baseline replay mismatch: {error}'
    assert float((weighted.turnover-base.turnover).abs().max()) < 1e-8
    weighted.to_csv(OUT/'full_model_common_cost_series.csv', index_label='date')
    replay(returns, base, True).to_csv(OUT/'selected_equal_weight_series.csv', index_label='date')
    reference = pd.read_csv(ROOT/'runs/full_scale_retail_alpha_ml_mpc_fixed_20260916_110248/partial/retail_alpha_ml_mpc_series.csv',index_col=0,parse_dates=True)
    difference = float((base.portfolio_returns-reference.portfolio_return).abs().max())
    (OUT/'validation.json').write_text(json.dumps({'replay_max_absolute_error':error,'original_run_max_absolute_return_difference':difference},indent=2))
    print(f'BASELINE VALIDATION: replay error={error:.3g}; original run difference={difference:.3g}',flush=True)
    allocator.reset_state()
    allocator._ml_signal_values = types.MethodType(disabled_ml, allocator)
    no_ml = original_run(returns, allocator, config, progress_every_rebalances=1, progress_label='no_ml_signal')
    save('no_ml_signal', no_ml)
    assert no_ml.diagnostics['signal_weight__ml_interaction_ensemble'].abs().max() == 0.
    print('ATTRIBUTION RUNS COMPLETE',flush=True)
    return base

if __name__ == '__main__':
    files = ['akm_hrp/allocators/retail_alpha_ml_mpc.py','akm_hrp/allocators/retail_alpha_mpc.py','akm_hrp/backtest/engine.py','akm_hrp/cli/compare_models.py']
    (OUT/'source_hashes.json').write_text(json.dumps({f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},indent=2))
    cli.run_walk_forward = experiment
    sys.argv = [sys.argv[0], '--returns','data/weekly_returns.csv','--models','retail_alpha_ml_mpc',
        '--pit','data/pit_universe.csv','--rebalance-every-weeks','13','--max-rebalance-turnover','2.0',
        '--lookback-weeks','260','--tc-bps','10','--dynamic-portfolio-value','100000',
        '--retail-mpc-horizon','3','--retail-max-added-assets','30','--retail-optimizer-max-iterations','450',
        '--retail-alpha-ml-max-total-assets','60','--retail-alpha-ml-min-weight','0.01',
        '--retail-alpha-ml-max-training-cross-sections','252','--data-start','2006-01-01',
        '--evaluation-start','2012-01-06','--deflated-sharpe-trials','6',
        '--retail-alpha-ml-allow-exposure-limit-relaxation','--retail-alpha-ml-allow-cvar-floor-relaxation',
        '--progress-every-rebalances','1','--dynamic-sector-history','data/sector_history.csv',
        '--dynamic-features','data/structural_features.csv','data/compustat_pit_features_long.csv.gz',
        '--dynamic-balanced-pit','data/balanced_hrp/pit_universe.csv',
        '--dynamic-balanced-returns','data/balanced_hrp/weekly_returns.csv',
        '--output',str(OUT/'baseline_results.csv')]
    cli.main()
