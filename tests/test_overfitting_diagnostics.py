"""Exercise diagnostic adapters against the real engine and report writer."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd

import run_overfitting_diagnostics as diagnostics


def test_allocator_config_cloning_and_pit():
    args = diagnostics.parse_args(['--quick'])
    config = diagnostics._build_config(args, crowding=True)
    cloned = diagnostics._clone_config_with(config, ml_shuffle_labels=True)
    assert cloned.ml_shuffle_labels and not config.ml_shuffle_labels
    pit = pd.DataFrame({'10001': [1]}, index=pd.to_datetime(['2024-01-05']))
    allocator = diagnostics._build_allocator(cloned, pit)
    pd.testing.assert_frame_equal(allocator.balanced_pit, pit.astype(bool), check_column_type=False)


def test_battery_with_real_engine_and_csv_reports(tmp_path, monkeypatch):
    dates = pd.date_range('2022-01-07', periods=160, freq='W-FRI', name='date')
    returns = pd.DataFrame(np.random.default_rng(4).normal(.002, .01, (160, 40)),
                           index=dates, columns=[str(10000 + i) for i in range(40)])
    pit = pd.DataFrame(1, index=dates, columns=returns.columns)
    returns.to_csv(tmp_path / 'returns.csv')
    pit.to_csv(tmp_path / 'pit.csv')
    created = []

    class CheapOnlineAllocator(diagnostics._EqualWeightAllocator):
        requires_split_training = True

        def __init__(self, config):
            self.config = replace(config, minimum_history_weeks=52)
            self.training_dates = None
            created.append(self)

        def set_training_dates(self, dates):
            self.training_dates = dates

    monkeypatch.setattr(diagnostics, '_build_allocator',
                        lambda config, balanced_pit: CheapOnlineAllocator(config))
    diagnostics.main([
        '--quick', '--returns', str(tmp_path / 'returns.csv'),
        '--pit', str(tmp_path / 'pit.csv'), '--evaluation-start', str(dates[80].date()),
        '--evaluation-end', str(dates[145].date()), '--output-dir', str(tmp_path / 'reports'),
    ])
    report_path, = (tmp_path / 'reports').glob('*/report.json')
    reports = {row['name']: row for row in json.loads(report_path.read_text())}
    assert not [r for r in reports.values() if r['status'] == 'ERROR']
    assert reports['walk-forward-oos']['metrics']['n_rebalances'] == 66
    assert reports['cpcv']['metrics']['n_paths'] == 3
    assert reports['block-bootstrap']['status'] in ('PASS', 'FLAG')
    assert set(reports['cost-stress']['metrics']) == {'1.0', '1.5', '2.0'}
    assert reports['significance']['status'] in ('PASS', 'FLAG')
    trained = [a for a in created if a.training_dates is not None]
    assert len(trained) == 3 and len({id(a) for a in trained}) == 3
    assert all(a.training_dates.max() <= dates[145] for a in trained)
    history = pd.read_csv(report_path.parent / 'robustness' / 'retail_alpha_ml_mpc_crowding_history.csv', index_col=0, parse_dates=True)
    assert history.index.max() == dates[145]
    assert history.index.min() == dates[0]  # Warmup is retained before scoring.


def test_robustness_benchmark_supports_large_universe():
    from akm_hrp.config import HRPConfig
    dates = pd.date_range('2023-01-06', periods=55, freq='W-FRI')
    returns = pd.DataFrame(np.random.default_rng(9).normal(.001, .01, (55, 120)), index=dates)
    result = diagnostics.ENGINE_MODULE.run_walk_forward(
        returns, diagnostics._EqualWeightAllocator(), HRPConfig(),
    )
    assert result.portfolio_returns.notna().sum() == 3
