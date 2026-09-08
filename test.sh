#!/bin/bash
cd /Users/anderskinch/AKMFOLIO
mkdir -p outputs
nohup python3 -u -c "
import sys
sys.argv = [
    'compare_models.py', '--returns', 'data/weekly_returns.csv',
    '--models', 'equal_weight', 'retail_alpha_ml_mpc_crowding',
    '--retail-alpha-ml-kelly-mix-enabled', '--retail-alpha-ml-kelly-scale-enabled',
    '--retail-alpha-ml-allow-cvar-floor-relaxation',
    '--data-start', '2012-01-01', '--evaluation-start', '2016-01-01',
    '--rebalance-every-weeks', '48', '--significance-benchmark', 'equal_weight',
    '--output', 'outputs/model_comparison_weekly1.csv',
    '--dashboard-png', 'outputs/dashboard_weekly1.png',
    '--dashboard-pdf', 'outputs/dashboard_weekly1.pdf',
    '--dashboard-focus-model', 'retail_alpha_ml_mpc_crowding',
    '--dashboard-title', 'Crowding Kappa vs Equal Weight1 (weekly, 2025 eval)',
    '--diagnostics-output', 'outputs/diagnostics_weekly1.csv',
    '--progress-every-rebalances', '1',
]
exec(open('run_with_kelly_capture.py').read())
" > outputs/crowding_test_run.log 2>&1 &
disown
echo "PID: $!"

