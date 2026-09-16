cd /Users/anderskinch/AKMFOLIO
mkdir -p runs/full_universe_2026-09-09

nohup python -m akm_hrp.cli.compare_models \
  --returns data/weekly_returns.csv \
  --models equal_weight ra_hrp regularized_minimum_variance dynamic_barra_alpha retail_alpha_mpc retail_alpha_ml_mpc \
  --rebalance-every-weeks 13 \
  --evaluation-start 1996-01-05 \
  --significance-benchmark equal_weight \
  --deflated-sharpe-trials 6 \
  --retail-alpha-ml-allow-cvar-floor-relaxation \
  --progress-every-rebalances 25 \
  --output runs/full_universe_2026-09-09/results.csv \
  --weights-output runs/full_universe_2026-09-09/latest_weights.csv \
  --weights-png runs/full_universe_2026-09-09/latest_weights.png \
  --diagnostics-output runs/full_universe_2026-09-09/diagnostics.csv \
  --robustness-output runs/full_universe_2026-09-09/robustness \
  --dashboard-pdf runs/full_universe_2026-09-09/dashboard.pdf \
  --dashboard-png runs/full_universe_2026-09-09/dashboard.png \
  --dashboard-title "Full Universe 1990-2025" \
  > runs/full_universe_2026-09-09/run.log 2>&1 &

echo "PID: $!"
