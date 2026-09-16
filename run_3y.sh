#!/usr/bin/env bash
# Full history, weekly rebalancing, annual dashboards exported during the run.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${RUN_DIR:-runs/full_scale_retail_alpha_ml_mpc_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

PYTHON="${PYTHON:-/Users/anderskinch/Portfolio/.venv/bin/python}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$(cd "$OUT" && pwd)/matplotlib"
mkdir -p "$MPLCONFIGDIR"
echo "Run directory: $PWD/$OUT"
"$PYTHON" -u -m akm_hrp.cli.compare_models \
  --returns data/weekly_returns.csv \
  --models equal_weight retail_alpha_ml_mpc \
  --pit data/pit_universe.csv \
  --dashboard-every-year \
  --rebalance-every-weeks 13 \
  --evaluation-start 2021-01-05 \
  --significance-benchmark equal_weight \
  --deflated-sharpe-trials 6 \
  --retail-alpha-ml-allow-cvar-floor-relaxation \
  --progress-every-rebalances 1 \
  --dynamic-sector-history data/sector_history.csv \
  --dynamic-features data/structural_features.csv \
  --dynamic-balanced-pit data/balanced_hrp/pit_universe.csv \
  --output "$OUT/results.csv" \
  --weights-output "$OUT/latest_weights.csv" \
  --weights-png "$OUT/latest_weights.png" \
  --diagnostics-output "$OUT/diagnostics.csv" \
  --robustness-output "$OUT/robustness" \
  --dashboard-pdf "$OUT/dashboard.pdf" \
  --dashboard-png "$OUT/dashboard.png" \
  --dashboard-focus-model retail_alpha_ml_mpc \
  --dashboard-title "Full Scale: retail_alpha_ml_mpc vs Equal Weight (through 2026-08-28)" \
  "$@" 2>&1 | tee "$OUT/run.log"

echo "Done. Results in $OUT/"
