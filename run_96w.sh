#!/usr/bin/env bash
# Full-scale walk-forward with the max-total-assets trim actually binding.
#
# Differs from run_full_scale_retail_alpha_ml_mpc.sh in one flag:
# --retail-alpha-ml-max-total-assets 60 instead of 40.
#
# Why not 40: with the trim fixed, 40 names cannot be simultaneously fully
# invested and inside all five +/-0.25 style-exposure caps at max_weight=0.03.
# The book gets to ~0.970 invested and the solve reports linear infeasibility
# after ~6 rebalances. 60 clears it with room to spare (verified over 16
# consecutive rebalances, every solve converging with status=0 and no repair
# pass). 80 also clears but costs ~4x the solve time for no obvious benefit.
#
# BUDGET and DATA_START can be overridden from the environment:
#   BUDGET=80 ./run_full_scale_retail_alpha_ml_mpc_fixed.sh
#   DATA_START=1998-01-01 ./run_full_scale_retail_alpha_ml_mpc_fixed.sh
#
# DATA_START is the cheap 1.6x: 730 of the 1,861 weekly rebalances run before
# --evaluation-start 2005-01-05 and exist only to warm state. The ML buffer
# (252 cross-sections), factor buffer (260 weeks) and covariance lookback
# (260 weeks) are all rolling, and the dynamic ICs decay geometrically
# (halflife 12 rebalances), so a 1998 start reaches 2005 fully warmed. The
# holdings path differs, so it is not bit-identical -- seven years of weekly
# turnover before the evaluation window makes that immaterial, but it is a
# judgement call, hence off by default.
set -euo pipefail
cd "$(dirname "$0")"

BUDGET="${BUDGET:-60}"
OUT="${RUN_DIR:-runs/full_scale_retail_alpha_ml_mpc_fixed_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

PYTHON="${PYTHON:-/Users/anderskinch/Portfolio/.venv/bin/python}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$(cd "$OUT" && pwd)/matplotlib"
mkdir -p "$MPLCONFIGDIR"

DATA_START_ARGS=()
if [[ -n "${DATA_START:-}" ]]; then
  DATA_START_ARGS=(--data-start "$DATA_START")
fi

echo "Run directory: $PWD/$OUT"
echo "max-total-assets budget: $BUDGET"
"$PYTHON" -u -m akm_hrp.cli.compare_models \
  --returns data/weekly_returns.csv \
  --models equal_weight retail_alpha_ml_mpc \
  --pit data/pit_universe.csv \
  --dashboard-every-year \
  --rebalance-every-weeks 96 \
  --lookback-weeks 260 \
  --tc-bps 10 \
  --dynamic-portfolio-value 100000 \
  --retail-mpc-horizon 3 \
  --retail-max-added-assets 30 \
  --retail-optimizer-max-iterations 450 \
  --retail-alpha-ml-max-total-assets "$BUDGET" \
  --retail-alpha-ml-max-training-cross-sections 252 \
  "${DATA_START_ARGS[@]}" \
  --evaluation-start 2005-01-05 \
  --significance-benchmark equal_weight \
  --deflated-sharpe-trials 6 \
  --retail-alpha-ml-allow-exposure-limit-relaxation \
  --retail-alpha-ml-allow-cvar-floor-relaxation \
  --progress-every-rebalances 1 \
  --dynamic-sector-history data/sector_history.csv \
  --dynamic-features data/structural_features.csv \
  --dynamic-balanced-pit data/balanced_hrp/pit_universe.csv \
  --dynamic-balanced-returns data/balanced_hrp/weekly_returns.csv \
  --asset-metadata data/crsp_security_metadata.csv \
  --output "$OUT/results.csv" \
  --weights-output "$OUT/latest_weights.csv" \
  --weights-png "$OUT/latest_weights.png" \
  --diagnostics-output "$OUT/diagnostics.csv" \
  --robustness-output "$OUT/robustness" \
  --dashboard-pdf "$OUT/dashboard.pdf" \
  --dashboard-png "$OUT/dashboard.png" \
  --dashboard-focus-model retail_alpha_ml_mpc \
  --dashboard-title "Full Scale (max-assets trim binding, budget=$BUDGET): retail_alpha_ml_mpc vs Equal Weight" \
  "$@" 2>&1 | tee "$OUT/run.log"

echo "Done. Results in $OUT/"
