#!/usr/bin/env bash
# Full-scale walk-forward, retail_alpha_ml_mpc vs equal_weight, rebalancing
# every 26 weeks (semi-annual), 60-name budget, 1% per-name floor.
#
# --max-rebalance-turnover 2.0: the engine's default 0.10 L1 cap per
# rebalance is sized for weekly trading. At 96 weeks it froze both books
# (planned turnover 1.2-1.8, executed 0.10 every time), so equal_weight vs
# the model was really two drifting buy-and-hold books. 2.0 lets a
# rebalance replace the whole book; the ML model's own ADV-participation
# and impact-cost limits still apply.
#
# Cadence notes: ~72 rebalances total, ~28 before --evaluation-start, so the
# ML ensemble (needs 12 cross-sections) is active from ~1998 and fully in
# play for the evaluation window. Refit every 5 rebalances = ~2.5 years;
# IC halflife 12 rebalances = ~6 years.
#
# BUDGET, DATA_START and RUN_DIR can be overridden from the environment.
# Launch under `caffeinate -is` -- there is no checkpointing.
set -euo pipefail
cd "$(dirname "$0")"

BUDGET="${BUDGET:-60}"
OUT="${RUN_DIR:-runs/retail_alpha_ml_mpc_26w_$(date +%Y%m%d_%H%M%S)}"
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
echo "cadence: 26w, max-total-assets budget: $BUDGET, min weight: 1%, turnover cap: 2.0"
"$PYTHON" -u -m akm_hrp.cli.compare_models \
  --returns data/weekly_returns.csv \
  --models equal_weight retail_alpha_ml_mpc \
  --pit data/pit_universe.csv \
  --dashboard-every-year \
  --rebalance-every-weeks 52 \
  --lookback-weeks 260 \
  --tc-bps 10 \
  --max-rebalance-turnover 2.0 \
  --dynamic-portfolio-value 100000 \
  --retail-mpc-horizon 3 \
  --retail-max-added-assets 30 \
  --retail-optimizer-max-iterations 450 \
  --retail-alpha-ml-max-total-assets "$BUDGET" \
  --retail-alpha-ml-min-weight 0.01 \
  --retail-alpha-ml-max-training-cross-sections 252 \
  ${DATA_START_ARGS[@]+"${DATA_START_ARGS[@]}"} \
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
  --dashboard-title "Full Scale (26w, budget=$BUDGET, min weight 1%, turnover cap 2.0): retail_alpha_ml_mpc vs Equal Weight" \
  "$@" 2>&1 | tee "$OUT/run.log"

echo "Done. Results in $OUT/"
