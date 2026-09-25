#!/usr/bin/env bash
# ra_hrp vs schur_hrp vs equal_weight in ONE invocation, on the allmodels.sh terms:
# full CRSP universe, 13-week cadence, --evaluation-start 2005-01-05, turnover cap 2.0
# (the 0.10 engine default freezes the book at this cadence), lookback 260w, 10bps.
#
# Replaces ra_hrp.sh, which was a copy of run_full_scale_retail_alpha_ml_mpc_fixed.sh:
#  - its --dashboard-focus-model retail_alpha_ml_mpc crashed the run after results.csv was
#    written (runs/full_scale_retail_alpha_ml_mpc_fixed_20260922_114314), so no dashboard;
#  - run dir, title and header comments described the ML model, not ra_hrp;
#  - --deflated-sharpe-trials 6 vs 18 in schur.sh made the DSR columns incomparable.
# Previous results (same terms, run separately, equal_weight bit-identical in both):
#   ra_hrp     SR 0.319  CAGR 4.68%  vol 21.8%  MDD -62.3%  turnover 3.74  vs EW -4.64%/yr (t -1.41)
#   schur_hrp  SR 0.652  CAGR 9.18%  vol 15.3%  MDD -44.6%  turnover 1.72  vs EW -1.62%/yr (t -1.11)
# ~3h. The desktop workspace kills long background jobs; run it from a normal terminal.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${RUN_DIR:-runs/ra_hrp_vs_schur_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

PYTHON="${PYTHON:-/Users/anderskinch/Portfolio/.venv/bin/python}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$(cd "$OUT" && pwd)/matplotlib"
mkdir -p "$MPLCONFIGDIR"

echo "Run directory: $PWD/$OUT"
"$PYTHON" -u -m akm_hrp.cli.compare_models \
  --returns data/weekly_returns.csv \
  --models equal_weight ra_hrp schur_hrp \
  --pit data/pit_universe.csv \
  --rebalance-every-weeks 13 \
  --max-rebalance-turnover 2.0 \
  --lookback-weeks 260 \
  --tc-bps 10 \
  --dynamic-portfolio-value 100000 \
  --schur-cov-halflife 26 \
  --evaluation-start 2005-01-05 \
  --significance-benchmark equal_weight \
  --external-benchmarks URTH \
  --deflated-sharpe-trials 18 \
  --progress-every-rebalances 1 \
  --dynamic-sector-history data/sector_history.csv \
  --dynamic-features data/structural_features.csv data/compustat_pit_features_long.csv.gz \
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
  --dashboard-focus-model schur_hrp \
  --dashboard-title "ra_hrp vs schur_hrp vs Equal Weight, 13w, 2005-2026, full universe" \
  "$@" 2>&1 | tee "$OUT/run.log"

echo "Done. Results in $OUT/"
