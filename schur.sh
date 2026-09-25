#!/usr/bin/env bash
# schur_hrp on the same terms as the allmodels.sh horse race:
# full CRSP universe, 13-week cadence, --evaluation-start 2005-01-05,
# turnover cap 2.0 (the 0.10 engine default freezes the book at this cadence),
# lookback 260w, 10bps costs, global max weight 0.10 -- identical to the other
# HRP-family models in allmodels.sh.
#
# schur_hrp_g1 is deliberately left out: at gamma 0.5 vs 1.0 the weight vectors
# correlate 0.993 and vol differs by 0.1pp, while turnover rises, so the second
# variant spends a deflated-Sharpe trial on a near-duplicate portfolio.
# See claude/schur_hrp_model_review_2026-09-23.md.
#
# --deflated-sharpe-trials 18 matches the trial count reported by the
# 2026-09-21 18-model run, so the DSR column is comparable with it.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${RUN_DIR:-runs/schur_full_scale_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

PYTHON="${PYTHON:-python3}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$(cd "$OUT" && pwd)/matplotlib"
mkdir -p "$MPLCONFIGDIR"

echo "Run directory: $PWD/$OUT"
"$PYTHON" -u -m akm_hrp.cli.compare_models \
  --returns data/weekly_returns.csv \
  --models equal_weight schur_hrp \
  --pit data/pit_universe.csv \
  --rebalance-every-weeks 13 \
  --max-rebalance-turnover 2.0 \
  --lookback-weeks 260 \
  --tc-bps 10 \
  --dynamic-portfolio-value 100000 \
  --schur-cov-halflife 26 \
  --evaluation-start 2005-01-05 \
  --significance-benchmark equal_weight \
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
  --dashboard-title "schur_hrp (gamma=0.5) vs Equal Weight, 13w, 2005-2026, full universe" \
  "$@" 2>&1 | tee "$OUT/run.log"

echo "Done. Results in $OUT/"
