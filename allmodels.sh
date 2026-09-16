#!/usr/bin/env bash
# Full-scale walk-forward, quarterly (13-week) cadence, with the
# max-total-assets trim and the style/sector caps actually binding.
#
# --retail-alpha-ml-max-total-assets 60 rather than 40. With the trim fixed,
# 40 names cannot be simultaneously fully invested and inside all five
# +/-0.25 style-exposure caps at max_weight=0.03: the book reaches ~0.970
# invested and the solve reports linear infeasibility. 60 clears it.
#
# Caveat on that "with room to spare" claim, which this header used to make:
# it was verified over 16 consecutive weekly rebalances, and 16 was not
# enough. The 2026-09-15 weekly run logged 48+ `infeasible exposure caps`
# events, including a near-continuous Aug2015-Jan2016 cluster peaking at a
# 0.327 additive relaxation -- the +/-0.25 caps widened to +/-0.577, with the
# largest values in a calm month. Budget 60 does not hold the caps across the
# full history; it merely survives, because
# --retail-alpha-ml-allow-exposure-limit-relaxation is on. Treat any period
# with relaxations as constraint-violated when reading results.
# See claude/ml_recency_weight_floor_nan_importances.md.
#
# --max-rebalance-turnover 2.0 is NOT optional at this cadence.
# HRPConfig.max_rebalance_turnover_l1 defaults to 0.10 L1 per rebalance, an
# engine-level cap applied to every model. At weekly cadence that is ~5.2 L1
# per year and rarely binds. At 13 weeks it is 0.4 L1 per year, which freezes
# the book: the 96-week runs planned 1.2-1.8 L1 per rebalance, executed
# exactly 0.10 every time, sold ~5% of the names flagged for exit, and drifted
# as buy-and-hold portfolios. Equal weight then "won" a comparison that
# measured nothing. See claude/turnover_cap_freezes_slow_cadence_2026-09-15.md.
#
# --retail-alpha-ml-min-weight 0.01 replaces the derived 1/60 floor with an
# explicit 1% floor, matching run_26w.sh.
#
# Note the ML config counts REBALANCES, not weeks, so the 13-week cadence
# rescales it: refits every 5 rebalances are ~15 months apart, the "fast"
# halflife of 6 rebalances is ~18 months, the slow one ~6 years, and the
# dynamic-IC halflife of 12 rebalances is ~3 years. Fast and slow are much
# closer together here than the names suggest.
#
# BUDGET and DATA_START can be overridden from the environment:
#   BUDGET=80 ./run_full_scale_retail_alpha_ml_mpc_fixed.sh
#   DATA_START=1998-01-01 ./run_full_scale_retail_alpha_ml_mpc_fixed.sh
#
# DATA_START trims warm-up: rebalances before --evaluation-start 2005-01-05
# exist only to warm state. The ML buffer, factor buffer (260 weeks) and
# covariance lookback (260 weeks) are rolling and the dynamic ICs decay
# geometrically, so a 1998 start reaches 2005 fully warmed. The holdings path
# differs, so it is not bit-identical -- hence off by default.
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
echo "cadence: 13w, max-total-assets budget: $BUDGET, min weight: 1%, turnover cap: 2.0"
"$PYTHON" -u -m akm_hrp.cli.compare_models \
  --returns data/weekly_returns.csv \
  --models barra_factor_hrp dynamic_barra_alpha equal_weight hrp_alpha_v1 hrp_alpha_v2 inverse_volatility legacy_ensemble_hrp low_overfit_hrp mapper_factor_nco ra_hrp ra_hrp_v2 regret_aware_core regret_aware_with_overlay regularized_minimum_variance retail_alpha_ml_mpc retail_alpha_mpc retail_edge_ml_mpc retail_edge_mpc \
  --pit data/pit_universe.csv \
  --dashboard-every-year \
  --rebalance-every-weeks 13 \
  --max-rebalance-turnover 2.0 \
  --lookback-weeks 260 \
  --tc-bps 10 \
  --dynamic-portfolio-value 100000 \
  --retail-mpc-horizon 3 \
  --retail-max-added-assets 30 \
  --retail-optimizer-max-iterations 450 \
  --retail-alpha-ml-max-total-assets "$BUDGET" \
  --retail-alpha-ml-min-weight 0.01 \
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
  --dashboard-title "13w, budget=$BUDGET, 1% floor, turnover cap 2.0: retail_alpha_ml_mpc vs Equal Weight" \
  "$@" 2>&1 | tee "$OUT/run.log"

echo "Done. Results in $OUT/"
