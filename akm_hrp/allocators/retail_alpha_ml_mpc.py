from __future__ import annotations

"""Causal nonlinear signal ensemble for the Retail Alpha MPC allocator.

Mirrors the design used for the retail-capacity MPC model, but swaps its
closed-form learner for a boosted-tree one. Completed cross-sections
(features + realized forward-return target) are appended to a bounded
rolling history buffer rather than folded into a Gram matrix, and "fast"
and "slow" become two `GradientBoostingRegressor` models refit from scratch
on that same buffer with different recency-weighted `sample_weight` decays
-- reproducing the original fast/slow-halflife idea through recency-weighted
*refitting* instead of recency-weighted *accumulation*, since a boosted
model has no closed-form incremental update. Because refitting boosted
trees on every rebalance is a genuine, non-trivial cost that ridge never
had, the refit cadence is its own explicit knob
(`ml_retrain_every_n_rebalances`) rather than something that happens on
every call. A third, unweighted ridge model is also fit on the same buffer
as a structurally different vote -- two GBMs differing only in
sample_weight decay are one model family fit twice, not real ensemble
diversity, so a stock now receives an ML score when at least a majority
(2 of 3 by default, `ml_consensus_min_votes`) of the fast/slow/linear
predictions agree on its sign, rather than requiring the old two-model
unanimity. All of it plugs into the existing `_build_signal_panel` hook, so
the parent's factor risk model, multi-period optimizer, execution-cost
model, and online IC-weighting machinery are left untouched -- the ML
signal is learned and weighted through exactly the same Bayesian
IC-shrinkage process as the other five signals.

The nonlinear design also adds a small size ("small-minus-big") sleeve --
none of the parent's five base signals is a size factor, which is the
single clearest gap flagged by a cross-section of size-effect research (see
`_ml_design` for citations and the caveats about sign instability and
data-quality sensitivity that come with it). The size features are disabled
outright for any formation date where market-cap coverage is too sparse to
trust, rather than ranking whatever partial sample is present.

Two more signals are added alongside the five inherited base signals and the
ML ensemble, as standalone entries in `signal_names` rather than as ML-only
features, so each earns its own IC-shrunk weight through the parent's
existing online IC-learning machinery instead of being folded invisibly into
one blended score:

  microcap_tilt
      A more direct small-cap tilt than the parent's `retail_agility`, which
      *multiplies* its size edge by a liquidity gate and so is deliberately
      throttled to retail-implementable capacity. `microcap_tilt` keeps a
      liquidity floor (to stay out of the untradeable tail) without that
      capacity throttle, and adds the same convex tail-decile kink used in
      `_ml_design`'s size block. See `_tilt_signal_panel` for the full
      construction and the same size-effect caveats that apply to the ML
      sleeve's size features -- plus one more: Hou, Xue & Zhang ("Replicating
      Anomalies", RFS 2020) find that a large fraction of published
      cross-sectional anomalies fail to clear a t>=1.96 hurdle once microcaps
      are handled the standard academic way (NYSE breakpoints, value
      weighting) rather than left in equal-weighted, all-cap samples -- a
      direct caution about overstating a microcap-tilted signal's edge from a
      naive backtest, not just about the sign risk `_ml_design` already notes.

  carry_quality_tilt
      A standalone "good carry / bad carry" tilt. The parent's
      `quality_value_carry` already blends in `shareholder_carry`, but only
      as 1 of 8 equally weighted components -- too diluted to function as a
      distinguishable carry bet. Bekaert & Panayotov ("Good Carry, Bad
      Carry", JFQA) show that FX carry trades built the textbook way (long
      the highest-yield currencies) are actually the worse-Sharpe,
      worse-skew half, and that conditioning the trade on which currencies'
      carry is durable rather than crash-prone recovers most of the
      difference. There's no per-stock skew/crash history to condition on
      the way their ~30 years of daily FX quotes across nine currencies
      allow, so `carry_quality_tilt` conditions `shareholder_carry` on the
      same quality proxies `quality_value_carry` already uses instead:
      high carry paired with strong quality is kept near full strength as
      "good" carry; high carry paired with weak quality -- the equity
      analogue of a crash-prone "bad" carry basket -- is damped toward zero.
      See `_tilt_signal_panel`.

All three tilts go through the same market/industry neutralization and
cap-weighted residualization the parent applies to its own five signals, so
they enter the factor risk model and the optimizer on the same footing.

Every signal in this file -- the three tilts, the ML ensemble, and (via the
parent, untouched) the five base signals -- already runs the full
winsorize -> rank-normal -> neutralize -> combine -> Grinold -> rank-IC
pipeline, not as a new addition but because that pipeline is how this
codebase has always worked: `_rank_normal` winsorizes (1st/99th-percentile
clip) and rank-normal-transforms; `_neutralize` (plus the cap-weighted
residual pass every raw signal and both `_tilt_signal_panel` and
`_clean_edge_signals` apply) removes market/industry exposure;
`_signal_combination` combines signals into blend weights via
correlation-shrunk, IC-uncertainty-scaled confidence; the per-name alpha the
optimizer actually trades -- `signal_weights[s] * signal_ics[s] *
specific_volatility * signal_score` -- *is* Grinold's Fundamental Law of
Active Management (alpha = IC x score x volatility) applied per signal; and
`signal_ics` are themselves rolling Spearman rank-ICs (`_update_dynamic_ics`).
The two new tilts and `robust_fundamental_carry` were built to fit this
pipeline exactly, not around it.

`RetailAlphaMPCAllocator` and every layer beneath it (`apply_bounds`'s
explicit rejection of a negative `min_weight`, the HRP prior, the
first-period participation bounds, and the multi-period optimizer's
`lower_matrix = np.zeros(...)` box constraints) are long-only by
construction -- not just bounded long-only, since HRP's recursive bisection
itself allocates non-negative *fractions* of a unit budget rather than
producing signed weights. Rebuilding that engine to carry signed weights
end-to-end is a real portfolio-construction redesign, so instead this file
adds a separate, opt-in short overlay (`short_overlay_enabled`, default
`False`) that leaves the inherited long-only engine completely untouched:

  RetailAlphaMLMPCAllocator.allocate() calls the parent's allocate() first
  to get exactly the long book it would produce on its own, then --  only if
  the overlay is enabled -- separately selects a small, capped set of the
  full universe's most negative-composite-score names (using the same
  IC-shrunk signal weights the long book's admission score uses, restricted
  to names the long engine didn't already select) and assigns them negative
  weights up to a fixed gross short budget (`short_gross_budget`, e.g. 15%
  of the book). The two are simply added together; nothing about the long
  book's HRP prior, factor risk model, or multi-period optimizer changes.

  This overlay is deliberately simple, and deliberately not a claim to model
  real short-selling mechanics: there is no hard-to-borrow / locate-
  availability data anywhere in this codebase, so it cannot know which
  candidate names are actually shortable, at what cost, or under what
  recall risk -- a material gap, and a particularly pointed one here since
  borrow scarcity is usually worst in exactly the small, thin names
  `microcap_tilt` and the ML sleeve's size features tilt toward.
  `short_minimum_market_cap` (default $300M) is a partial, mechanical
  mitigant -- it keeps the deepest microcap tail out of the short
  candidate set -- not a substitute for real borrow data. Its execution-cost
  estimate reuses the parent's own spread/temporary/permanent impact cost
  functions (`_spread_vector`, `_impact_cost_arrays`) against a realized-
  volatility proxy computed directly from the trailing return window, since
  the parent's own specific-volatility risk model is only computed for the
  names it selects. The overlay is also stateless across rebalances -- it
  does not track a persistent short book the way the long engine tracks
  `previous_actual`, so its turnover and cost are a same-rebalance,
  build-from-scratch estimate, not a true holding-period cost. See
  `_short_overlay_weights` for the full construction.
"""

from collections import deque
from dataclasses import dataclass, field as dataclass_field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import LinearConstraint, linprog, minimize
from scipy.spatial.distance import squareform
from scipy.optimize import brentq
from scipy.stats import f as f_distribution
from scipy.stats import kurtosis, ncf, norm, skew, spearmanr
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from akm_hrp.allocators.dynamic_barra_alpha import (
    _EPS,
    _current_exposures,
    _ewma_covariance,
    _latest_pit_assets,
    _neutralize,
    _project_box_simplex,
    _rank_normal,
    _robust_zscore,
)
from akm_hrp.allocators.retail_alpha_mpc import (
    RetailAlphaMPCAllocator,
    RetailAlphaMPCConfig,
    RetailAlphaMPCDiagnostics,
    _capped_normalize,
    _field,
    _impact_cost_arrays,
    _mean_available,
    _spread_vector,
)
from akm_hrp.cov.ensemble import covariance_to_correlation
from akm_hrp.hrp.allocation import hrp_allocate, risk_contribution_regularize
from akm_hrp.hrp.trees import quasi_diagonalize
from akm_hrp.overlay.bounds import apply_bounds

_BASE_SIGNALS = (
    "momentum_12_1",
    "post_earnings_drift",
    "quality_value_carry",
    "liquid_reversal",
    "retail_agility",
)
_ML_SIGNAL = "ml_interaction_ensemble"
# Small-minus-big is deliberately absent from _BASE_SIGNALS: none of the
# parent allocator's five economic signals is a size factor, which is the
# gap a cross-section of size-effect research points at directly. See the
# size-feature block in `_ml_design` for the sourcing and the caveats.
_SIZE_FEATURES = (
    "size_small_minus_big",
    "size_tail_kink",
    "size_x_quality",
)
# Standalone microcap / good-carry-bad-carry tilts -- see module docstring
# and `_tilt_signal_panel` for the full rationale. Unlike the size features
# below (which are ML-only inputs), these are entries in `signal_names`:
# each gets its own online IC-shrunk weight from the parent's existing
# machinery, not just a hidden contribution to the blended ML score.
_TILT_SIGNALS = ("microcap_tilt", "carry_quality_tilt", "robust_fundamental_carry")
# Modest priors in the same spirit as retail_agility's 0.006: all three tilts
# are narrower, more concentrated slices of ideas the panel already
# represents (size, carry) rather than a wholly new information source, so
# they start smaller than the broadest base signals and let the online
# IC-shrinkage machinery do the rest of the work from live data.
_TILT_IC_PRIORS = {
    "microcap_tilt": 0.008,
    "carry_quality_tilt": 0.010,
    "robust_fundamental_carry": 0.009,
}
# Quality proxies used to gate "good" vs "bad" carry -- the same fields (and
# directions) `quality_value_carry` folds shareholder_carry into, so a name
# doesn't get a materially different notion of "quality" from one signal to
# the next.
_QUALITY_GATE_COLUMNS = (
    ("gross_profitability", 1.0),
    ("return_on_assets", 1.0),
    ("cash_return_on_assets", 1.0),
    ("accruals_to_assets", -1.0),
    ("leverage", -1.0),
)
# "Investment" leg (q-factor style: firms that invest conservatively tend to
# earn higher subsequent returns) used to build `robust_fundamental_carry` --
# deliberately a *different* fundamental axis from the quality gate above, so
# the two carry-adjacent tilts aren't just re-ranking the same information.
_INVESTMENT_GATE_COLUMNS = (
    ("asset_growth", -1.0),
    ("sales_growth", -1.0),
)
# Minimum trailing weekly observations required before a name's realized
# skewness is trusted for the carry_quality_tilt skew decomposition below.
_SKEW_MIN_OBSERVATIONS = 26
_ML_FEATURES = (
    *_BASE_SIGNALS,
    "momentum_x_earnings",
    "earnings_x_quality",
    "quality_x_agility",
    "momentum_x_reversal",
    "quality_conviction",
    *_SIZE_FEATURES,
    *_TILT_SIGNALS,
    "microcap_x_carry_quality",
)
# Rank-normal cutoff for the 90th percentile -- used to give the size tail
# feature a kink at roughly the smallest decile (see _ml_design).
_SIZE_TAIL_Z = 1.2816
# Minimum fraction of the cross-section that must have usable market-cap data
# before the size features are trusted for that formation date; below this,
# small-cap coverage is exactly the kind of gap the Datastream/German-equity
# data-quality literature warns about, so the feature is disabled outright
# rather than ranked on a compromised sample (see _ml_design).
_SIZE_MIN_COVERAGE = 0.5


@dataclass(frozen=True)
class RetailAlphaMLMPCConfig(RetailAlphaMPCConfig):
    """Predeclared capacity, learning-speed, and refit-cadence controls for the ML sleeve.

    Cross-sections, rather than individual securities, are the unit of the
    rolling history buffer: each completed formation date contributes one
    entry, oldest entries drop off once the buffer hits
    `ml_max_training_cross_sections`, and this bounds memory and refit cost
    independently of universe size.
    """

    ml_max_training_cross_sections: int = 252
    ml_fast_halflife_rebalances: float = 6.0
    ml_slow_halflife_rebalances: float = 24.0
    ml_minimum_training_cross_sections: int = 12
    ml_retrain_every_n_rebalances: int = 5
    ml_target_clip: float = 2.5
    ml_prediction_clip: float = 4.0
    ml_gbm_n_estimators: int = 100
    ml_gbm_max_depth: int = 2
    ml_gbm_learning_rate: float = 0.05
    ml_gbm_subsample: float = 0.8
    ml_gbm_random_state: int = 0
    # A third, structurally different vote alongside fast/slow GBM: an
    # unweighted ridge fit on the whole buffer. Two GBMs differing only in
    # sample_weight decay are one model family fit twice, not real ensemble
    # diversity; a linear model contributes a genuinely different inductive
    # bias (smooth, no interaction discovery of its own) and is cheap enough
    # to refit every time the trees do.
    ml_linear_ridge_alpha: float = 5.0
    # Consensus requires at least this many of the three models' predictions
    # to share the majority sign (2 of 3 by default -- a majority vote, not
    # unanimity, since unanimity across three independent-ish models is a
    # much stricter bar than the original two-model AND).
    ml_consensus_min_votes: int = 2
    # Eight competing signals are present now (five inherited base signals,
    # microcap_tilt, carry_quality_tilt, and the ML ensemble); keep any
    # single signal from dominating the blend.
    maximum_signal_weight: float = 0.30

    # --- Short overlay -------------------------------------------------
    # Opt-in and off by default: the inherited long-only engine (HRP prior,
    # factor risk model, multi-period optimizer) is completely unmodified
    # either way. See the module docstring and `_short_overlay_weights` for
    # the full construction and its caveats (no borrow/locate data, no
    # persistent short-book state across rebalances).
    short_overlay_enabled: bool = False
    # Total |short weight| budget, e.g. 0.15 -> book gross exposure becomes
    # roughly 1.0 (long) + 0.15 (short) = 1.15, net roughly 1.0 - 0.15 = 0.85.
    short_gross_budget: float = 0.15
    short_maximum_added_assets: int = 10
    short_max_added_per_sector: int = 2
    short_max_weight_per_name: float = 0.03
    # Composite (IC-weighted, rank-normal) score a name must clear on the
    # downside to be a short candidate. Deliberately not just the negative of
    # `minimum_candidate_score` -- good shorts only need the model's most
    # negative view, not a symmetric mirror of what makes a good long.
    short_minimum_candidate_score: float = -0.50
    # Deliberately well above the deepest microcap tail this file otherwise
    # tilts toward (see module docstring): with no borrow/locate data, this
    # is a partial, mechanical mitigant against recommending shorts in names
    # that are likely difficult or impossible to actually borrow.
    short_minimum_market_cap: float = 3.0e8

    # --- Max-assets / min-weight coupling constraint ---------------------
    # A hard cap on total book size, and a lower weight floor for any name
    # that IS held, sized so the floor is exactly reachable at full
    # occupancy: min_weight = 1 / max_total_assets. See
    # `_apply_min_weight_floor` for why this is enforced as a post-solve
    # prune-and-reproject step rather than a solver constraint (SLSQP can't
    # express "either exactly 0 or >= floor" as a convex bound).
    ml_max_total_assets: int = 40

    # --- Enhanced predictive risk model -----------------------------------
    # Multi-scale factor covariance: three EWMA halflives blended together,
    # the weekly-cadence analogue of a 5d/21d/63d daily multi-horizon scheme
    # (this codebase rebalances weekly, so "trading days" become weeks).
    ml_factor_halflife_short_weeks: float = 1.0
    ml_factor_halflife_medium_weeks: float = 4.0
    ml_factor_halflife_long_weeks: float = 13.0
    ml_factor_scale_weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
    # Ledoit-Wolf shrinkage (closed-form, sklearn.covariance.ledoit_wolf --
    # not an approximation, this is the real 2004 analytic estimator) blended
    # with the multi-scale EWMA factor covariance above.
    ml_factor_shrinkage_blend: float = 0.5
    # EWMA/RiskMetrics-style proxy for the DCC residual-correlation
    # recursion (Q_t = (1-decay) e e' + decay Q_{t-1}, normalized to a
    # correlation matrix): a standard, well-documented tractable stand-in
    # for a fully MLE-fit DCC-GARCH, consistent with the rest of this
    # codebase's no-black-box philosophy. See module docstring.
    ml_residual_dcc_halflife_weeks: float = 8.0
    # HAR-style specific-variance horizons (short/medium/long realized mean)
    # plus an EWMA-recursive "clustering" term, blended via the same
    # ridge-regularized cross-sectional regression the parent's own
    # _predictive_factor_risk_model already uses.
    ml_specific_ewma_halflife_weeks: float = 6.0

    # --- Tax-aware, vol-scaled turnover cost -------------------------------
    # Tax cost applies only to the realized-gain portion of a position
    # *reduction* (a sell), never to a buy or an unrealized position:
    # tax_rate_per_name * unrealized_gain_frac * reduction_notional. Both
    # per-name inputs are supplied via `set_tax_lot_info`, not this frozen
    # config, since they are per-rebalance holdings data, not a model
    # parameter -- see `set_tax_lot_info`.
    ml_tax_aware_costs_enabled: bool = False
    # Multiplies the temporary-impact term by
    # 1 + ml_vol_scale_coefficient * (recent_realized_vol / baseline_vol - 1),
    # clipped to stay positive, so trading is modeled as costing more in a
    # volatile regime than the baseline HAR/EWMA risk model alone implies.
    ml_vol_scale_coefficient: float = 0.5
    ml_vol_scale_short_weeks: int = 4
    ml_vol_scale_baseline_weeks: int = 26

    # --- VSK-tilted Return-Adjusted HRP (Boudt et al. 2020 + RA-HRP) ------
    # Replaces the parent's plain inverse-variance recursive bisection with
    # (a) a Return-Adjusted split rule (floored cluster Sharpe scores instead
    # of inverse cluster variance, per Noguer i Alonso's RA-HRP) and (b) a
    # bounded post-hoc tilt in the direction of higher estimated skewness and
    # lower excess kurtosis (Boudt, Cornilly, Van Holle & Willems,
    # "Algorithmic portfolio tilting to harvest higher moment gains",
    # Heliyon 2020), in place of equal-weighting the two objectives.
    ml_ra_hrp_enabled: bool = True
    # How much the RA-HRP split defers to cluster Sharpe vs. inverse
    # variance (0 = identical to the parent's plain HRP, 1 = pure
    # floored-Sharpe splits).
    ml_ra_hrp_return_blend: float = 0.5
    # Floor applied to a cluster's estimated Sharpe before it enters a split,
    # matching RA-HRP's "floored cluster Sharpe" -- prevents a single noisy,
    # deeply negative expected-return estimate from dominating a split.
    ml_ra_hrp_sharpe_floor: float = -1.0
    # Maximum L1 weight moved away from the RA-HRP prior by the VSK tilt, a
    # direct, bounded step rather than an unconstrained higher-moment
    # optimization -- consistent with keeping this tractable at O(n) using
    # each asset's own marginal skewness/kurtosis rather than full
    # multivariate co-skewness/co-kurtosis tensors.
    ml_vsk_tilt_strength: float = 0.15
    ml_vsk_kurtosis_penalty: float = 0.5

    # --- Growth-optimal (Kelly) sizing of the long-short factor sleeve ---
    #
    # The two switches are deliberately separate because they answer different
    # questions and the literature says only one of them is likely to pay.
    #
    # `ml_kelly_mix_enabled` changes HOW the signals are blended: it replaces
    # the cross-sectional Spearman correlation of signal *scores* (which
    # measures signal overlap) with the time-series covariance of realized
    # long-short factor-portfolio *returns* (which is the object growth
    # optimality is actually defined over). Reschenhofer's grid-searched
    # optimal factor weights beat equal weighting by only ~0.08 Sharpe out of
    # sample, and DeMiguel-Garlappi-Uppal make the same point, so this is not
    # expected to be where the value is.
    #
    # `ml_kelly_scale_enabled` changes HOW MUCH is bet in total. The optimizer
    # solves min 0.5*risk_aversion*w'Sw - alpha_strength*alpha'w, whose FOC is
    # w = (alpha_strength/risk_aversion) * S^-1 alpha; growth-optimal is
    # w = S^-1 alpha, so that ratio *is* the Kelly fraction and is currently
    # set by two unrelated hand-tuned knobs. This switch makes it the derived
    # c* below instead. This is the half with a real theoretical claim behind
    # it.
    ml_kelly_mix_enabled: bool = False
    ml_kelly_scale_enabled: bool = False
    # Rolling buffer of realized factor-portfolio returns. 260 weeks = 5y, the
    # same horizon the risk model already treats as the useful memory of the
    # market; below `ml_kelly_minimum_observations` Kelly stays switched off
    # entirely rather than sizing off a handful of points.
    ml_kelly_buffer_weeks: int = 260
    ml_kelly_minimum_observations: int = 52
    # Hard cap on the derived Kelly fraction, as a last-resort guard only. The
    # theory already drives c* to zero when the signal is indistinguishable
    # from noise, and on the verified Monte Carlo c* lands near 0.11 for k=9,
    # T=260 at a true annualized Sharpe of 0.5 -- an order of magnitude below
    # this cap, which therefore should never bind. It exists so that a
    # pathological covariance estimate cannot produce a levered book.
    ml_kelly_maximum_fraction: float = 0.50
    # Which growth-optimal estimator to use.
    #   "ridge"      Kozak-Nagel-Santosh (Omega + gamma I)^-1 mu -- shrinks each
    #                principal component by how well it is estimated. Simulation
    #                puts this at ~43% of oracle log-growth versus ~6% for the
    #                scalar rule, and it loses 12x less on pure noise than it
    #                gains on real signal. Recommended.
    #   "confidence" scalar fractional Kelly driven by a 95% lower confidence
    #                bound on theta^2. Has a hard false-positive guarantee, but
    #                buys it by discarding most of the edge; kept because that
    #                guarantee is occasionally the property you want.
    ml_kelly_estimator: str = "ridge"
    # kappa: prior belief about the maximum annualized Sharpe attainable from
    # the whole factor set. An economic statement, not a tuned constant.
    # Growth is flat over [0.5, 1.0]; the failure mode is asymmetric, so err
    # low -- too small forgoes edge, too large (>= 2) makes growth negative.
    ml_kelly_prior_maximum_sharpe: float = 0.5
    ml_kelly_periods_per_year: float = 52.0

    # --- Crowding-regime-conditioned kappa (pre-registered, see
    # crowding_kappa_preregistration.md) ---
    #
    # `kappa` above is a static prior. MSCI's account of the June-July 2025
    # quant drawdown ("Unraveling Summer 2025's Quant Fund Wobble") is a
    # concrete case where realized factor returns ran up to ~2x forecast and
    # the crowding gauge moved *before* the damage was fully realized -- a
    # leading signal that a slow, 5y-window covariance estimate cannot supply
    # on its own. This shrinks kappa when the factor set's own realized
    # co-movement (Fisher-z-averaged pairwise correlation of the same
    # `_factor_return_buffer` already collected for the ridge, over a rolling
    # `ml_kelly_crowding_window_weeks` window) makes an abnormal move
    # relative to its own expanding history -- direction-agnostic, because
    # MSCI's own episode showed that correlation *falling* was the unwind
    # signature, not correlation spiking, so the sign cannot be fixed a
    # priori. Zero new data: this is a derived read of a buffer that already
    # exists. Default off; the ridge above is unchanged until this is
    # deliberately enabled, matching `ml_kelly_mix_enabled` /
    # `ml_kelly_scale_enabled`.
    ml_kelly_crowding_kappa_enabled: bool = False
    # One quarter: matches the ~8-week duration of the MSCI-documented
    # episode with room either side, and is the standard "short regime"
    # convention in the crowding literature this is drawn from. Fixed by
    # stated convention, not fit to this codebase's own returns.
    ml_kelly_crowding_window_weeks: int = 13
    # Weeks of prior rolling-correlation history required before the z-score
    # baseline is trusted; below this, kappa is left unadjusted (kappa_0).
    ml_kelly_crowding_burn_in_weeks: int = 26
    # Squash rate: chosen so a 3-sigma correlation-regime shock exactly
    # halves kappa, with no penalty inside 1 sigma of ordinary noise --
    # ln(2)/2, fixed by that stated convention.
    ml_kelly_crowding_beta: float = float(np.log(2.0) / 2.0)

    # Sizing control. Runs the full selection stack -- balanced core, admission,
    # sector diversification, the max-assets trim, participation limits, the
    # exit schedule and the min-weight floor -- then equal-weights the surviving
    # book instead of solving the multi-period optimizer.
    #
    # This is deliberately NOT the same benchmark as the `equal_weight` model in
    # compare_models.py, which is 1/N over the entire eligible universe. That one
    # answers "does the model beat naive diversification?"; this one answers
    # "does the optimizer beat naive weighting of the model's own picks?" -- the
    # second question is the one that tells you whether the risk model, alpha
    # and Kelly sizing are earning their keep, because it holds selection fixed.
    # Run both: the pair decomposes performance into selection and sizing.
    ml_equal_weight_benchmark: bool = False


def _tilt_signal_panel(
    returns: pd.DataFrame,
    snapshot: pd.DataFrame,
    exposures: pd.DataFrame,
    market_caps: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build `microcap_tilt`, `carry_quality_tilt`, and `robust_fundamental_carry`
    as standalone signals.

    Mirrors the parent's own `_retail_signal_panel` construction pattern
    (raw score -> per-name mask -> market/industry neutralize -> cap-weighted
    residualize), and mirrors `retail_edge_mpc._clean_edge_signals` for the
    cleaning step specifically, so these signals are held to the same causal,
    point-in-time standard as everything else in the panel. See the module
    docstring for the economic rationale behind each.
    """

    assets = market_caps.index

    # --- microcap_tilt ---------------------------------------------------
    log_cap = np.log(market_caps.clip(lower=1.0))
    size_z = _robust_zscore(log_cap)
    size_rank = _rank_normal(-log_cap)
    adv = _field(snapshot, "dollar_volume_20d", assets)
    # A light liquidity floor -- not `retail_agility`'s full multiplicative
    # gate -- keeps this out of the truly untradeable tail without
    # throttling the tilt back down to capacity-constrained strength.
    liquidity_floor = adv.notna() & (adv.fillna(0.0) > 0.0)
    raw_microcap = (-size_z).clip(lower=0.0) + (size_rank - _SIZE_TAIL_Z).clip(
        lower=0.0
    )

    # --- carry_quality_tilt ------------------------------------------------
    carry = _field(snapshot, "shareholder_carry", assets)
    quality_parts: list[pd.Series] = []
    quality_columns: list[str] = []
    for column, direction in _QUALITY_GATE_COLUMNS:
        if column in snapshot:
            quality_columns.append(column)
            values = _field(snapshot, column, assets)
            quality_parts.append(
                direction * _rank_normal(values).where(values.notna())
            )
    quality_composite = _mean_available(quality_parts, assets)
    # Same sigmoid-of-a-composite-zscore shape the parent uses for its own
    # liquidity_gate, applied here to quality instead: high raw carry paired
    # with weak quality is damped toward zero rather than ranked as if it
    # were the same bet as high carry backed by real profitability.
    quality_gate = (1.0 / (1.0 + np.exp(-quality_composite))).clip(0.15, 0.95)

    # Quality-decomposed skewness: Bekaert & Panayotov's actual "good vs bad
    # carry" classification is a Sharpe/skewness split across ~30 years of
    # daily FX quotes -- there's no equivalent per-stock skew/crash history
    # here, but there IS a trailing realized-return series per name. Decompose
    # its skewness into the part explained by quality (a cap-weighted linear
    # fit of realized skew on the same quality_composite above) and use the
    # *fitted* component as a second, moment-based gate alongside the
    # fundamentals-based quality_gate: carry paired with quality-explained
    # positive skew is treated as more durably "good" than carry paired with
    # quality-explained negative skew, independent of the raw fundamentals
    # gate already applied.
    trailing_returns = returns.reindex(columns=assets).iloc[
        -min(104, len(returns)) :
    ]
    observation_count = trailing_returns.notna().sum()
    has_skew_history = observation_count >= _SKEW_MIN_OBSERVATIONS
    realized_skew = pd.Series(
        skew(trailing_returns.to_numpy(dtype=float), axis=0, nan_policy="omit"),
        index=assets,
    )
    realized_skew = realized_skew.where(has_skew_history)
    skew_names = assets[has_skew_history.to_numpy(dtype=bool) & quality_composite.notna().to_numpy()]
    if len(skew_names) >= 5:
        cap_weight = np.sqrt(
            market_caps.reindex(skew_names)
            .fillna(market_caps.median())
            .clip(lower=_EPS)
            .to_numpy(dtype=float)
        )
        design = np.column_stack(
            [np.ones(len(skew_names)), quality_composite.reindex(skew_names).to_numpy(dtype=float)]
        )
        target = realized_skew.reindex(skew_names).to_numpy(dtype=float)
        coefficient = np.linalg.lstsq(
            design * cap_weight[:, None], target * cap_weight, rcond=1e-8
        )[0]
        fitted_skew = pd.Series(design @ coefficient, index=skew_names).reindex(
            assets
        )
    else:
        fitted_skew = pd.Series(np.nan, index=assets)
    quality_explained_skew_z = _robust_zscore(fitted_skew)
    # Neutral (gate = 1.0, i.e. no skew adjustment) wherever there isn't
    # enough trailing history to trust a skewness estimate -- a data gap
    # should not silently damp or boost the fundamentals-only gate.
    skew_gate = (1.0 / (1.0 + np.exp(-0.5 * quality_explained_skew_z))).where(
        fitted_skew.notna(), 1.0
    ).clip(0.5, 1.5)

    raw_carry = _rank_normal(carry) * quality_gate * skew_gate

    # --- robust_fundamental_carry -------------------------------------------
    # A deliberately different construction from carry_quality_tilt: a
    # robust (winsorized via rank_normal's 1st/99th-percentile clip),
    # multi-fundamental blend of raw carry with the q-factor "investment"
    # leg (conservative asset/sales growth), rather than the profitability
    # /leverage-based quality gate above. Unlike carry_quality_tilt this is
    # additive, not multiplicatively gated -- it is meant to capture carry
    # that is robustly *supported* by conservative capital allocation, a
    # different economic story from "carry that survives a quality filter".
    investment_parts: list[pd.Series] = []
    investment_columns: list[str] = []
    for column, direction in _INVESTMENT_GATE_COLUMNS:
        if column in snapshot:
            investment_columns.append(column)
            values = _field(snapshot, column, assets)
            investment_parts.append(
                direction * _rank_normal(values).where(values.notna())
            )
    investment_composite = (
        _mean_available(investment_parts, assets)
        if investment_columns
        # No asset_growth/sales_growth coverage this date: fall back to carry
        # alone rather than silently diluting it against a bogus all-zero
        # "investment" series (`_mean_available([], ...)`'s own default).
        else pd.Series(np.nan, index=assets)
    )
    raw_fundamental_carry = _mean_available(
        [_rank_normal(carry).where(carry.notna()), investment_composite],
        assets,
    )

    raw = pd.DataFrame(
        {
            "microcap_tilt": _rank_normal(raw_microcap),
            "carry_quality_tilt": _rank_normal(raw_carry),
            "robust_fundamental_carry": _rank_normal(raw_fundamental_carry),
        },
        index=assets,
    ).fillna(0.0)

    masks = pd.DataFrame(
        {
            "microcap_tilt": (
                _field(snapshot, "market_cap_usd", assets).notna() & liquidity_floor
            ),
            "carry_quality_tilt": carry.notna() & bool(quality_columns),
            "robust_fundamental_carry": (
                carry.notna() | investment_composite.notna()
            ),
        },
        index=assets,
    ).fillna(False).astype(bool)

    neutral_columns = [
        column
        for column in exposures
        if column == "MARKET" or column.startswith("IND_")
    ]
    neutral_exposures = exposures.loc[:, neutral_columns]
    cleaned: dict[str, pd.Series] = {}
    for column in raw:
        available = masks[column]
        result = pd.Series(0.0, index=assets)
        if int(available.sum()) < 3:
            cleaned[column] = result
            continue
        names = assets[available.to_numpy()]
        x_frame = neutral_exposures.loc[names]
        caps = market_caps.reindex(names)
        ranked = _neutralize(raw.loc[names, column], x_frame, caps)
        x = x_frame.to_numpy(dtype=float)
        y = ranked.to_numpy(dtype=float)
        root_weight = np.sqrt(
            caps.fillna(caps.median()).clip(lower=_EPS).to_numpy(dtype=float)
        )
        coefficient = np.linalg.lstsq(
            x * root_weight[:, None], y * root_weight, rcond=1e-8
        )[0]
        residual = pd.Series(y - x @ coefficient, index=names)
        result.loc[names] = residual / max(float(residual.std(ddof=0)), _EPS)
        cleaned[column] = result

    coverage = masks.mean(axis=0).reindex(_TILT_SIGNALS).fillna(0.0)
    return pd.DataFrame(cleaned, index=assets), coverage, masks


def _ml_design(signals: pd.DataFrame, market_caps: pd.Series) -> pd.DataFrame:
    """Create a fixed, low-dimensional nonlinear design from cleaned signals."""

    base = signals.reindex(columns=_BASE_SIGNALS).fillna(0.0).clip(-4.0, 4.0)
    design = base.copy()
    design["momentum_x_earnings"] = (
        base["momentum_12_1"] * base["post_earnings_drift"]
    )
    design["earnings_x_quality"] = (
        base["post_earnings_drift"] * base["quality_value_carry"]
    )
    design["quality_x_agility"] = (
        base["quality_value_carry"] * base["retail_agility"]
    )
    design["momentum_x_reversal"] = (
        base["momentum_12_1"] * base["liquid_reversal"]
    )
    # A signed square lets the model distinguish weak from unusually strong
    # quality/value observations without adding an unconstrained feature search.
    quality = base["quality_value_carry"]
    design["quality_conviction"] = quality * quality.abs()

    # --- Size / small-cap features ----------------------------------------
    # Peek (2019, "A Study of Differences in Returns Between Large and Small
    # Companies in Europe", 17 W. European markets, 1990-2018) finds a size
    # premium that (a) survives beta control and a distress ("high financial
    # risk") filter, and (b) is strongly non-linear -- essentially flat
    # across the top nine size deciles and then jumping sharply in the
    # smallest decile. Serrano & Hoesli (2007) similarly find SMB to be a
    # persistent, if time-varying, positive predictor of returns for a
    # related hybrid asset class, with book-to-market (already this
    # allocator's `quality_value_carry`) contributing alongside it.
    #
    # Caveat worth keeping in mind: Brueckner (2013) documents that the sign
    # of the German size effect actually *reverses* over 1990-2007, and ties
    # much of the confusion in the literature to Datastream-specific data
    # errors that concentrate in small caps (missing dividend adjustments,
    # bad number-of-shares series feeding market cap directly). Nothing here
    # hard-codes "small is long" for that reason -- the sign is left for the
    # fast/slow models and their sign-agreement gate to learn from live data,
    # the same as every other feature, so a period or market where size
    # reverses is something the model can pick up rather than something this
    # code assumes away.
    #
    # A composite "fundamental" size measure (Peek uses the first principal
    # component of log market cap, invested capital, book equity, total
    # assets, sales, and employee count) is more robust than market cap
    # alone, since market cap is mechanically entangled with the discount
    # rate used to produce it (Berk, 1995/1997). This hook only receives
    # `market_caps`, not raw fundamentals, so it uses market cap alone; if
    # `snapshot`/`exposures` carries sales, total assets, or book equity,
    # blending those in before ranking would be a direct upgrade here.
    log_cap = np.log(market_caps.reindex(signals.index).clip(lower=1e-6))
    coverage = float(log_cap.notna().mean()) if len(log_cap) else 0.0
    if coverage < _SIZE_MIN_COVERAGE:
        # Not enough market-cap coverage to trust a cross-sectional rank this
        # date -- a data outage or a small-cap coverage gap should silently
        # disable the feature, not silently rank whatever partial sample
        # happens to be present.
        design["size_small_minus_big"] = 0.0
        design["size_tail_kink"] = 0.0
        design["size_x_quality"] = 0.0
    else:
        size_rank = _rank_normal(-log_cap.dropna())
        size = size_rank.reindex(signals.index).fillna(0.0).clip(-4.0, 4.0)
        design["size_small_minus_big"] = size

        # Explicit convex tail feature so a shallow tree doesn't have to
        # rediscover the decile-10 kink from a single linear split: zero away
        # from the small-cap tail, ramping up only inside roughly the
        # smallest decile of the cross-section.
        design["size_tail_kink"] = (size - _SIZE_TAIL_Z).clip(lower=0.0)

        # Peek finds the premium present both with and without a distress
        # filter, but cleaner among financially healthy firms. This sleeve
        # has no raw fundamentals to build an independent distress filter
        # causally, so it interacts size with the allocator's existing
        # quality/value proxy instead of adding an unfiltered raw size score.
        design["size_x_quality"] = size * quality
    # -----------------------------------------------------------------------

    # --- Microcap / carry-quality tilt features -----------------------------
    # `signals` already carries `microcap_tilt` / `carry_quality_tilt` by the
    # time this runs (`_build_signal_panel` merges `_tilt_signal_panel`'s
    # output in before calling here), cleaned the same way as every other
    # column. Feeding them into the ensemble too -- on top of giving them
    # their own standalone signal weight -- lets the fast/slow/linear models
    # learn any conditional interaction between "how small" and "how good the
    # carry is" that a purely additive blend of the two standalone signals
    # can't express.
    tilt = signals.reindex(columns=_TILT_SIGNALS).fillna(0.0).clip(-4.0, 4.0)
    design["microcap_tilt"] = tilt["microcap_tilt"]
    design["carry_quality_tilt"] = tilt["carry_quality_tilt"]
    design["robust_fundamental_carry"] = tilt["robust_fundamental_carry"]
    design["microcap_x_carry_quality"] = (
        tilt["microcap_tilt"] * tilt["carry_quality_tilt"]
    )
    # -----------------------------------------------------------------------

    return design.reindex(columns=_ML_FEATURES).clip(-6.0, 6.0)


def _multi_scale_factor_covariance(
    factor_returns: np.ndarray, config: "RetailAlphaMLMPCConfig"
) -> np.ndarray:
    """Blend three EWMA factor covariances at different halflives -- the
    weekly-cadence analogue of a short/medium/long (5d/21d/63d) multi-horizon
    scheme, so a short, sharp regime shift and a slow-moving structural
    change both get represented rather than picking one lookback."""

    halflives = (
        config.ml_factor_halflife_short_weeks,
        config.ml_factor_halflife_medium_weeks,
        config.ml_factor_halflife_long_weeks,
    )
    weights = np.asarray(config.ml_factor_scale_weights, dtype=float)
    weights = weights / max(float(weights.sum()), _EPS)
    blended = np.zeros((factor_returns.shape[1], factor_returns.shape[1]))
    for halflife, weight in zip(halflives, weights):
        blended += weight * _ewma_covariance(factor_returns, halflife)
    return blended


def _shrunk_factor_covariance(
    factor_returns: np.ndarray, config: "RetailAlphaMLMPCConfig"
) -> np.ndarray:
    """Multi-scale EWMA factor covariance, shrunk toward a scaled-identity
    target using the real, closed-form Ledoit & Wolf (2004) analytically
    optimal shrinkage intensity -- not an approximation of it, just applied
    to a non-uniformly-weighted (EWMA) covariance rather than the
    equal-weighted sample covariance the original estimator assumes, which
    is the standard way practitioners combine the two. Guarantees positive
    semi-definiteness: a convex combination of a PSD EWMA covariance and a
    PSD scaled-identity target is PSD."""

    ewma_cov = _multi_scale_factor_covariance(factor_returns, config)
    n_obs, n_factors = factor_returns.shape
    if n_obs < n_factors + 2:
        return ewma_cov
    lw_intensity = float(ledoit_wolf_shrinkage(factor_returns))
    target = np.eye(n_factors) * (np.trace(ewma_cov) / max(n_factors, 1))
    blend = float(np.clip(config.ml_factor_shrinkage_blend * lw_intensity, 0.0, 1.0))
    return (1.0 - blend) * ewma_cov + blend * target


def _dcc_proxy_residual_correlation(
    standardized_residuals: np.ndarray, halflife_weeks: float
) -> np.ndarray:
    """EWMA/RiskMetrics-style proxy for a multi-asset DCC correlation
    recursion: Q_t = (1 - decay) * e_{t-1} e_{t-1}' + decay * Q_{t-1},
    normalized to a correlation matrix at the final period. This is the
    standard, widely-used tractable stand-in for a fully MLE-fit DCC-GARCH
    (Engle 2002's own paper notes the EWMA/RiskMetrics recursion as the
    integrated, alpha+beta=1 special case of DCC) -- deliberately chosen
    over fitting per-asset GARCH(1,1) models and a DCC(1,1) by maximum
    likelihood, which this codebase's other risk-model code avoids for the
    same reason: convergence-sensitive, opaque estimation in exchange for a
    refinement this signal-to-noise regime is unlikely to reward.
    `standardized_residuals` must already be divided by each asset's own
    trailing volatility (unit variance per column) before this call.
    """

    decay = float(np.exp(np.log(0.5) / max(halflife_weeks, _EPS)))
    n_obs, n_assets = standardized_residuals.shape
    if n_obs < 3:
        return np.eye(n_assets)
    # Seed the recursion with the unconditional covariance of the
    # standardized residuals (already close to a correlation matrix) rather
    # than an uninformative identity, so a short history converges faster.
    q = np.cov(standardized_residuals, rowvar=False)
    if q.shape != (n_assets, n_assets) or not np.all(np.isfinite(q)):
        q = np.eye(n_assets)
    for t in range(n_obs):
        e = standardized_residuals[t]
        q = (1.0 - decay) * np.outer(e, e) + decay * q
    d = np.sqrt(np.clip(np.diag(q), _EPS, None))
    corr = q / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -0.999, 0.999)


def _har_specific_variance(
    residuals: np.ndarray, config: "RetailAlphaMLMPCConfig"
) -> np.ndarray:
    """Multi-horizon (HAR-style) realized-variance forecast -- short (4wk),
    medium (13wk), and long (52wk) trailing means of squared residuals --
    blended with an EWMA-recursive "clustering" term standing in for the
    GARCH component of a HAR-GARCH forecast (again the RiskMetrics/DCC-style
    integrated-EWMA proxy rather than an MLE-fit GARCH(1,1), for the same
    reason as `_dcc_proxy_residual_correlation`). Fixed horizon weights,
    matching the parent's own `_predictive_factor_risk_model`'s preference
    for fixed-weight blending over another layer of fitted parameters."""

    squared = residuals**2
    n_obs = len(squared)

    def horizon_mean(weeks: int) -> np.ndarray:
        return np.mean(squared[-min(weeks, n_obs) :], axis=0)

    short = horizon_mean(4)
    medium = horizon_mean(13)
    long = horizon_mean(52)

    decay = float(
        np.exp(np.log(0.5) / max(config.ml_specific_ewma_halflife_weeks, _EPS))
    )
    ewma_variance = squared[0].copy()
    for t in range(1, n_obs):
        ewma_variance = decay * ewma_variance + (1.0 - decay) * squared[t]

    return 0.40 * short + 0.30 * medium + 0.20 * long + 0.10 * ewma_variance


def _ml_predictive_factor_risk_model(
    returns: pd.DataFrame,
    exposures: pd.DataFrame,
    sectors: pd.Series,
    snapshot: pd.DataFrame,
    market_caps: pd.Series,
    config: "RetailAlphaMLMPCConfig",
) -> tuple[pd.DataFrame, pd.Series, float]:
    """Enhanced predictive factor risk model: multi-scale EWMA + Ledoit-Wolf
    shrunk factor covariance, a DCC-proxy multi-asset residual correlation
    (rather than assuming zero specific correlation, the classic Barra-style
    assumption the parent's own risk model makes), and a HAR-style
    multi-horizon specific-variance forecast. See the three helper functions
    above and the module docstring for the full rationale and the explicit
    choice of EWMA-based proxies over MLE-fit GARCH/DCC.

    Signature and return shape mirror the parent's own
    `_predictive_factor_risk_model` exactly (same call site shape in the
    overridden `allocate()` below), but this function is self-contained and
    does not call the parent's private implementation, since that
    implementation is exactly what's being replaced here.
    """

    assets = returns.columns
    r = returns.iloc[-config.risk_lookback_weeks :].to_numpy(dtype=float)
    b = exposures.to_numpy(dtype=float)
    cap_weight = np.sqrt(
        market_caps.reindex(assets)
        .fillna(market_caps.median())
        .clip(lower=1.0)
        .to_numpy(dtype=float)
    )
    cap_weight /= max(float(np.mean(cap_weight)), _EPS)
    gram = b.T @ (cap_weight[:, None] * b)
    ridge = 1e-6 * max(float(np.trace(gram) / max(len(gram), 1)), _EPS)
    projection = np.linalg.solve(
        gram + ridge * np.eye(len(gram)), b.T * cap_weight[None, :]
    )
    factor_returns = (projection @ r.T).T

    factor_covariance = _shrunk_factor_covariance(factor_returns, config)
    residuals = r - factor_returns @ b.T

    specific_variance = _har_specific_variance(residuals, config)
    specific_vol = np.sqrt(np.clip(specific_variance, _EPS, None))
    standardized_residuals = residuals / np.clip(
        np.sqrt(np.clip(np.mean(residuals**2, axis=0), _EPS, None)), _EPS, None
    )
    residual_correlation = _dcc_proxy_residual_correlation(
        standardized_residuals, config.ml_residual_dcc_halflife_weeks
    )
    specific_covariance = (
        specific_vol[:, None] * residual_correlation * specific_vol[None, :]
    )

    total_covariance = b @ factor_covariance @ b.T + specific_covariance
    # Numerical hygiene: a DCC-proxy correlation matrix and two independently
    # shrunk pieces summed together can pick up tiny asymmetries; symmetrize
    # and nudge the diagonal before this feeds a Cholesky/solve downstream.
    total_covariance = 0.5 * (total_covariance + total_covariance.T)
    total_covariance += 1e-10 * np.eye(len(assets))

    factor_share = float(
        np.mean(np.diag(b @ factor_covariance @ b.T))
        / max(float(np.mean(np.diag(total_covariance))), _EPS)
    )
    risk_r2 = float(np.clip(factor_share, 0.0, 1.0))

    covariance_frame = pd.DataFrame(total_covariance, index=assets, columns=assets)
    specific_volatility = pd.Series(specific_vol, index=assets)
    return covariance_frame, specific_volatility, risk_r2


def _cluster_variance(cov: pd.DataFrame, cluster: list[str]) -> float:
    sub = cov.loc[cluster, cluster].to_numpy(dtype=float)
    w = np.ones(len(cluster), dtype=float) / len(cluster)
    return float(w @ sub @ w)


def _cluster_sharpe(alpha: pd.Series, cov: pd.DataFrame, cluster: list[str], floor: float) -> float:
    w = np.ones(len(cluster), dtype=float) / len(cluster)
    cluster_alpha = float(w @ alpha.reindex(cluster).fillna(0.0).to_numpy(dtype=float))
    cluster_vol = np.sqrt(max(_cluster_variance(cov, cluster), _EPS))
    return max(cluster_alpha / max(cluster_vol, _EPS), floor)


def _ra_hrp_bisection(
    order: list[str],
    cov: pd.DataFrame,
    alpha: pd.Series,
    config: "RetailAlphaMLMPCConfig",
) -> pd.Series:
    """Return-Adjusted HRP recursive bisection (Noguer i Alonso, "Return-
    Adjusted Hierarchical Risk Parity and Schur Portfolios"): blends the
    classic inverse-cluster-variance split with a floored-cluster-Sharpe
    split, `ml_ra_hrp_return_blend` in [0, 1] controlling how much the split
    defers to estimated return (0 = identical to plain HRP). Two floored
    Sharpe ratios are combined into a split fraction via a sigmoid of their
    difference rather than a raw ratio, since a raw ratio misbehaves once
    either floored Sharpe can be negative.
    """

    weights = pd.Series(1.0, index=order, dtype=float)
    blend = float(np.clip(config.ml_ra_hrp_return_blend, 0.0, 1.0))
    floor = config.ml_ra_hrp_sharpe_floor

    def split_and_assign(cluster: list[str]) -> None:
        if len(cluster) <= 1:
            return
        mid = len(cluster) // 2
        left, right = cluster[:mid], cluster[mid:]

        var_left = max(_cluster_variance(cov, left), _EPS)
        var_right = max(_cluster_variance(cov, right), _EPS)
        variance_alloc_left = var_right / (var_left + var_right)

        sharpe_left = _cluster_sharpe(alpha, cov, left, floor)
        sharpe_right = _cluster_sharpe(alpha, cov, right, floor)
        sharpe_alloc_left = 1.0 / (1.0 + np.exp(-(sharpe_left - sharpe_right)))

        alloc_left = float(
            np.clip(
                (1.0 - blend) * variance_alloc_left + blend * sharpe_alloc_left,
                0.0,
                1.0,
            )
        )
        alloc_right = 1.0 - alloc_left

        weights.loc[left] *= alloc_left
        weights.loc[right] *= alloc_right

        split_and_assign(left)
        split_and_assign(right)

    split_and_assign(order)
    total = float(weights.sum())
    if total > _EPS:
        weights /= total
    return weights


def _vsk_tilt(
    prior: pd.Series,
    returns: pd.DataFrame,
    config: "RetailAlphaMLMPCConfig",
    upper_bound: pd.Series,
) -> pd.Series:
    """A bounded post-hoc tilt of the RA-HRP prior toward higher estimated
    skewness and lower excess kurtosis, in the spirit of Boudt, Cornilly,
    Van Holle & Willems ("Algorithmic portfolio tilting to harvest higher
    moment gains", Heliyon 2020): they recommend tilting a portfolio in the
    direction that increases its estimated mean and third central moment
    while decreasing variance and the fourth central moment.

    This uses each asset's own *marginal* skewness/excess-kurtosis as the
    tilt direction rather than the paper's full shortage-minimization
    optimization over the true portfolio-level third/fourth central moments
    (which need the full co-skewness/co-kurtosis tensors -- O(n^3)/O(n^4) to
    estimate and well beyond what a rolling weekly walk-forward window can
    support reliably for more than a handful of names). It is a tractable
    approximation consistent with the direction Boudt et al. recommend, not
    a reproduction of their exact optimization. The perturbation's total L1
    magnitude is capped at `ml_vsk_tilt_strength`, and the result is
    reprojected onto the same box-simplex the RA-HRP prior itself satisfies.
    """

    assets = prior.index
    trailing = returns.reindex(columns=assets).iloc[-min(104, len(returns)) :]
    valid_counts = trailing.notna().sum()
    has_history = (valid_counts >= _SKEW_MIN_OBSERVATIONS).to_numpy(dtype=bool)
    arr = trailing.to_numpy(dtype=float)
    asset_skew = pd.Series(
        skew(arr, axis=0, nan_policy="omit"), index=assets
    ).where(has_history, 0.0)
    asset_kurtosis = pd.Series(
        kurtosis(arr, axis=0, nan_policy="omit", fisher=True), index=assets
    ).where(has_history, 0.0)

    tilt_score = _robust_zscore(asset_skew) - config.ml_vsk_kurtosis_penalty * (
        _robust_zscore(asset_kurtosis)
    )
    tilt_direction = tilt_score - float(tilt_score.mean())
    magnitude = float(tilt_direction.abs().sum())
    if magnitude <= _EPS or config.ml_vsk_tilt_strength <= _EPS:
        return prior

    step = config.ml_vsk_tilt_strength / magnitude
    tilted = prior + step * tilt_direction
    return _project_box_simplex(
        tilted, pd.Series(0.0, index=assets), upper_bound
    )


def _ra_hrp_vsk_allocate(
    correlation: pd.DataFrame,
    covariance: pd.DataFrame,
    tree,
    asset_names: list[str],
    alpha: pd.Series,
    returns: pd.DataFrame,
    config: "RetailAlphaMLMPCConfig",
    risk_cap: float = 0.15,
) -> pd.Series:
    """Full RA-HRP + VSK-tilt pipeline: quasi-diagonalize -> RA-HRP recursive
    bisection -> risk-contribution regularization (both reused verbatim from
    the parent's own `hrp_allocate` pipeline) -> bounded VSK tilt. Signature
    mirrors `hrp_allocate` with `alpha`, `returns`, and `config` added."""

    order = quasi_diagonalize(tree, asset_names)
    ra_weights = _ra_hrp_bisection(order, covariance, alpha, config)
    regularized = risk_contribution_regularize(ra_weights, covariance, cap=risk_cap)
    if not config.ml_ra_hrp_enabled:
        return regularized
    upper_bound = pd.Series(config.max_weight, index=regularized.index)
    return _vsk_tilt(regularized, returns, config, upper_bound)


def _apply_min_weight_floor(
    weights: pd.Series,
    min_weight: float,
    upper_bound: pd.Series,
    protected: pd.Index,
) -> pd.Series:
    """Enforce `min(min_weights) = 1 / max_total_assets` as a post-solve
    prune-and-reproject step: any non-protected name below the floor is
    pruned to zero and the freed budget flows to the remaining names via the
    same box-simplex projection HRP and the optimizer already use. This is
    deliberately a cleanup, not a solver constraint -- SLSQP's continuous
    bounds can express "weight in [lower, upper]" but not "weight is exactly
    0 or at least `min_weight`" (that disjunction is combinatorial: exact
    MIQP/MINLP territory this codebase's continuous solver doesn't attempt).
    `protected` names (currently-held positions on a forced-exit schedule,
    or the balanced-core universe) are never pruned here even if under the
    floor -- that would silently override the exit schedule or the
    admission logic's own decisions, which is not what a diversification
    floor is for.
    """

    if weights.empty or min_weight <= 0.0:
        return weights
    keep = (weights >= min_weight - 1e-12) | weights.index.isin(protected)
    if bool(keep.all()):
        return weights
    pruned = weights.where(keep, 0.0)
    if float(pruned.sum()) <= _EPS:
        # Pruning would zero out the entire book -- refuse rather than
        # return an empty portfolio; leave the pre-cleanup weights in place.
        return weights
    lower = pd.Series(0.0, index=weights.index)
    # A plain box-simplex reprojection with the original (nonzero) upper
    # bounds on every name -- pruned or not -- can shift the freed budget
    # right back onto a just-pruned name: the projection only minimizes
    # distance to `pruned` subject to the box, and a small positive weight
    # there is often closer to that target than staying at 0. That would
    # silently reintroduce a below-floor weight on exactly the names this
    # function exists to zero out. Pin each pruned name's upper bound to 0
    # too, so the projection is only free to reallocate the freed budget
    # across the kept/protected names it is meant to flow to.
    reproject_upper = upper_bound.reindex(weights.index).where(keep, 0.0)
    return _project_box_simplex(pruned, lower, reproject_upper)


def _tax_aware_cost(
    trade: np.ndarray,
    previous_weights: np.ndarray,
    tax_rate: np.ndarray,
    unrealized_gain_frac: np.ndarray,
) -> float:
    """Tax cost as a capital fraction (additive with spread/temporary/
    permanent impact, which `_impact_cost_arrays` already expresses the same
    way): applies only to the realized-gain portion of a position
    *reduction*. A buy, or an increase to an existing position, never
    realizes a gain and is never taxed here; a reduction can realize at most
    the fraction of the position actually sold, hence the `clip(..., 0,
    previous_weights)`.
    """

    reduction = np.clip(-trade, 0.0, previous_weights)
    return float(np.sum(tax_rate * unrealized_gain_frac * reduction))


def _vol_scale_multiplier(
    returns: pd.DataFrame, config: "RetailAlphaMLMPCConfig"
) -> float:
    """Realized-volatility regime multiplier for the temporary-impact term:
    1 + coefficient * (recent_vol / baseline_vol - 1), clipped to stay
    positive and bounded. Uses a simple equal-weighted return proxy across
    the eligible universe -- a regime signal, not a precise estimate of this
    specific book's volatility (which is exactly what the HAR/DCC risk model
    above already estimates per-asset; this multiplier only scales the cost
    side of the objective, not the risk side)."""

    proxy = returns.mean(axis=1, skipna=True)
    recent = float(proxy.iloc[-config.ml_vol_scale_short_weeks :].std(ddof=0))
    baseline = float(proxy.iloc[-config.ml_vol_scale_baseline_weeks :].std(ddof=0))
    if not np.isfinite(baseline) or baseline <= _EPS or not np.isfinite(recent):
        return 1.0
    ratio = recent / baseline
    multiplier = 1.0 + config.ml_vol_scale_coefficient * (ratio - 1.0)
    return float(np.clip(multiplier, 0.2, 5.0))


def _shrunk_factor_covariance_from_returns(factor_returns: np.ndarray) -> np.ndarray:
    """Covariance of realized long-short factor-portfolio returns, shrunk
    toward its own diagonal.

    The target is the diagonal of the sample covariance, not a scaled
    identity: factor portfolios genuinely differ in volatility, so shrinking
    toward equal variances would import a bias this estimator does not need.
    Only the off-diagonal (correlation) structure -- the part that is both
    noisiest and, per Reschenhofer and DeMiguel-Garlappi-Uppal, least
    valuable -- is shrunk. The intensity is Ledoit-Wolf's analytic value, the
    same estimator the factor risk model above already uses, so nothing here
    is a tuned constant.
    """

    observations, factors = factor_returns.shape
    sample = np.cov(factor_returns, rowvar=False, ddof=1)
    sample = np.atleast_2d(sample)
    if factors == 1 or observations <= factors + 2:
        return np.diag(np.diag(sample)) + _EPS * np.eye(factors)
    try:
        intensity = float(ledoit_wolf_shrinkage(factor_returns, assume_centered=False))
    except Exception:
        intensity = 0.5
    intensity = float(np.clip(intensity, 0.0, 1.0))
    target = np.diag(np.diag(sample))
    shrunk = (1.0 - intensity) * sample + intensity * target
    shrunk = 0.5 * (shrunk + shrunk.T)
    return shrunk + 1e-12 * np.eye(factors)


def _bias_adjusted_squared_sharpe(
    raw_squared_sharpe: float, factors: int, observations: int
) -> float:
    """Kan-Zhou style correction for the upward bias of a plug-in squared Sharpe.

    A sample squared Sharpe computed from k series over T observations has
    expectation (T-1)(k + T*theta^2) / (T*(T-k-2)), i.e. roughly
    `theta^2 + k/T` -- it is inflated by the parameter count even when there
    is no signal at all. At k=9, T=260 and a true annualized Sharpe of 0.5,
    Monte Carlo puts the raw estimate at 0.0409 against a true 0.0048: eight
    and a half times too large. Inverting the expectation gives the estimator
    below, which the same simulation confirms is close to unbiased (0.00457
    against a true 0.00481).

    Floored at zero: a negative adjusted value means the measured performance
    is smaller than what pure estimation noise would produce, which is
    evidence of no signal, not of negative signal.
    """

    if observations <= factors + 2:
        return 0.0
    adjusted = (
        (observations - factors - 2) * float(raw_squared_sharpe) - factors
    ) / observations
    return float(max(adjusted, 0.0))


def _lower_confidence_squared_sharpe(
    raw_squared_sharpe: float,
    factors: int,
    observations: int,
    confidence: float = 0.95,
) -> float:
    """One-sided lower confidence bound on the true squared Sharpe.

    Sizing off the *point* estimate is not safe even after the bias
    correction, because the correction fixes the mean and not the dispersion.
    Simulation of pure noise (no edge whatsoever) put the point-estimate rule
    at a mean Kelly fraction of 0.10, occasionally at the cap: roughly 41% of
    noise draws bought a positive risk budget. That is the winner's curse --
    conditioning on a noisy estimate being positive selects precisely the
    upward-noise draws.

    The fix is to bet on the evidence rather than on the estimate. Under
    normality `((T-k)/k) * theta_hat^2 ~ F(k, T-k)` with non-centrality
    `lambda = T*theta^2`, so inverting that distribution gives an exact
    confidence bound. Under the null of no edge the bound exceeds zero with
    probability `1 - confidence` *by construction* -- a guarantee from the
    definition of a confidence set, not a tuned threshold. The 95% level is
    the ordinary convention, deliberately not a free parameter.

    Returns 0.0 whenever the data are consistent with no edge at all, which
    switches the factor sleeve off rather than sizing it small.
    """

    if observations <= factors + 2 or factors <= 0 or raw_squared_sharpe <= 0.0:
        return 0.0
    denominator_df = observations - factors
    observed = (denominator_df / factors) * float(raw_squared_sharpe)
    if not np.isfinite(observed) or observed <= 0.0:
        return 0.0

    # The non-central F CDF at a fixed point decreases in the non-centrality,
    # so if even lambda = 0 is not extreme enough there is no evidence at all.
    if float(f_distribution.cdf(observed, factors, denominator_df)) < confidence:
        return 0.0

    def shortfall(non_centrality: float) -> float:
        return (
            float(ncf.cdf(observed, factors, denominator_df, non_centrality))
            - confidence
        )

    upper = max(observed * factors, 1.0)
    for _ in range(60):
        if shortfall(upper) < 0.0:
            break
        upper *= 2.0
    else:
        return 0.0
    try:
        non_centrality = float(brentq(shortfall, 0.0, upper, xtol=1e-8, rtol=1e-10))
    except (ValueError, RuntimeError):
        return 0.0
    return float(max(non_centrality, 0.0) / observations)


def _growth_optimal_kelly_fraction(
    adjusted_squared_sharpe: float, factors: int, observations: int
) -> float:
    """The fraction of the plug-in Kelly bet that is actually growth-optimal.

    Scaling the plug-in rule `w_hat = Omega^-1 mu_hat` by `c` gives expected
    log-growth `c*E[w_hat'mu] - 0.5*c^2*E[w_hat'Omega w_hat]`, maximized at
    `c* = E[w_hat'mu] / E[w_hat'Omega w_hat] ~= theta^2 / (theta^2 + k/T)`:
    the share of the measured squared Sharpe that is real rather than
    estimation noise.

    This is why the whole design lives at the factor level. Monte Carlo of the
    *unscaled* plug-in rule gives negative expected log-growth at every
    realistic parameter count -- -0.017 at k=9, -0.064 at k=25, -0.185 at k=50
    (T=260, true annualized Sharpe 0.5) -- so full Kelly on estimated moments
    destroys capital. With k=9 factors c* is about 0.11; with k=50 assets it
    is about 0.016, i.e. the growth-optimal response to an asset-level Kelly
    estimate is to almost entirely ignore it.
    """

    if observations <= 0 or factors <= 0 or adjusted_squared_sharpe <= 0.0:
        return 0.0
    noise = factors / float(observations)
    return float(adjusted_squared_sharpe / (adjusted_squared_sharpe + noise))


def effective_breadth(factor_returns: np.ndarray) -> float:
    """How many genuinely independent bets the factor set actually contains.

    Participation ratio of the correlation matrix's eigenvalue spectrum,
    `(sum lambda)^2 / sum lambda^2 = k^2 / sum lambda^2`. Equals k when the
    factors are mutually uncorrelated and collapses toward 1 as they become
    redundant.

    This matters because of an asymmetry in the growth-optimal fraction
    `c* = theta^2 / (theta^2 + k/T)`. A duplicated signal raises the parameter
    count k -- and therefore the noise floor k/T -- **without** raising
    theta^2, because it adds no independent information. Redundancy is
    charged twice: once in worse estimation, once in a smaller risk budget.
    At T=260 and a true annualized Sharpe of 0.5:

        k =  4  ->  c* = 0.238
        k =  9  ->  c* = 0.122
        k = 20  ->  c* = 0.059

    So dropping four redundant signals roughly doubles the defensible bet
    without improving a single forecast. This is reported rather than silently
    substituted into the formula: k in `c*` must stay the number of parameters
    actually estimated (it is also the numerator degrees of freedom of the
    non-central F used for the confidence bound). Effective breadth tells you
    how much pruning is available; the pruning itself is a modelling decision.

    Relevant here because two of this book's signals are, on the published
    evidence, not independent bets: Novy-Marx shows price momentum's
    cross-sectional coefficient falls to t = 0.70 full-sample and t = -0.00
    over 1994-2012 once earnings surprise is included.
    """

    array = np.asarray(factor_returns, dtype=float)
    if array.ndim != 2 or array.shape[1] == 0:
        return 0.0
    finite = array[np.isfinite(array).all(axis=1)]
    factors = finite.shape[1]
    if finite.shape[0] <= factors or factors == 1:
        return float(factors)
    deviations = finite - finite.mean(axis=0)
    scale = deviations.std(axis=0, ddof=1)
    if not np.all(scale > _EPS):
        return float(factors)
    correlation = np.corrcoef(deviations / scale, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(np.atleast_2d(correlation))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    denominator = float(np.sum(eigenvalues**2))
    if denominator <= _EPS:
        return float(factors)
    return float(np.sum(eigenvalues) ** 2 / denominator)


def _fisher_mean_pairwise_correlation(window: np.ndarray) -> float:
    """Average pairwise Pearson correlation across a factor-return window.

    Averaged via Fisher's z-transform, `rho_bar = tanh(mean(arctanh(rho_ij)))`
    -- the standard convention for averaging correlations, since a plain
    arithmetic mean of correlations is biased. `nan` if fewer than two usable
    factors or too few rows to estimate a correlation at all.
    """

    rows, factors = window.shape
    if factors < 2 or rows < 3:
        return float("nan")
    scale = window.std(axis=0, ddof=1)
    if not np.all(scale > _EPS):
        return float("nan")
    correlation = np.corrcoef(window, rowvar=False)
    pairs = correlation[np.triu_indices(factors, k=1)]
    pairs = np.clip(pairs, -0.999999, 0.999999)
    pairs = pairs[np.isfinite(pairs)]
    if pairs.size == 0:
        return float("nan")
    return float(np.tanh(np.arctanh(pairs).mean()))


def _crowding_regime_kappa(
    factor_returns: np.ndarray,
    prior_maximum_sharpe: float,
    short_window: int,
    burn_in: int,
    beta: float,
) -> tuple[float, dict[str, float]]:
    """Shrink the KNS ridge's kappa prior on an abnormal co-movement shock.

    See `crowding_kappa_preregistration.md` for the full pre-registered
    specification; this is a direct implementation of it. Computes, for every
    window of `short_window` rows in the buffer, the Fisher-z-averaged
    pairwise correlation across the k live factor-return series
    (`_fisher_mean_pairwise_correlation`); the most recent such value is
    compared, in z-score terms, against the expanding history of all earlier
    such values (at least `burn_in` of them required, else kappa is returned
    unadjusted). The shock size `|z|` -- not `z` itself, deliberately
    direction-agnostic, since MSCI's own documented episode showed
    correlation *falling* as the unwind signature rather than spiking --
    squashes kappa via `kappa_0 * exp(-beta * max(0, |z| - 1))`: no penalty
    inside 1 sigma of ordinary noise, and `beta` fixed in advance so a 3-sigma
    shock exactly halves kappa. Never floored (asymptotes to 0), never
    capped, matching the "no floor, no cap" convention already used for `c*`.

    Every constant here is fixed by stated convention, not fit against this
    codebase's own returns -- see the pre-registration for why.
    """

    kappa0 = max(float(prior_maximum_sharpe), _EPS)
    diagnostics = {
        "crowding_active": 0.0,
        "crowding_rho_bar": float("nan"),
        "crowding_z": float("nan"),
        "crowding_kappa_multiplier": 1.0,
    }
    observations, factors = factor_returns.shape
    if factors < 2 or observations < short_window + burn_in:
        return kappa0, diagnostics

    rho_series = np.array(
        [
            _fisher_mean_pairwise_correlation(factor_returns[end - short_window : end])
            for end in range(short_window, observations + 1)
        ],
        dtype=float,
    )
    if rho_series.size <= burn_in or not np.isfinite(rho_series[-1]):
        return kappa0, diagnostics

    current = float(rho_series[-1])
    baseline = rho_series[:-1]
    baseline = baseline[np.isfinite(baseline)]
    if baseline.size < burn_in:
        return kappa0, diagnostics

    mean = float(baseline.mean())
    std = float(baseline.std(ddof=1))
    if std <= _EPS:
        return kappa0, diagnostics

    z = (current - mean) / std
    shock = max(0.0, abs(z) - 1.0)
    multiplier = float(np.exp(-max(beta, 0.0) * shock))
    kappa_t = kappa0 * multiplier

    diagnostics.update(
        crowding_active=1.0,
        crowding_rho_bar=current,
        crowding_z=float(z),
        crowding_kappa_multiplier=multiplier,
    )
    return kappa_t, diagnostics


def _kns_ridge_factor_weights(
    factor_returns: np.ndarray, prior_maximum_sharpe: float, periods_per_year: float
) -> tuple[np.ndarray, float]:
    """Kozak-Nagel-Santosh shrinkage of the growth-optimal factor direction.

    `b = (Omega + gamma*I)^-1 mu` with `gamma = tr(Omega) / (kappa^2 * T)`,
    where kappa is the prior root expected maximum Sharpe ratio of the factor
    set ("Shrinking the Cross-Section", JFE 2020). Their SDF coefficient `b`
    *is* the tangency/Kelly direction, so this is Bayesian Kelly with an
    economically stated prior rather than an arbitrary haircut.

    Why this and not a scalar fractional-Kelly multiplier: in the eigenbasis
    the estimator becomes `b_j = [d_j/(d_j+gamma)] * (mu_j/d_j)`, i.e. it
    shrinks each principal component in proportion to how badly that component
    is estimated. Low-eigenvalue directions -- the ones sample noise dominates
    -- are damped hard; well-estimated directions are left near full size. A
    scalar haircut cannot distinguish them and so pays for the noisy directions
    by giving up the good ones too.

    Simulated out-of-sample expected log-growth (200 draws per cell, k=9,
    T=260, true annualized Sharpe 0.5, oracle 0.00832):

        full Kelly plug-in   -0.01165     (negative: destroys capital)
        scalar fractional     0.00047
        this estimator        0.00352     (~43% of oracle)

    and on factors with *no* edge it loses 0.0003 while running 4.3x gross,
    against 0.0200 and 41x gross for the unshrunk plug-in.

    `kappa` is a genuine prior, not a fitted constant: growth is flat across
    kappa in [0.5, 1.0] and the failure mode is asymmetric -- too small merely
    forgoes edge, too large (>= 2) turns growth negative. Default conservative.
    """

    observations, factors = factor_returns.shape
    if observations < 2 or factors == 0:
        return np.zeros(factors, dtype=float), 0.0
    mean = factor_returns.mean(axis=0)
    covariance = np.cov(factor_returns, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    kappa = max(float(prior_maximum_sharpe), _EPS)
    # kappa is quoted annualized; the moments here are per-period.
    per_period_squared = (kappa**2) / max(periods_per_year, _EPS)
    gamma = float(np.trace(covariance)) / max(
        per_period_squared * observations, _EPS
    )
    ridged = covariance + gamma * np.eye(factors)
    try:
        direction = np.linalg.solve(ridged, mean)
    except np.linalg.LinAlgError:
        direction = np.linalg.pinv(ridged) @ mean
    if not np.isfinite(direction).all():
        return np.zeros(factors, dtype=float), gamma
    return direction, gamma


def _growth_optimal_factor_allocation(
    factor_returns: np.ndarray,
    maximum_fraction: float,
    estimator: str = "confidence",
    prior_maximum_sharpe: float = 0.5,
    periods_per_year: float = 52.0,
    crowding_kappa_enabled: bool = False,
    crowding_window: int = 13,
    crowding_burn_in: int = 26,
    crowding_beta: float = 0.0,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Growth-optimal weights across long-short factor portfolios.

    Returns `(mix, kelly_fraction, diagnostics)` where `mix` is the L1-
    normalized direction `Omega^-1 mu` (negative entries allowed -- a factor
    whose estimated premium is negative is a legitimate tilt away from that
    signal, not something to clip to zero) and `kelly_fraction` is the derived
    scale `c*`. The two are returned separately because they answer different
    questions and are switched on independently.

    `crowding_kappa_enabled` (ridge estimator only) additionally conditions
    the `prior_maximum_sharpe` fed to the ridge on realized cross-factor
    co-movement via `_crowding_regime_kappa` -- see that function and
    `crowding_kappa_preregistration.md`.
    """

    observations, factors = factor_returns.shape
    diagnostics = {
        "observations": float(observations),
        "factors": float(factors),
        "effective_breadth": effective_breadth(factor_returns),
        "raw_squared_sharpe": 0.0,
        "adjusted_squared_sharpe": 0.0,
        "lower_bound_squared_sharpe": 0.0,
        "kelly_fraction": 0.0,
    }
    if observations <= factors + 2 or factors == 0:
        return np.zeros(factors, dtype=float), 0.0, diagnostics

    mean = factor_returns.mean(axis=0)

    if estimator == "ridge":
        # Kozak-Nagel-Santosh: the shrinkage lives in the direction, per
        # principal component, and already carries the whole risk budget --
        # so no separate scalar Kelly fraction is applied on top. Stacking both
        # shrinks twice and, in simulation, collapses the position to zero.
        kappa_used = prior_maximum_sharpe
        crowding_diagnostics: dict[str, float] = {}
        if crowding_kappa_enabled:
            kappa_used, crowding_diagnostics = _crowding_regime_kappa(
                factor_returns,
                prior_maximum_sharpe,
                short_window=crowding_window,
                burn_in=crowding_burn_in,
                beta=crowding_beta,
            )
        direction, gamma = _kns_ridge_factor_weights(
            factor_returns, kappa_used, periods_per_year
        )
        magnitude = float(np.abs(direction).sum())
        mix = (
            direction / magnitude if magnitude > _EPS
            else np.zeros(factors, dtype=float)
        )
        covariance = np.atleast_2d(np.cov(factor_returns, rowvar=False, ddof=1))
        inverse = np.linalg.pinv(covariance)
        raw = float(mean @ inverse @ mean)

        # Both estimators must return the SAME semantic quantity here: the
        # scalar multiplier applied to alpha downstream. For the ridge that is
        # its gross exposure *relative to the unshrunk plug-in* -- "the ridge
        # takes this share of the position full Kelly would take" -- which is
        # bounded, comparable across modes, and cannot silently lever the book.
        # Returning the raw gross notional instead would multiply alpha by ~8x.
        plug_in = inverse @ mean
        plug_in_gross = float(np.abs(plug_in).sum())
        fraction = (
            magnitude / plug_in_gross if plug_in_gross > _EPS else 0.0
        )
        fraction = float(np.clip(fraction, 0.0, max(maximum_fraction, 0.0)))

        diagnostics.update(
            raw_squared_sharpe=raw,
            adjusted_squared_sharpe=_bias_adjusted_squared_sharpe(
                raw, factors, observations
            ),
            kelly_fraction=fraction,
            ridge_gamma=float(gamma),
            ridge_gross=magnitude,
            plug_in_gross=plug_in_gross,
            kappa_used=float(kappa_used),
            **crowding_diagnostics,
        )
        return mix, fraction, diagnostics

    covariance = _shrunk_factor_covariance_from_returns(factor_returns)
    try:
        direction = np.linalg.solve(covariance, mean)
    except np.linalg.LinAlgError:
        direction = np.linalg.pinv(covariance) @ mean
    if not np.isfinite(direction).all():
        return np.zeros(factors, dtype=float), 0.0, diagnostics

    raw_squared_sharpe = float(mean @ direction)
    adjusted = _bias_adjusted_squared_sharpe(
        raw_squared_sharpe, factors, observations
    )
    # The Kelly fraction is driven by the confidence bound, not the point
    # estimate: the bias correction fixes the mean but not the dispersion, and
    # on pure noise the point-estimate rule still bought a ~0.10 average risk
    # budget. `adjusted` is retained only as a reported diagnostic.
    lower_bound = _lower_confidence_squared_sharpe(
        raw_squared_sharpe, factors, observations
    )
    fraction = _growth_optimal_kelly_fraction(lower_bound, factors, observations)
    fraction = float(np.clip(fraction, 0.0, max(maximum_fraction, 0.0)))

    magnitude = float(np.abs(direction).sum())
    mix = direction / magnitude if magnitude > _EPS else np.zeros(factors, dtype=float)

    diagnostics.update(
        raw_squared_sharpe=raw_squared_sharpe,
        adjusted_squared_sharpe=adjusted,
        lower_bound_squared_sharpe=lower_bound,
        kelly_fraction=fraction,
    )
    return mix, fraction, diagnostics


class RetailAlphaMLMPCAllocator(RetailAlphaMPCAllocator):
    """Retail Alpha MPC with a causal, consensus-gated nonlinear ML signal."""

    signal_names = (*_BASE_SIGNALS, *_TILT_SIGNALS, _ML_SIGNAL)
    ic_priors = {
        **RetailAlphaMPCAllocator.ic_priors,
        **_TILT_IC_PRIORS,
        # The ML signal must earn its weight out of sample; unlike the economic
        # signals, it receives no positive information-coefficient prior.
        _ML_SIGNAL: 0.0,
    }

    def __init__(self, balanced_pit: pd.DataFrame, **kwargs) -> None:
        if kwargs.get("config") is None:
            kwargs["config"] = RetailAlphaMLMPCConfig()
        super().__init__(balanced_pit, **kwargs)
        if not isinstance(self.config, RetailAlphaMLMPCConfig):
            raise TypeError(
                "RetailAlphaMLMPCAllocator requires RetailAlphaMLMPCConfig."
            )
        self._initialize_ml_state()

    def _initialize_ml_state(self) -> None:
        # Rolling buffer of per-formation-date (design, target) arrays, bounded
        # in cross-sections rather than in individual security-observations.
        self._ml_history: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=self.config.ml_max_training_cross_sections
        )
        self._ml_training_cross_sections = 0
        self._ml_rebalances_since_retrain = 0
        self._ml_fast_model: GradientBoostingRegressor | None = None
        self._ml_slow_model: GradientBoostingRegressor | None = None
        self._ml_linear_model: Ridge | None = None
        self._previous_ml_design: pd.DataFrame | None = None
        self._previous_ml_eligible: pd.Series | None = None
        self._previous_ml_date: pd.Timestamp | None = None
        self._last_ml_training_date: pd.Timestamp | None = None
        self.last_ml_feature_importances = pd.DataFrame()
        self.last_ml_linear_coefficients = pd.Series(dtype=float)
        self.last_ml_consensus_fraction = 0.0
        self.last_short_overlay_weights = pd.Series(dtype=float)
        self.last_short_overlay_cost = 0.0
        self._tax_rate_per_name: pd.Series | None = None
        self._unrealized_gain_frac: pd.Series | None = None
        self.last_tax_cost: float = 0.0
        self.last_vol_scale_multiplier: float = 1.0
        # Realized long-short factor-portfolio returns, one row per formation
        # date. This is the series growth optimality is defined over, and it
        # is a different object from the cross-sectional signal-score
        # correlation the parent's `_signal_combination` uses.
        self._factor_return_buffer: deque[tuple[pd.Timestamp, np.ndarray]] = deque(
            maxlen=self.config.ml_kelly_buffer_weeks
        )
        self._last_factor_return_date: pd.Timestamp | None = None
        self.last_kelly_fraction: float = 0.0
        self.last_kelly_mix = pd.Series(dtype=float)
        self.last_kelly_diagnostics: dict[str, float] = {}
        self.last_vsk_tilt_l1: float = 0.0
        self.last_pruned_dust_assets: pd.Index = pd.Index([])

    def reset_state(self) -> None:
        super().reset_state()
        self._initialize_ml_state()

    def set_tax_lot_info(
        self, tax_rate_per_name: pd.Series, unrealized_gain_frac: pd.Series
    ) -> None:
        """Supply per-name tax-lot info ahead of the next `allocate()` call.

        Optional: if never called, or if `config.ml_tax_aware_costs_enabled`
        is False, tax cost is zero. Mirrors the parent's own
        `set_current_weights()` pattern -- side-channel state a caller (a
        backtest engine, or a live trading harness with real lot data) pushes
        in before calling `allocate(returns)`, rather than a parameter on
        `allocate()` itself, so `allocate(returns)` keeps exactly the
        parent's signature and stays a drop-in for anything already calling
        it positionally.
        """

        self._tax_rate_per_name = pd.Series(tax_rate_per_name, dtype=float).copy()
        self._unrealized_gain_frac = pd.Series(
            unrealized_gain_frac, dtype=float
        ).copy()

    def _update_dynamic_ics(
        self, returns: pd.DataFrame, current_date: pd.Timestamp
    ) -> None:
        """Parent IC bookkeeping, plus the realized factor-portfolio returns.

        Overridden rather than called separately so the factor-return buffer
        is updated at exactly the same causality point as the ICs: both use
        the *previous* formation date's signal panel scored against returns
        that have only now been observed. Anything else would leak.
        """

        super()._update_dynamic_ics(returns, current_date)
        self._accumulate_factor_returns(returns, current_date)

    def _accumulate_factor_returns(
        self, returns: pd.DataFrame, current_date: pd.Timestamp
    ) -> None:
        """Append one row of realized long-short factor-portfolio returns.

        For each signal, the factor portfolio is the cleaned rank-normal score
        vector normalized to unit gross notional. Those scores are already
        cross-sectionally mean-zero, so the portfolio is dollar-neutral by
        construction -- a genuine long-short factor return, which is what the
        growth-optimal math needs and what the parent never computes.
        """

        if (
            self._previous_signal_panel is None
            or self._previous_signal_date is None
            or self._last_factor_return_date == current_date
        ):
            return
        source = returns
        if (
            self._engine_return_context is not None
            and current_date in self._engine_return_context.index
        ):
            source = self._engine_return_context
        forward_rows = source.loc[
            (source.index > self._previous_signal_date)
            & (source.index <= current_date)
        ]
        if forward_rows.empty:
            return
        forward = (1.0 + forward_rows).prod(min_count=1) - 1.0
        forward = forward.dropna()
        if forward.empty:
            return

        realized = np.full(len(self.signal_names), np.nan, dtype=float)
        for position, signal in enumerate(self.signal_names):
            if signal not in self._previous_signal_panel:
                continue
            score = self._previous_signal_panel[signal]
            common = score.index.intersection(forward.index)
            if self._previous_signal_availability is not None and (
                signal in self._previous_signal_availability
            ):
                observed = (
                    self._previous_signal_availability[signal]
                    .reindex(common)
                    .fillna(False)
                    .astype(bool)
                )
                common = common[observed.to_numpy()]
            if len(common) < 8:
                continue
            values = score.reindex(common).to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0)
            # Re-centre on the surviving names so the portfolio stays
            # dollar-neutral after the availability mask has removed some.
            values = values - values.mean()
            gross = float(np.abs(values).sum())
            if gross <= _EPS:
                continue
            weights = values / gross
            realized[position] = float(
                weights @ forward.reindex(common).to_numpy(dtype=float)
            )

        if np.isfinite(realized).any():
            self._factor_return_buffer.append((current_date, realized))
        self._last_factor_return_date = current_date

    def _kelly_factor_allocation(self) -> tuple[pd.Series, float]:
        """Growth-optimal mix and scale over the currently live signals.

        Returns an empty mix and a zero fraction whenever there is not enough
        history, which is the correct behaviour rather than a fallback: with
        no evidence, the growth-optimal bet is nothing.
        """

        empty = pd.Series(dtype=float)
        if len(self._factor_return_buffer) < self.config.ml_kelly_minimum_observations:
            self.last_kelly_fraction = 0.0
            self.last_kelly_mix = empty
            self.last_kelly_diagnostics = {"observations": float(
                len(self._factor_return_buffer)
            )}
            return empty, 0.0

        matrix = np.vstack([row for _, row in self._factor_return_buffer])
        usable = np.isfinite(matrix).all(axis=0)
        names = [
            name for name, keep in zip(self.signal_names, usable, strict=True) if keep
        ]
        if len(names) < 2:
            self.last_kelly_fraction = 0.0
            self.last_kelly_mix = empty
            return empty, 0.0
        panel = matrix[:, usable]

        mix, fraction, diagnostics = _growth_optimal_factor_allocation(
            panel,
            self.config.ml_kelly_maximum_fraction,
            estimator=self.config.ml_kelly_estimator,
            prior_maximum_sharpe=self.config.ml_kelly_prior_maximum_sharpe,
            periods_per_year=self.config.ml_kelly_periods_per_year,
            crowding_kappa_enabled=self.config.ml_kelly_crowding_kappa_enabled,
            crowding_window=self.config.ml_kelly_crowding_window_weeks,
            crowding_burn_in=self.config.ml_kelly_crowding_burn_in_weeks,
            crowding_beta=self.config.ml_kelly_crowding_beta,
        )
        mix_series = pd.Series(mix, index=names)
        self.last_kelly_fraction = fraction
        self.last_kelly_mix = mix_series
        self.last_kelly_diagnostics = diagnostics
        return mix_series, fraction

    def _signal_combination(
        self,
        availability: pd.Series | None = None,
        signal_panel: pd.DataFrame | None = None,
    ) -> tuple[pd.Series, pd.Series]:
        """Blend signals by growth optimality when enabled, else defer to the parent.

        The parent computes `Omega^-1 * confidence` on the cross-sectional
        Spearman correlation of signal scores. That measures how much two
        signals pick the same stocks; it is not the covariance of the two
        strategies' returns, which is the quantity growth optimality is
        defined over. This override swaps in the latter. The Kelly *scale* is
        applied separately, to the alpha vector, because the weights returned
        here are normalized downstream.
        """

        posterior, weights = super()._signal_combination(availability, signal_panel)
        mix, _ = self._kelly_factor_allocation()
        if not self.config.ml_kelly_mix_enabled or mix.empty:
            return posterior, weights

        live = weights.index[weights > _EPS]
        usable = mix.index.intersection(live)
        if len(usable) < 2:
            return posterior, weights
        # Only the sign-positive part of the growth-optimal direction can be
        # expressed here: the downstream alpha is a non-negative-weighted sum
        # of signal scores, and `_capped_normalize` clips at zero anyway. A
        # negative growth-optimal weight therefore surfaces as an exclusion,
        # which is the honest projection of the unconstrained solution onto
        # what this pipeline can represent.
        kelly = mix.reindex(live).fillna(0.0).clip(lower=0.0)
        if float(kelly.sum()) <= _EPS:
            return posterior, weights
        blended = _capped_normalize(kelly, self.config.maximum_signal_weight)
        return posterior, blended.reindex(weights.index).fillna(0.0)

    def _learn_previous_cross_section(
        self, returns: pd.DataFrame, current_date: pd.Timestamp
    ) -> None:
        """Append the old formation date's (design, target) once its target is observable."""

        if (
            self._previous_ml_design is None
            or self._previous_ml_eligible is None
            or self._previous_ml_date is None
            or self._last_ml_training_date == current_date
        ):
            return
        source = returns
        if (
            self._engine_return_context is not None
            and current_date in self._engine_return_context.index
        ):
            source = self._engine_return_context
        forward_rows = source.loc[
            (source.index > self._previous_ml_date)
            & (source.index <= current_date)
        ]
        if forward_rows.empty:
            return
        forward = (1.0 + forward_rows).prod(min_count=1) - 1.0
        common = self._previous_ml_design.index.intersection(forward.dropna().index)
        eligible = self._previous_ml_eligible.reindex(common).fillna(False)
        common = common[eligible.to_numpy(dtype=bool)]
        if len(common) < 12:
            self._last_ml_training_date = current_date
            return

        x = self._previous_ml_design.loc[common].to_numpy(dtype=float)
        target = _rank_normal(forward.reindex(common)).clip(
            -self.config.ml_target_clip, self.config.ml_target_clip
        )
        y = target.fillna(0.0).to_numpy(dtype=float)
        # One buffer entry per formation date, regardless of how many names
        # were eligible that day -- this is what keeps the buffer's capacity
        # measured in cross-sections rather than in raw row count.
        self._ml_history.append((x, y))
        self._ml_training_cross_sections = len(self._ml_history)
        self._last_ml_training_date = current_date
        self._ml_rebalances_since_retrain += 1

        have_enough_history = (
            self._ml_training_cross_sections
            >= self.config.ml_minimum_training_cross_sections
        )
        due_for_refit = (
            self._ml_fast_model is None
            or self._ml_rebalances_since_retrain
            >= self.config.ml_retrain_every_n_rebalances
        )
        if have_enough_history and due_for_refit:
            self._refit_models()
            self._ml_rebalances_since_retrain = 0

    def _recency_weights(self, decay: float) -> np.ndarray:
        """Per-row sample weights: 1.0 for the newest buffered cross-section,
        decaying by `decay` per rebalance for each cross-section further back."""

        weights = []
        for age, (_, y) in enumerate(reversed(self._ml_history)):
            weights.append(np.full(len(y), decay**age, dtype=float))
        weights.reverse()
        return np.concatenate(weights, axis=0)

    def _refit_models(self) -> None:
        """Refit fast/slow/linear models from scratch on the current rolling
        buffer, using recency-weighted sample_weight for the trees in place
        of the old recency-weighted sufficient-statistic accumulation."""

        x = np.concatenate([entry[0] for entry in self._ml_history], axis=0)
        y = np.concatenate([entry[1] for entry in self._ml_history], axis=0)
        fast_decay = float(
            np.exp(np.log(0.5) / self.config.ml_fast_halflife_rebalances)
        )
        slow_decay = float(
            np.exp(np.log(0.5) / self.config.ml_slow_halflife_rebalances)
        )
        fast_weights = self._recency_weights(fast_decay)
        slow_weights = self._recency_weights(slow_decay)

        gbm_kwargs = dict(
            n_estimators=self.config.ml_gbm_n_estimators,
            max_depth=self.config.ml_gbm_max_depth,
            learning_rate=self.config.ml_gbm_learning_rate,
            subsample=self.config.ml_gbm_subsample,
            random_state=self.config.ml_gbm_random_state,
        )
        fast_model = GradientBoostingRegressor(**gbm_kwargs)
        fast_model.fit(x, y, sample_weight=fast_weights)
        slow_model = GradientBoostingRegressor(**gbm_kwargs)
        slow_model.fit(x, y, sample_weight=slow_weights)

        # Third vote: an unweighted ridge over the whole buffer. Deliberately
        # not recency-weighted -- the trees already cover "recent matters
        # more"; this one instead contributes a different weighting
        # philosophy (every buffered cross-section counts equally, as in a
        # rolling-window Fama-MacBeth average) plus a different functional
        # form (linear, no learned interactions), which is what makes it a
        # genuinely separate opinion rather than a third copy of the trees.
        linear_model = Ridge(alpha=self.config.ml_linear_ridge_alpha)
        linear_model.fit(x, y)

        self._ml_fast_model = fast_model
        self._ml_slow_model = slow_model
        self._ml_linear_model = linear_model
        self.last_ml_feature_importances = pd.DataFrame(
            {
                "fast": fast_model.feature_importances_,
                "slow": slow_model.feature_importances_,
            },
            index=_ML_FEATURES,
        )
        self.last_ml_linear_coefficients = pd.Series(
            linear_model.coef_, index=_ML_FEATURES
        )

    def _ml_signal_values(
        self, design: pd.DataFrame, eligible: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        score = pd.Series(0.0, index=design.index)
        mask = pd.Series(False, index=design.index)
        if (
            self._ml_fast_model is None
            or self._ml_slow_model is None
            or self._ml_linear_model is None
            or self._ml_training_cross_sections
            < self.config.ml_minimum_training_cross_sections
        ):
            return score, mask

        values = design.to_numpy(dtype=float)
        predictions = np.vstack(
            [
                self._ml_fast_model.predict(values),
                self._ml_slow_model.predict(values),
                self._ml_linear_model.predict(values),
            ]
        )
        signs = np.sign(predictions)
        # Net vote per name: +/-3 unanimous, +/-1 a 2-1 split, 0 a tie or a
        # zero-sign prediction. min_votes=2 (default) requires at least a 2-1
        # majority -- |net| >= 1 -- rather than the stricter unanimity a
        # third, independent model would otherwise impose relative to the
        # old two-model AND gate.
        net_votes = signs.sum(axis=0)
        majority_sign = np.sign(net_votes)
        required_net = 2 * self.config.ml_consensus_min_votes - 3
        agreement = (np.abs(net_votes) >= required_net) & (
            majority_sign != 0
        ) & eligible.to_numpy(dtype=bool)
        self.last_ml_consensus_fraction = float(np.mean(agreement))
        if int(agreement.sum()) < 8:
            return score, mask
        # Average only the predictions that sided with each name's majority,
        # so a lone dissenting model doesn't drag the combined score toward
        # zero.
        agrees_with_majority = signs == majority_sign[None, :]
        masked = np.where(agrees_with_majority, predictions, np.nan)
        combined = np.clip(
            np.nanmean(masked, axis=0),
            -self.config.ml_prediction_clip,
            self.config.ml_prediction_clip,
        )
        names = design.index[agreement]
        score.loc[names] = _rank_normal(pd.Series(combined[agreement], index=names))
        mask.loc[names] = True
        return score, mask

    def _build_signal_panel(
        self,
        returns: pd.DataFrame,
        snapshot: pd.DataFrame,
        exposures: pd.DataFrame,
        market_caps: pd.Series,
        observed_returns: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        current_date = pd.Timestamp(returns.index[-1])
        self._learn_previous_cross_section(returns, current_date)

        # Delegate to the parent's own hook rather than importing its private
        # signal-construction helper directly, so this stays correct even if
        # the base class's internals change.
        signals, coverage, masks = super()._build_signal_panel(
            returns, snapshot, exposures, market_caps, observed_returns
        )

        # Merge in the three standalone tilt signals before building the ML
        # design, so the ensemble can see them as features on top of each
        # earning its own IC-shrunk weight below.
        tilt_signals, tilt_coverage, tilt_masks = _tilt_signal_panel(
            returns, snapshot, exposures, market_caps.reindex(returns.columns)
        )
        for name in _TILT_SIGNALS:
            signals[name] = tilt_signals[name]
            masks[name] = tilt_masks[name]
            coverage.loc[name] = tilt_coverage[name]

        design = _ml_design(signals, market_caps)
        eligible = masks.any(axis=1)
        ml_score, ml_mask = self._ml_signal_values(design, eligible)
        signals[_ML_SIGNAL] = ml_score
        masks[_ML_SIGNAL] = ml_mask
        coverage.loc[_ML_SIGNAL] = float(ml_mask.mean())

        # Save formation-time inputs only after producing today's prediction.
        # They cannot enter training until a later allocate call observes the
        # corresponding forward return interval.
        self._previous_ml_design = design.copy()
        self._previous_ml_eligible = eligible.copy()
        self._previous_ml_date = current_date
        return signals.reindex(columns=self.signal_names), coverage, masks

    def _short_overlay_weights(
        self, returns: pd.DataFrame, long_weights: pd.Series
    ) -> pd.Series | None:
        """Select a small, capped set of the model's most disliked names and
        assign them negative weight, entirely separate from the long book.

        Runs only after `self._run_long_engine()` has returned, and only reads
        state that call already saved (`_previous_signal_panel`,
        `_previous_signal_date`, `last_signal_weights`, `last_selected_assets`
        -- see the module docstring for why each is safe to reuse here). It
        never touches the long-only HRP prior, factor risk model, or
        multi-period optimizer.
        """

        if (
            self._previous_signal_panel is None
            or self._previous_signal_date is None
            or self.last_signal_weights is None
        ):
            return None

        broad_signals = self._previous_signal_panel
        as_of = self._previous_signal_date
        universe = broad_signals.index
        # The composite score mirrors the parent's own admission_score
        # formula exactly (`_rank_normal(broad_signals @ signal_weights)`),
        # just evaluated over the full universe instead of only the assets
        # it went on to admit.
        composite = _rank_normal(broad_signals @ self.last_signal_weights)

        already_long = (
            self.last_selected_assets
            if self.last_selected_assets is not None
            else pd.Index([])
        )
        forced_exit = self._forced_exit_assets
        snapshot = self.features.snapshot(as_of, universe)
        sectors = self.sectors.lookup(as_of, universe)
        price = _field(snapshot, "price", universe)
        adv = _field(snapshot, "dollar_volume_20d", universe)
        market_caps = _field(snapshot, "market_cap_usd", universe, 0.0)
        has_signal = broad_signals.abs().sum(axis=1) > _EPS

        eligible = (
            composite.le(self.config.short_minimum_candidate_score)
            & has_signal
            & ~universe.isin(already_long)
            & ~universe.isin(forced_exit)
            & price.ge(self.config.minimum_price).to_numpy()
            & adv.ge(self.config.minimum_dollar_volume).to_numpy()
            & market_caps.ge(self.config.short_minimum_market_cap).to_numpy()
        )
        candidates = composite.loc[universe[eligible]].sort_values()
        if candidates.empty:
            return None

        chosen: list[str] = []
        sector_counts: dict[str, int] = {}
        for asset in candidates.index:
            sector = str(sectors.get(asset, "UNKNOWN"))
            if sector_counts.get(sector, 0) >= self.config.short_max_added_per_sector:
                continue
            chosen.append(str(asset))
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if len(chosen) >= self.config.short_maximum_added_assets:
                break
        if not chosen:
            return None

        conviction = (-candidates.loc[chosen]).clip(lower=_EPS)
        # `_capped_normalize` caps and sums to 1.0 over a long-only budget;
        # rescale its per-name cap into a fraction of *this* budget, then
        # scale the whole thing down to the short sleeve's own gross target
        # and flip the sign.
        relative_cap = self.config.short_max_weight_per_name / max(
            self.config.short_gross_budget, _EPS
        )
        normalized = _capped_normalize(conviction, relative_cap)
        short_weights = -self.config.short_gross_budget * normalized

        # Simple, self-contained cost estimate: a realized-volatility proxy
        # from the trailing return window in place of the long book's
        # (unavailable-here) predictive specific-volatility model, run
        # through the same spread/temporary/permanent impact cost functions
        # the long engine itself uses. Since this sleeve carries no state
        # across rebalances, the "trade" is the full target weight itself --
        # a build-from-scratch cost, not a true incremental turnover cost.
        trailing = returns.reindex(columns=short_weights.index).iloc[
            -min(52, len(returns)) :
        ]
        specific_proxy = (
            trailing.std(ddof=0).reindex(short_weights.index).fillna(
                trailing.std(ddof=0).median()
            )
        ).clip(lower=1e-4).to_numpy(dtype=float)
        short_snapshot = snapshot.reindex(short_weights.index)
        spread = _spread_vector(short_snapshot, short_weights.index, self.config)
        short_adv = adv.reindex(short_weights.index).fillna(
            self.config.minimum_dollar_volume
        ).to_numpy(dtype=float)
        spread_cost, temporary, permanent = _impact_cost_arrays(
            short_weights.to_numpy(dtype=float),
            specific_proxy,
            short_adv,
            spread,
            self.config,
        )
        self.last_short_overlay_weights = short_weights.copy()
        self.last_short_overlay_cost = spread_cost + temporary + permanent
        return short_weights

    def _run_long_engine(self, returns: pd.DataFrame) -> pd.Series:
        """Duplicate-and-modify of the parent `RetailAlphaMPCAllocator.allocate`
        method body. A duplicate, not an override, because the parent calls
        `_predictive_factor_risk_model(...)` and `hrp_allocate(...)` as bare
        module-level functions rather than `self.` methods, so a subclass
        cannot intercept them by overriding a method the parent would call --
        the only way to substitute the enhanced risk model and RA-HRP+VSK
        construction while otherwise keeping the parent's admission,
        candidate-selection, participation-limit, multi-period optimizer, and
        diagnostics logic untouched is to copy that logic here and swap in
        the specific pieces this subclass changes. Everything not called out
        below is preserved verbatim from the parent; the changes are:

          1. `_predictive_factor_risk_model` -> `_ml_predictive_factor_risk_model`
             (the EWMA multi-scale + Ledoit-Wolf shrunk factor covariance,
             DCC-proxy residual correlation, HAR-style specific-variance risk
             model).
          2. `hrp_allocate` -> `_ra_hrp_vsk_allocate` (Return-Adjusted HRP
             bisection + VSK higher-moment tilt in place of plain HRP).
          3. `apply_bounds(hrp, 0.0, self.config.max_weight)` kept as an
             extra safety net exactly as the parent has it.
          4. A max-total-assets trim applied to the ranked `additions` list
             (lowest-conviction additions dropped first), implementing
             `min(min_weights) = 1 / max_total_assets` together with (9).
          5/6. The optimizer's `objective`/`objective_gradient` closures gain
             a realized-vol regime multiplier on the temporary-impact term
             and a tax-aware cost term (with its hand-derived gradient) on
             position reductions.
          7/8. The parametric CVaR constraint and the spread/temporary/
             permanent transaction-cost model are otherwise untouched.
          9. A post-solve min-weight-floor prune-and-reproject is applied to
             the final weights before diagnostics are computed.
        """

        window = returns.copy().astype(float).sort_index()
        window = window.replace([np.inf, -np.inf], np.nan)
        valid = (window.notna().sum() >= self.config.minimum_history_weeks) & (
            window.std(skipna=True) > _EPS
        )
        observed_window = window.loc[:, valid].copy()
        window = observed_window.fillna(observed_window.mean())
        if window.shape[1] < 2:
            raise ValueError("Retail Alpha MPC has too few eligible assets.")
        if window.shape[1] * self.config.max_weight < 1.0 - 1e-9:
            raise ValueError("max_weight is infeasible for the eligible asset count.")

        as_of = pd.Timestamp(window.index[-1])
        snapshot = self.features.snapshot(as_of, window.columns)
        sectors = self.sectors.lookup(as_of, window.columns)
        exposures, _ = _current_exposures(window, snapshot, sectors)
        market_caps = _field(
            snapshot, "market_cap_usd", window.columns, 1.0
        ).fillna(1.0)
        broad_signals, signal_coverage, signal_masks = self._build_signal_panel(
            window, snapshot, exposures, market_caps, observed_window
        )
        self._update_dynamic_ics(window, as_of)
        signal_availability = (
            broad_signals.std(ddof=0).gt(1e-8)
            & signal_coverage.ge(self.config.minimum_signal_coverage)
        ).astype(float)
        signal_ics, signal_weights = self._signal_combination(
            signal_availability, broad_signals.where(signal_masks)
        )
        admission_score = _rank_normal(broad_signals @ signal_weights)

        core = _latest_pit_assets(self.balanced_pit, as_of, window.columns)
        if len(core) < 2:
            raise ValueError("Balanced PIT core contains fewer than two assets.")
        live_previous = (
            self._engine_current_weights
            if self._engine_current_weights is not None
            else self._previous_weights
        )
        has_previous = live_previous is not None and float(live_previous.sum()) > _EPS
        previously_held = (
            pd.Index([])
            if not has_previous
            else pd.Index(
                pd.Series(live_previous, dtype=float)
                .loc[lambda values: values > _EPS]
                .index
            ).intersection(window.columns)
        )
        price = _field(snapshot, "price", window.columns)
        adv_broad = _field(snapshot, "dollar_volume_20d", window.columns)
        # Availability alone is not alpha. If cleaning/noise filtering assigns
        # every signal zero weight, do not admit arbitrary assets whose tied
        # zero score would otherwise pass a permissive candidate threshold.
        has_live_signal = bool(signal_weights.sum() > _EPS)
        asset_has_live_signal = (
            signal_masks.astype(float) @ signal_weights
        ).gt(_EPS)
        hold_score = (
            self.config.minimum_candidate_score
            if self.config.minimum_candidate_hold_score is None
            else self.config.minimum_candidate_hold_score
        )
        required_score = pd.Series(
            self.config.minimum_candidate_score,
            index=window.columns,
            dtype=float,
        )
        required_score.loc[
            previously_held.difference(core).intersection(window.columns)
        ] = hold_score
        candidate_mask = (
            ~window.columns.isin(core)
            & ~window.columns.isin(self._forced_exit_assets)
            & has_live_signal
            & asset_has_live_signal.reindex(window.columns).fillna(False).to_numpy()
            & admission_score.ge(required_score).to_numpy()
            & price.ge(self.config.minimum_price).to_numpy()
            & adv_broad.ge(self.config.minimum_dollar_volume).to_numpy()
        )
        ranked = admission_score.loc[window.columns[candidate_mask]].sort_values(
            ascending=False
        )
        additions: list[str] = []
        sector_counts: dict[str, int] = {}
        for asset in ranked.index:
            sector = str(sectors.get(asset, "UNKNOWN"))
            if sector_counts.get(sector, 0) >= self.config.max_added_per_sector:
                continue
            additions.append(str(asset))
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if len(additions) >= self.config.maximum_added_assets:
                break

        # --- Modification (4): max-total-assets trim. `core` and
        # `previously_held` are never trimmed here (their fate is decided by
        # the admission/exit-schedule logic above and below); only the
        # lowest-conviction tail of `additions` is dropped, and only if the
        # forced names alone don't already exceed the cap -- an over-budget
        # book from `core`+`previously_held` alone is tolerated rather than
        # raised, since neither the balanced-core universe nor an existing
        # live holding should be silently dropped by a diversification cap.
        forced_names = core.union(previously_held)
        available_for_additions = max(
            int(self.config.ml_max_total_assets) - len(forced_names), 0
        )
        if len(additions) > available_for_additions:
            additions = additions[:available_for_additions]

        continuing = core.append(pd.Index(additions)).drop_duplicates()
        held_assets = previously_held
        exiting = held_assets.difference(continuing)
        selected = continuing.append(exiting).drop_duplicates()
        selected_returns = window.loc[:, selected]
        selected_snapshot = snapshot.reindex(selected)
        selected_sectors = sectors.reindex(selected).fillna("UNKNOWN")
        selected_exposures, _ = _current_exposures(
            selected_returns, selected_snapshot, selected_sectors
        )
        selected_caps = market_caps.reindex(selected).fillna(1.0)
        selected_signals = broad_signals.reindex(selected).fillna(0.0)
        # --- Modification (1): enhanced predictive risk model.
        covariance, specific_volatility, risk_r2 = _ml_predictive_factor_risk_model(
            selected_returns,
            selected_exposures,
            selected_sectors,
            selected_snapshot,
            selected_caps,
            self.config,
        )
        alpha = pd.Series(0.0, index=selected)
        for signal in self.signal_names:
            alpha += (
                signal_weights[signal]
                * signal_ics[signal]
                * specific_volatility
                * selected_signals[signal]
            )
        alpha.loc[exiting] = 0.0
        # Repaired held assets are present to model liquidation and costs, not
        # to receive a new alpha-backed allocation from synthetic history.
        alpha.loc[self._forced_exit_assets.intersection(alpha.index)] = 0.0

        # --- Growth-optimal scaling of the alpha vector ---
        # The optimizer's first-order condition is
        #   w = (alpha_strength / risk_aversion) * Sigma^-1 alpha,
        # so that ratio is the Kelly fraction, currently the accidental
        # product of two independently hand-set knobs. Rescaling alpha by
        # `c* * risk_aversion / alpha_strength` makes the effective ratio
        # exactly the derived c*, i.e. fractional Kelly by construction.
        # Monte Carlo says the unscaled plug-in rule has negative expected
        # log-growth, so this is the half of the design with a real claim
        # behind it.
        if self.config.ml_kelly_scale_enabled:
            _, kelly_fraction = self._kelly_factor_allocation()
            strength = float(self.config.alpha_strength)
            if kelly_fraction > 0.0 and strength > _EPS:
                alpha = alpha * (
                    kelly_fraction * float(self.config.risk_aversion) / strength
                )
            else:
                # No usable evidence yet: the growth-optimal bet on the
                # signal is zero, so the book falls back to the risk-model
                # and HRP-prior terms rather than trading an unsized alpha.
                alpha = alpha * 0.0

        correlation = covariance_to_correlation(covariance)
        distance = np.sqrt(np.maximum(0.5 * (1.0 - correlation.to_numpy()), 0.0))
        np.fill_diagonal(distance, 0.0)
        tree = linkage(squareform(distance, checks=False), method="average")
        # --- Modification (2): RA-HRP + VSK tilt in place of plain HRP.
        hrp = _ra_hrp_vsk_allocate(
            correlation,
            covariance,
            tree,
            list(selected),
            alpha,
            selected_returns,
            self.config,
            risk_cap=0.15,
        )
        hrp = apply_bounds(hrp, 0.0, self.config.max_weight)
        if len(exiting):
            hrp.loc[exiting] = 0.0
            if float(hrp.sum()) <= _EPS:
                raise RuntimeError("No continuing assets remain after forced exits.")
            hrp /= hrp.sum()
        previous_actual = (
            live_previous.reindex(selected).fillna(0.0)
            if has_previous
            else hrp.copy()
        )
        if has_previous:
            omitted_weight = float(
                live_previous.drop(index=selected, errors="ignore")
                .clip(lower=0.0)
                .sum()
            )
            if omitted_weight > self.config.maximum_unrepresentable_weight:
                raise RuntimeError(
                    "Live holdings could not be represented in the MPC window; "
                    f"omitted weight={omitted_weight:.6f}."
                )
        adv = _field(
            selected_snapshot,
            "dollar_volume_20d",
            selected,
            self.config.minimum_dollar_volume,
        ).fillna(self.config.minimum_dollar_volume)
        capacity = (
            self.config.maximum_adv_participation
            * self.config.execution_days
            * adv
            / self.config.portfolio_value
        ).clip(lower=0.0)
        first_lower = (previous_actual - capacity).clip(lower=0.0)
        first_upper = np.minimum(
            self.config.max_weight, previous_actual + capacity
        ).clip(lower=0.0)
        if not has_previous:
            first_lower[:] = 0.0
            first_upper[:] = self.config.max_weight
        if float(first_upper.sum()) < 1.0 - 1e-9:
            raise ValueError(
                "Participation limits cannot fund a fully invested first step."
            )

        covariance_values = covariance.to_numpy(dtype=float)
        alpha_values = alpha.to_numpy(dtype=float)
        prior = hrp.reindex(selected).to_numpy(dtype=float)
        previous_values = previous_actual.to_numpy(dtype=float)
        specific_values = specific_volatility.to_numpy(dtype=float)
        adv_values = adv.to_numpy(dtype=float)
        spread_values = _spread_vector(selected_snapshot, selected, self.config)
        horizon = self.config.planning_horizon
        asset_count = len(selected)
        exposure_columns = [
            column
            for column in ("SIZE", "VALUE", "MOMENTUM", "QUALITY", "LOW_VOL")
            if column in selected_exposures
        ]
        style_matrix = selected_exposures.loc[:, exposure_columns].to_numpy(dtype=float)
        sector_groups = list(
            selected_sectors.groupby(selected_sectors, observed=True).groups.items()
        )
        sector_members = [
            selected.get_indexer(pd.Index(members)) for _, members in sector_groups
        ]
        sector_labels = [str(label) for label, _ in sector_groups]

        # --- Modifications (5)/(6) setup: the vol-scale multiplier and the
        # tax-aware cost's per-name inputs. Tax cost is zero whenever tax-lot
        # info hasn't been supplied via `set_tax_lot_info`, or the feature is
        # switched off in config -- `objective`/`objective_gradient` below
        # add nothing beyond the parent's own cost terms in that case.
        vol_scale_multiplier = _vol_scale_multiplier(selected_returns, self.config)
        self.last_vol_scale_multiplier = vol_scale_multiplier
        tax_enabled = bool(
            self.config.ml_tax_aware_costs_enabled
            and self._tax_rate_per_name is not None
            and self._unrealized_gain_frac is not None
        )
        if tax_enabled:
            tax_rate_values = (
                self._tax_rate_per_name.reindex(selected)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            gain_frac_values = (
                self._unrealized_gain_frac.reindex(selected)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
        else:
            tax_rate_values = np.zeros(asset_count, dtype=float)
            gain_frac_values = np.zeros(asset_count, dtype=float)

        def unpack(flat: np.ndarray) -> np.ndarray:
            return flat.reshape(horizon, asset_count)

        discounts = self.config.alpha_decay ** (
            np.arange(horizon, dtype=float) * self.config.planning_step_weeks
        )
        participation_scale = self.config.portfolio_value / (
            np.maximum(adv_values, 1.0) * self.config.execution_days
        )
        permanent_slope = (
            self.config.permanent_impact_coefficient
            * specific_values
            * participation_scale
        )
        lower_matrix = np.zeros((horizon, asset_count), dtype=float)
        upper_matrix = np.full(
            (horizon, asset_count), self.config.max_weight, dtype=float
        )
        lower_matrix[0] = first_lower.to_numpy(dtype=float)
        upper_matrix[0] = first_upper.to_numpy(dtype=float)
        exit_positions = selected.get_indexer(exiting)
        exit_positions = exit_positions[exit_positions >= 0]
        for position in exit_positions:
            schedule = np.maximum(
                previous_values[position]
                - capacity.to_numpy(dtype=float)[position]
                * np.arange(1, horizon + 1, dtype=float),
                0.0,
            )
            lower_matrix[:, position] = schedule
            upper_matrix[:, position] = schedule

        initial_rows: list[np.ndarray] = []
        for step in range(horizon):
            initial_rows.append(
                _project_box_simplex(
                    pd.Series(prior, index=selected),
                    pd.Series(lower_matrix[step], index=selected),
                    pd.Series(upper_matrix[step], index=selected),
                ).to_numpy(dtype=float)
            )
        initial_path = np.vstack(initial_rows)

        def objective(flat: np.ndarray) -> float:
            path = unpack(flat)
            total = 0.0
            predecessor = previous_values
            permanent_state = np.zeros(asset_count, dtype=float)
            for step, weights in enumerate(path):
                alpha_discount = discounts[step]
                trade = weights - predecessor
                spread_cost, temporary, _ = _impact_cost_arrays(
                    trade,
                    specific_values,
                    adv_values,
                    spread_values,
                    self.config,
                )
                # --- Modification (5): vol-scaled temporary impact + tax cost.
                temporary *= vol_scale_multiplier
                tax_cost = (
                    _tax_aware_cost(trade, predecessor, tax_rate_values, gain_frac_values)
                    if tax_enabled
                    else 0.0
                )
                signed_participation = trade * self.config.portfolio_value / (
                    np.maximum(adv_values, 1.0) * self.config.execution_days
                )
                impact_increment = (
                    self.config.permanent_impact_coefficient
                    * specific_values
                    * signed_participation
                )
                permanent = float(
                    np.sum(trade * (permanent_state + 0.5 * impact_increment))
                )
                permanent_state += impact_increment
                total += (
                    0.5
                    * self.config.risk_aversion
                    * float(weights @ covariance_values @ weights)
                    - alpha_discount
                    * self.config.alpha_strength
                    * float(alpha_values @ weights)
                    + spread_cost
                    + temporary
                    + permanent
                    + tax_cost
                )
                predecessor = weights
            total += self.config.hrp_anchor_strength * float(
                np.sum((path[-1] - prior) ** 2)
            )
            return float(total)

        def objective_gradient(flat: np.ndarray) -> np.ndarray:
            """Exact gradient avoids O(horizon * assets) finite differences."""

            path = unpack(flat)
            gradient = np.zeros_like(path)
            predecessors = np.vstack([previous_values, path[:-1]])
            trades = path - predecessors
            trade_gradient = np.zeros_like(trades)

            for step, weights in enumerate(path):
                gradient[step] += (
                    self.config.risk_aversion * covariance_values @ weights
                    - discounts[step]
                    * self.config.alpha_strength
                    * alpha_values
                )
                trade = trades[step]
                absolute = np.abs(trade)
                sign = np.sign(trade)
                participation = participation_scale * absolute
                temporary_gradient = (
                    self.config.temporary_impact_coefficient
                    * specific_values
                    * sign
                    * (
                        np.power(
                            participation + _EPS,
                            self.config.temporary_impact_exponent,
                        )
                        + absolute
                        * self.config.temporary_impact_exponent
                        * np.power(
                            participation + _EPS,
                            self.config.temporary_impact_exponent - 1.0,
                        )
                        * participation_scale
                    )
                )
                # --- Modification (6): matching gradient for (5). The vol
                # multiplier scales `temporary_gradient` exactly as it scales
                # `temporary` above. The tax term is piecewise-linear in
                # `trade`: strictly interior to a partial sale it contributes
                # a constant `-tax_rate * gain_frac`; at trade==0 or at a
                # full liquidation (the boundary where `_tax_aware_cost`'s own
                # clip saturates) the subgradient is left at 0, matching the
                # `_apply_min_weight_floor` policy of treating those points as
                # cleanup rather than solver-differentiable interior states.
                temporary_gradient = temporary_gradient * vol_scale_multiplier
                if tax_enabled:
                    predecessor_step = predecessors[step]
                    interior = (trade < 0) & (trade > -predecessor_step)
                    tax_gradient = np.where(
                        interior, -tax_rate_values * gain_frac_values, 0.0
                    )
                else:
                    tax_gradient = 0.0
                trade_gradient[step] += (
                    spread_values * sign + temporary_gradient + tax_gradient
                )

            cumulative_trade = np.cumsum(trades, axis=0) - trades
            weighted_future_trade = np.zeros(asset_count, dtype=float)
            for step in range(horizon - 1, -1, -1):
                trade_gradient[step] += (
                    permanent_slope
                    * (cumulative_trade[step] + trades[step])
                    + permanent_slope * weighted_future_trade
                )
                weighted_future_trade += trades[step]

            for step in range(horizon):
                gradient[step] += trade_gradient[step]
                if step > 0:
                    gradient[step - 1] -= trade_gradient[step]
            gradient[-1] += (
                2.0 * self.config.hrp_anchor_strength * (path[-1] - prior)
            )
            return gradient.ravel()

        sum_jacobian = np.zeros((horizon, horizon * asset_count), dtype=float)
        for step in range(horizon):
            start = step * asset_count
            sum_jacobian[step, start : start + asset_count] = 1.0

        def cvar_constraint(flat: np.ndarray) -> np.ndarray:
            path = unpack(flat)
            variance = np.einsum("ij,jk,ik->i", path, covariance_values, path)
            return self.config.weekly_cvar_95_limit - 2.0627 * np.sqrt(
                np.maximum(variance, 0.0)
            )

        def cvar_jacobian(flat: np.ndarray) -> np.ndarray:
            path = unpack(flat)
            result = np.zeros((horizon, horizon * asset_count), dtype=float)
            for step, weights in enumerate(path):
                volatility = np.sqrt(
                    max(float(weights @ covariance_values @ weights), 0.0)
                )
                if volatility <= _EPS:
                    continue
                start = step * asset_count
                result[step, start : start + asset_count] = (
                    -2.0627 * covariance_values @ weights / volatility
                )
            return result

        constraints: list[dict[str, object]] = [
            {
                "type": "eq",
                "fun": lambda flat: unpack(flat).sum(axis=1) - 1.0,
                "jac": lambda _flat: sum_jacobian,
            },
            {
                "type": "ineq",
                "fun": cvar_constraint,
                "jac": cvar_jacobian,
            },
        ]
        capacity_steps = list(range(0 if has_previous else 1, horizon))
        if capacity_steps:
            capacity_values = capacity.to_numpy(dtype=float)
            capacity_jacobian = np.zeros(
                (2 * len(capacity_steps) * asset_count, horizon * asset_count),
                dtype=float,
            )
            row_start = 0
            for step in capacity_steps:
                trade_jacobian = np.zeros(
                    (asset_count, horizon * asset_count), dtype=float
                )
                column = step * asset_count
                trade_jacobian[:, column : column + asset_count] = np.eye(asset_count)
                if step > 0:
                    previous_column = (step - 1) * asset_count
                    trade_jacobian[
                        :, previous_column : previous_column + asset_count
                    ] = -np.eye(asset_count)
                capacity_jacobian[
                    row_start : row_start + asset_count
                ] = -trade_jacobian
                capacity_jacobian[
                    row_start + asset_count : row_start + 2 * asset_count
                ] = trade_jacobian
                row_start += 2 * asset_count

            def capacity_constraint(flat: np.ndarray) -> np.ndarray:
                path = unpack(flat)
                values: list[np.ndarray] = []
                for step in capacity_steps:
                    predecessor = previous_values if step == 0 else path[step - 1]
                    trade = path[step] - predecessor
                    values.extend([capacity_values - trade, capacity_values + trade])
                return np.concatenate(values)

            constraints.append(
                {
                    "type": "ineq",
                    "fun": capacity_constraint,
                    "jac": lambda _flat: capacity_jacobian,
                }
            )
        for column in range(style_matrix.shape[1]):
            vector = style_matrix[:, column].copy()
            style_jacobian = np.zeros(
                (horizon, horizon * asset_count), dtype=float
            )
            for step in range(horizon):
                start = step * asset_count
                style_jacobian[step, start : start + asset_count] = vector
            constraints.extend(
                [
                    {
                        "type": "ineq",
                        "fun": lambda flat, v=vector: (
                            self.config.max_absolute_style_exposure
                            - unpack(flat) @ v
                        ),
                        "jac": lambda _flat, j=style_jacobian: -j,
                    },
                    {
                        "type": "ineq",
                        "fun": lambda flat, v=vector: (
                            self.config.max_absolute_style_exposure
                            + unpack(flat) @ v
                        ),
                        "jac": lambda _flat, j=style_jacobian: j,
                    },
                ]
            )
        for positions in sector_members:
            sector_jacobian = np.zeros(
                (horizon, horizon * asset_count), dtype=float
            )
            for step in range(horizon):
                start = step * asset_count
                sector_jacobian[step, start + positions] = -1.0
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda flat, p=positions: (
                        self.config.max_sector_weight
                        - unpack(flat)[:, p].sum(axis=1)
                    ),
                    "jac": lambda _flat, j=sector_jacobian: j,
                }
            )

        flat_bounds = list(zip(lower_matrix.ravel(), upper_matrix.ravel(), strict=True))

        if self.config.ml_equal_weight_benchmark:
            # 1/N over the book THIS MODEL SELECTED, as the sizing control.
            #
            # The `equal_weight` model already registered in compare_models.py
            # is 1/N over the whole eligible universe, so it differs from this
            # allocator on selection *and* sizing and cannot separate them.
            # This benchmark reuses the entire selection stack -- balanced core,
            # admission scoring, sector diversification, the max-assets trim,
            # participation limits and the exit schedule -- and replaces only
            # the optimizer, so the two differences decompose cleanly:
            #
            #   selection value = EW-on-selected  minus  EW-on-universe
            #   sizing value    = full model      minus  EW-on-selected
            #
            # DeMiguel-Garlappi-Uppal is why this has to exist: 1/N is hard to
            # beat, and a model that cannot clear its own selected-book 1/N is
            # not being helped by its optimizer. It also runs roughly an order
            # of magnitude faster, since SLSQP is ~68% of a walk-forward.
            #
            # Exits keep their scheduled decay: they sit in `selected` to model
            # liquidation cost, not to receive a fresh equal allocation.
            equal = pd.Series(0.0, index=selected, dtype=float)
            live = continuing.intersection(selected).difference(exiting)
            if len(live):
                equal.loc[live] = 1.0 / float(len(live))
            equal = _capped_normalize(equal, self.config.max_weight)
            candidate = np.tile(equal.to_numpy(dtype=float), horizon)
            optimizer_repaired = False
            risk_limit_relaxed = False
            effective_cvar_limit = float(self.config.weekly_cvar_95_limit)
        else:
            result = minimize(
                objective,
                initial_path.ravel(),
                method="SLSQP",
                jac=objective_gradient,
                bounds=flat_bounds,
                constraints=constraints,
                options={
                    "maxiter": self.config.optimizer_max_iterations,
                    "ftol": 1e-9,
                    "disp": False,
                },
            )

            lower_flat = lower_matrix.ravel()
            upper_flat = upper_matrix.ravel()
            risk_limit_relaxed = False
            effective_cvar_limit = float(self.config.weekly_cvar_95_limit)

            def feasibility_violation(flat: np.ndarray) -> float:
                if flat.shape != lower_flat.shape or not np.isfinite(flat).all():
                    return float("inf")
                violation = max(
                    float(np.max(lower_flat - flat)),
                    float(np.max(flat - upper_flat)),
                    0.0,
                )
                for constraint in constraints:
                    values = np.asarray(constraint["fun"](flat), dtype=float)
                    if not np.isfinite(values).all():
                        return float("inf")
                    if constraint["type"] == "eq":
                        violation = max(violation, float(np.max(np.abs(values))))
                    else:
                        violation = max(violation, float(np.max(-values)), 0.0)
                return violation

            candidate = (
                np.asarray(result.x, dtype=float)
                if np.asarray(result.x).shape == lower_flat.shape
                else initial_path.ravel()
            )
            main_violation = feasibility_violation(candidate)
            optimizer_repaired = main_violation > 1e-7
            if optimizer_repaired:
                # First solve the affine feasibility problem exactly. SLSQP is
                # unreliable when started far outside simultaneous style, sector,
                # participation, and exit constraints; HiGHS gives it a valid
                # linear starting point. CVaR remains a convex nonlinear constraint
                # and is handled by the second-stage repair below.
                zero = np.zeros_like(lower_flat)
                a_ub: list[np.ndarray] = []
                b_ub: list[np.ndarray] = []
                a_eq: list[np.ndarray] = []
                b_eq: list[np.ndarray] = []
                for position, constraint in enumerate(constraints):
                    if position == 1:  # nonlinear CVaR constraint
                        continue
                    values_at_zero = np.atleast_1d(
                        np.asarray(constraint["fun"](zero), dtype=float)
                    )
                    jacobian = np.atleast_2d(
                        np.asarray(constraint["jac"](zero), dtype=float)
                    )
                    if constraint["type"] == "eq":
                        a_eq.append(jacobian)
                        b_eq.append(-values_at_zero)
                    else:
                        a_ub.append(-jacobian)
                        b_ub.append(values_at_zero)
                linear_feasible = linprog(
                    np.zeros_like(lower_flat),
                    A_ub=np.vstack(a_ub) if a_ub else None,
                    b_ub=np.concatenate(b_ub) if b_ub else None,
                    A_eq=np.vstack(a_eq) if a_eq else None,
                    b_eq=np.concatenate(b_eq) if b_eq else None,
                    bounds=flat_bounds,
                    method="highs",
                )
                if not linear_feasible.success:
                    asset_cap_bound = asset_count * self.config.max_weight
                    sector_cap_bound = (
                        len(sector_members) * self.config.max_sector_weight
                        if sector_members
                        else float("inf")
                    )
                    diagnostics: list[str] = []
                    if asset_cap_bound < 1.0 - 1e-9:
                        diagnostics.append(
                            f"max_weight={self.config.max_weight:.4f} across "
                            f"{asset_count} selected assets caps total investable "
                            f"weight at {asset_cap_bound:.4f} (< 1.0)"
                        )
                    if sector_cap_bound < 1.0 - 1e-9:
                        diagnostics.append(
                            f"max_sector_weight={self.config.max_sector_weight:.4f} "
                            f"across {len(sector_members)} represented sectors "
                            f"({', '.join(sector_labels)}) caps total investable "
                            f"weight at {sector_cap_bound:.4f} (< 1.0)"
                        )
                    if not diagnostics:
                        diagnostics.append(
                            "no single cap (max_weight x count, max_sector_weight x "
                            "sector count) alone explains this; likely a combination "
                            "of sector, style-exposure, and/or participation caps "
                            "jointly excludes any fully-invested portfolio"
                        )
                    raise RuntimeError(
                        "Retail Alpha MPC constraints are linearly infeasible: "
                        + "; ".join(diagnostics)
                        + f". Raw solver message={linear_feasible.message}"
                    )
                repair_start = np.asarray(linear_feasible.x, dtype=float)

                # Among linearly feasible portfolios, the global minimum-variance
                # path is the strongest possible CVaR repair. Passing the affine
                # constraints as two dense matrices avoids SLSQP repeatedly
                # evaluating dozens of Python constraint callables.
                def repair_objective(flat: np.ndarray) -> float:
                    path = unpack(flat)
                    return 0.5 * float(
                        np.einsum("ij,jk,ik->", path, covariance_values, path)
                    )

                def repair_gradient(flat: np.ndarray) -> np.ndarray:
                    return (unpack(flat) @ covariance_values).ravel()

                matrix_constraints: list[LinearConstraint] = []
                if a_eq:
                    equality_matrix = np.vstack(a_eq)
                    equality_target = np.concatenate(b_eq)
                    matrix_constraints.append(
                        LinearConstraint(
                            equality_matrix, equality_target, equality_target
                        )
                    )
                if a_ub:
                    inequality_matrix = np.vstack(a_ub)
                    inequality_upper = np.concatenate(b_ub)
                    matrix_constraints.append(
                        LinearConstraint(
                            inequality_matrix,
                            np.full_like(inequality_upper, -np.inf),
                            inequality_upper,
                        )
                    )

                repaired = minimize(
                    repair_objective,
                    repair_start,
                    method="SLSQP",
                    jac=repair_gradient,
                    bounds=flat_bounds,
                    constraints=matrix_constraints,
                    options={
                        "maxiter": max(self.config.optimizer_max_iterations, 100),
                        "ftol": 1e-10,
                        "disp": False,
                    },
                )
                repaired_values = np.asarray(repaired.x, dtype=float)
                repaired_violation = feasibility_violation(repaired_values)
                if repaired_violation > 1e-7:
                    linear_violation = max(
                        float(np.max(lower_flat - repaired_values)),
                        float(np.max(repaired_values - upper_flat)),
                        0.0,
                    )
                    for position, constraint in enumerate(constraints):
                        if position == 1:
                            continue
                        values = np.asarray(
                            constraint["fun"](repaired_values), dtype=float
                        )
                        if constraint["type"] == "eq":
                            linear_violation = max(
                                linear_violation, float(np.max(np.abs(values)))
                            )
                        else:
                            linear_violation = max(
                                linear_violation, float(np.max(-values)), 0.0
                            )
                    if (
                        self.config.allow_cvar_floor_relaxation
                        and linear_violation <= 1e-7
                    ):
                        repaired_path = unpack(repaired_values)
                        repaired_variance = np.einsum(
                            "ij,jk,ik->i",
                            repaired_path,
                            covariance_values,
                            repaired_path,
                        )
                        effective_cvar_limit = max(
                            effective_cvar_limit,
                            2.0627
                            * float(np.sqrt(np.maximum(repaired_variance, 0.0)).max()),
                        )
                        risk_limit_relaxed = True
                    else:
                        raise RuntimeError(
                            "Retail Alpha MPC could not produce a constraint-feasible "
                            "portfolio after repair; "
                            f"maximum violation={repaired_violation:.3e}; "
                            f"solver message={repaired.message}"
                        )
                candidate = repaired_values

        optimizer_success = True
        planned = unpack(candidate)
        final = _project_box_simplex(
            pd.Series(planned[0], index=selected),
            pd.Series(lower_matrix[0], index=selected),
            pd.Series(upper_matrix[0], index=selected),
        )
        planned[0] = final.to_numpy(dtype=float)

        # --- Modification (9): post-solve min-weight-floor prune-and-reproject.
        # `min(min_weights) = 1 / ml_max_total_assets` is enforced here, not as
        # a solver bound -- see `_apply_min_weight_floor`'s own docstring for
        # why. Forced-exit names and the balanced-core universe are protected
        # from pruning; the diagnostics and state below are computed from the
        # post-floor book, which is the actual portfolio being returned.
        upper_bound_series = pd.Series(upper_matrix[0], index=selected)
        pre_floor = final
        final = _apply_min_weight_floor(
            pre_floor,
            1.0 / max(int(self.config.ml_max_total_assets), 1),
            upper_bound_series,
            protected=self._forced_exit_assets.union(core),
        )
        pruned_mask = (pre_floor > _EPS) & (final <= _EPS)
        self.last_pruned_dust_assets = pre_floor.index[pruned_mask.to_numpy(dtype=bool)]

        change = final - previous_actual
        spread_cost, temporary, permanent = _impact_cost_arrays(
            change.to_numpy(dtype=float),
            specific_values,
            adv_values,
            spread_values,
            self.config,
        )
        values = final.to_numpy(dtype=float)
        volatility = float(np.sqrt(max(values @ covariance_values @ values, 0.0)))
        sector_weight = final.groupby(selected_sectors).sum()
        style_exposure = selected_exposures.loc[:, exposure_columns].T @ final
        utilization = (
            change.abs()
            * self.config.portfolio_value
            / (
                self.config.maximum_adv_participation
                * self.config.execution_days
                * adv.replace(0.0, np.nan)
            )
        )

        self._previous_weights = final.copy()
        self._previous_signal_panel = broad_signals.copy()
        self._previous_signal_availability = signal_masks.copy()
        self._previous_signal_date = as_of
        self.last_covariance = covariance
        self.last_alpha = alpha
        self.last_selected_assets = selected
        self.last_specific_volatility = specific_volatility
        self.last_signal_weights = signal_weights
        self.last_signal_ics = signal_ics
        self.last_signal_coverage = signal_coverage
        self.last_planned_weights = pd.DataFrame(
            planned,
            index=pd.Index(range(1, horizon + 1), name="planning_step"),
            columns=selected,
        )
        self._last_snapshot = selected_snapshot.copy()
        self.last_execution_cost = spread_cost + temporary + permanent
        self.last_tax_cost = (
            _tax_aware_cost(
                change.to_numpy(dtype=float),
                previous_values,
                tax_rate_values,
                gain_frac_values,
            )
            if tax_enabled
            else 0.0
        )
        self.last_diagnostics = RetailAlphaMPCDiagnostics(
            core_asset_count=len(core),
            added_asset_count=len(additions),
            selected_asset_count=len(selected),
            factor_count=selected_exposures.shape[1],
            industry_count=int(selected_sectors.nunique()),
            planning_horizon=horizon,
            predictive_specific_volatility_median=float(specific_volatility.median()),
            specific_risk_cross_sectional_r2=risk_r2,
            covariance_condition_number=float(np.linalg.cond(covariance_values)),
            predicted_weekly_volatility=volatility,
            parametric_weekly_cvar_95=2.0627 * volatility,
            effective_asset_count=float(1.0 / np.sum(values**2)),
            maximum_weight=float(final.max()),
            maximum_sector_weight=float(sector_weight.max()),
            maximum_absolute_style_exposure=(
                float(style_exposure.abs().max()) if len(style_exposure) else 0.0
            ),
            target_turnover_l1=float(change.abs().sum()) if has_previous else 0.0,
            estimated_spread_cost=spread_cost if has_previous else 0.0,
            estimated_temporary_impact=temporary if has_previous else 0.0,
            estimated_permanent_impact=permanent if has_previous else 0.0,
            estimated_total_execution_cost=(
                spread_cost + temporary + permanent if has_previous else 0.0
            ),
            maximum_participation_utilization=(
                float(utilization.max(skipna=True))
                if has_previous and utilization.notna().any()
                else 0.0
            ),
            optimizer_success=optimizer_success,
            optimizer_repaired=optimizer_repaired,
            risk_limit_relaxed=risk_limit_relaxed,
            effective_weekly_cvar_limit=effective_cvar_limit,
            unrepresentable_exit_weight=omitted_weight if has_previous else 0.0,
            signal_weights=signal_weights.to_dict(),
            signal_rank_ics=signal_ics.to_dict(),
            signal_ic_observations=self._ic_weight.to_dict(),
            signal_coverages=signal_coverage.to_dict(),
        )
        self._engine_current_weights = None
        return final

    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        long_weights = self._run_long_engine(returns)
        self.last_short_overlay_weights = pd.Series(dtype=float)
        self.last_short_overlay_cost = 0.0
        if not self.config.short_overlay_enabled:
            return long_weights
        short_weights = self._short_overlay_weights(returns, long_weights)
        if short_weights is None or short_weights.empty:
            return long_weights
        return long_weights.add(short_weights, fill_value=0.0)
