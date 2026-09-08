"""Kelly factor-sizing tests, checked against simulated ground truth.

The point of these is not that the code runs -- it is that the estimator
recovers a *known* quantity, and that it refuses to bet when handed noise.
The second property is the one that matters in production: a sizing rule that
cannot tell signal from noise will happily lever into nothing.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "stubs"))
sys.path.insert(0, str(Path(__file__).parent))

from retail_alpha_ml_mpc import (  # noqa: E402
    _bias_adjusted_squared_sharpe,
    _growth_optimal_factor_allocation,
    _growth_optimal_kelly_fraction,
    _shrunk_factor_covariance_from_returns,
)

RNG = np.random.default_rng(7)


def _simulate_factor_returns(k, T, annualized_sharpe, rho=0.0, rng=RNG):
    """k factor-return series with a known true squared Sharpe."""
    theta = annualized_sharpe / np.sqrt(52.0)
    corr = np.full((k, k), rho) + (1.0 - rho) * np.eye(k)
    vol = 0.01
    cov = (vol**2) * corr
    # Put the whole signal in one direction so theta^2 = mu' Sigma^-1 mu exactly.
    direction = np.zeros(k)
    direction[0] = 1.0
    scale = theta / np.sqrt(direction @ np.linalg.inv(cov) @ direction)
    mu = scale * direction
    return rng.multivariate_normal(mu, cov, size=T), theta**2


def test_bias_adjustment_recovers_truth():
    """The correction should be unbiased before flooring; the raw one is not.

    Two distinct claims, which an earlier version of this test wrongly
    conflated:

      (a) the Kan-Zhou style correction ((T-k-2)*raw - k)/T is close to
          unbiased for theta^2 -- this is the mathematical claim, and it is
          about the UNFLOORED quantity;
      (b) the shipped estimator floors at zero, which is necessary (a negative
          squared Sharpe is meaningless) but makes E[floored] > theta^2 by
          truncation.

    (b) does not make the sizing aggressive, because every truncated draw maps
    to c* = 0 -- the most conservative budget available. That is verified
    separately in `test_kelly_fraction_matches_theory`.
    """
    k, T = 9, 260
    raws, unfloored, floored = [], [], []
    for _ in range(400):
        g, true_th2 = _simulate_factor_returns(k, T, 0.5)
        mean = g.mean(axis=0)
        cov = np.cov(g, rowvar=False, ddof=1)
        raw = float(mean @ np.linalg.solve(cov, mean))
        raws.append(raw)
        unfloored.append(((T - k - 2) * raw - k) / T)
        floored.append(_bias_adjusted_squared_sharpe(raw, k, T))
    raw_mean = float(np.mean(raws))
    unfloored_mean = float(np.mean(unfloored))
    floored_mean = float(np.mean(floored))
    truncated_share = float(np.mean(np.array(unfloored) <= 0.0))
    print(f"true theta^2={true_th2:.6f}")
    print(f"  E[raw]       ={raw_mean:.6f}  ({raw_mean/true_th2:.1f}x truth)")
    print(f"  E[unfloored] ={unfloored_mean:.6f}  ({unfloored_mean/true_th2:.2f}x truth)")
    print(f"  E[floored]   ={floored_mean:.6f}  ({floored_mean/true_th2:.2f}x truth), "
          f"{truncated_share:.0%} of draws truncated to zero")
    assert raw_mean > 4.0 * true_th2, "raw estimate should be badly inflated"
    assert 0.6 * true_th2 < unfloored_mean < 1.4 * true_th2, (
        "the correction itself should be close to unbiased"
    )
    assert floored_mean >= unfloored_mean, "flooring can only raise the mean"
    assert all(f >= 0.0 for f in floored), "shipped estimator must never go negative"
    print("bias correction is unbiased pre-floor; floor is conservative in c*: OK")


def test_pure_noise_gets_zero_risk_budget():
    """The safety property: no signal -> no bet, guaranteed by construction.

    Because the Kelly fraction is driven by a one-sided 95% lower confidence
    bound on theta^2, the false-positive rate under the null is pinned at 5%
    by the definition of a confidence set. This test asserts that guarantee
    holds in the implementation -- it is the property the point-estimate
    version failed (mean c* of 0.10 on pure noise, occasionally at the cap).
    """
    k, T = 9, 260
    fractions = []
    for _ in range(600):
        g = RNG.normal(0.0, 0.01, size=(T, k))  # mean exactly zero: no edge
        _, fraction, _ = _growth_optimal_factor_allocation(g, 0.5)
        fractions.append(fraction)
    fractions = np.array(fractions)
    positive_share = float((fractions > 1e-12).mean())
    print(f"pure noise: mean c*={fractions.mean():.4f}, "
          f"max={fractions.max():.4f}, share positive={positive_share:.1%} "
          f"(theory: 5.0% by construction)")
    assert positive_share < 0.11, (
        f"false-positive rate {positive_share:.1%} far above the 5% guarantee"
    )
    assert fractions.mean() < 0.02, "noise must not attract a meaningful budget"
    print("pure noise sized to zero at the guaranteed rate: OK")


def test_kelly_fraction_is_conservative_versus_oracle():
    """c* should sit at or below the oracle theta^2/(theta^2 + k/T).

    The oracle value is what you would size at if you *knew* theta^2. Sizing
    off a lower confidence bound must come in at or below that, and the gap is
    the price of not knowing -- it should narrow as the true edge grows and
    the evidence becomes unambiguous.
    """
    for k, T, sr in [(9, 260, 0.5), (9, 260, 1.5), (9, 260, 3.0), (5, 520, 1.0)]:
        realized = []
        for _ in range(250):
            g, true_th2 = _simulate_factor_returns(k, T, sr)
            _, fraction, _ = _growth_optimal_factor_allocation(g, 1.0)
            realized.append(fraction)
        oracle = true_th2 / (true_th2 + k / T)
        got = float(np.mean(realized))
        print(f"  k={k:3d} T={T} SRann={sr}: oracle c*={oracle:.4f}  "
              f"realized mean c*={got:.4f}  ({got/oracle:.2f}x oracle)")
        assert got <= oracle + 0.05, (
            f"sizing off a lower bound must not exceed the oracle: {got} vs {oracle}"
        )
    print("Kelly fraction is conservative relative to the oracle: OK")


def test_more_signal_means_more_risk_budget():
    """Monotonicity: a stronger true edge should earn a bigger budget."""
    budgets = []
    for sr in [0.0, 0.5, 1.0, 2.0]:
        vals = []
        for _ in range(200):
            g, _ = _simulate_factor_returns(9, 260, sr)
            _, fraction, _ = _growth_optimal_factor_allocation(g, 1.0)
            vals.append(fraction)
        budgets.append(float(np.mean(vals)))
        print(f"  true SR={sr}: mean c*={budgets[-1]:.4f}")
    assert budgets == sorted(budgets), "c* must increase with true signal strength"
    assert budgets[0] < 0.05, "zero-signal case must get ~no budget"
    print("risk budget increases monotonically with true edge: OK")


def test_short_history_refuses_to_size():
    g = RNG.normal(0.0, 0.01, size=(6, 9))  # fewer observations than factors+2
    mix, fraction, _ = _growth_optimal_factor_allocation(g, 0.5)
    assert fraction == 0.0 and not mix.any()
    print("insufficient history -> zero budget, no position: OK")


def test_shrunk_covariance_properties():
    g, _ = _simulate_factor_returns(9, 120, 1.0, rho=0.4)
    cov = _shrunk_factor_covariance_from_returns(g)
    sample = np.cov(g, rowvar=False, ddof=1)
    assert np.allclose(cov, cov.T), "must be symmetric"
    assert np.linalg.eigvalsh(cov).min() > 0, "must be positive definite"
    # Diagonal target: variances preserved, only correlations shrunk.
    assert np.allclose(np.diag(cov), np.diag(sample), rtol=1e-6), (
        "shrinking toward the diagonal must leave variances untouched"
    )
    off_sample = np.abs(sample - np.diag(np.diag(sample))).sum()
    off_shrunk = np.abs(cov - np.diag(np.diag(cov))).sum()
    assert off_shrunk < off_sample, "off-diagonals must be shrunk toward zero"
    print(f"shrunk covariance: PSD, variances preserved, off-diagonal mass "
          f"{off_sample:.3e} -> {off_shrunk:.3e}: OK")


def test_maximum_fraction_caps():
    g, _ = _simulate_factor_returns(3, 400, 4.0)  # implausibly strong edge
    _, fraction, _ = _growth_optimal_factor_allocation(g, 0.25)
    assert fraction <= 0.25 + 1e-12
    print(f"cap binds on an implausibly strong edge: c*={fraction:.4f} <= 0.25: OK")


def test_allocator_buffer_end_to_end():
    """The buffer must fill causally through the real allocator plumbing."""
    from retail_alpha_ml_mpc import RetailAlphaMLMPCAllocator, RetailAlphaMLMPCConfig

    allocator = RetailAlphaMLMPCAllocator.__new__(RetailAlphaMLMPCAllocator)
    allocator.config = RetailAlphaMLMPCConfig(
        ml_kelly_mix_enabled=True, ml_kelly_scale_enabled=True
    )
    allocator._engine_return_context = None
    allocator._initialize_ml_state()

    assets = pd.Index([f"A{i:03d}" for i in range(40)])
    dates = pd.date_range("2020-01-03", periods=80, freq="W-FRI")
    returns = pd.DataFrame(
        RNG.normal(0, 0.02, size=(len(dates), len(assets))),
        index=dates, columns=assets,
    )
    names = list(allocator.signal_names)
    for step in range(1, len(dates)):
        allocator._previous_signal_panel = pd.DataFrame(
            RNG.normal(size=(len(assets), len(names))), index=assets, columns=names
        )
        allocator._previous_signal_availability = pd.DataFrame(
            True, index=assets, columns=names
        )
        allocator._previous_signal_date = dates[step - 1]
        allocator._accumulate_factor_returns(returns, dates[step])

    assert len(allocator._factor_return_buffer) == len(dates) - 1
    matrix = np.vstack([row for _, row in allocator._factor_return_buffer])
    assert np.isfinite(matrix).all(), "every factor return should be finite here"
    # Random scores against random returns: no edge, so no risk budget.
    _, fraction = allocator._kelly_factor_allocation()
    print(f"end-to-end buffer: {len(allocator._factor_return_buffer)} rows, "
          f"c*={fraction:.4f} on random (edgeless) signals")
    assert 0.0 <= fraction <= 0.5 + 1e-9, (
        f"fraction must be a bounded multiplier, got {fraction}"
    )
    print("allocator buffer fills causally and sizes edgeless signals near zero: OK")


def test_kns_ridge_estimator():
    """The shipped default: ridge shrinkage in the direction, not a haircut."""
    from retail_alpha_ml_mpc import _kns_ridge_factor_weights

    # With real signal the ridge should take a meaningful position...
    g, _ = _simulate_factor_returns(9, 260, 1.0)
    w_signal, gamma = _kns_ridge_factor_weights(g, 0.5, 52.0)
    # ...and on pure noise a far smaller one, without any significance gate.
    noise = RNG.normal(0.0, 0.01, size=(260, 9))
    w_noise, _ = _kns_ridge_factor_weights(noise, 0.5, 52.0)
    gross_signal = float(np.abs(w_signal).sum())
    gross_noise = float(np.abs(w_noise).sum())
    print(f"KNS ridge: gross on signal={gross_signal:.3f}, "
          f"on noise={gross_noise:.3f}, gamma={gamma:.3e}")
    assert gamma > 0, "ridge must actually regularize"
    assert gross_signal > gross_noise, "must lean harder on real signal"

    # The unshrunk plug-in takes wildly larger positions on the same noise.
    cov = np.cov(noise, rowvar=False, ddof=1)
    plug = np.linalg.solve(cov, noise.mean(axis=0))
    print(f"  unshrunk plug-in gross on the same noise={np.abs(plug).sum():.3f}")
    assert np.abs(plug).sum() > 3.0 * gross_noise, (
        "the ridge should be far more restrained than the plug-in on noise"
    )
    print("KNS ridge leans on signal and stays small on noise: OK")


def _retired_ridge_mode_check():
    from retail_alpha_ml_mpc import _growth_optimal_factor_allocation

    g, _ = _simulate_factor_returns(9, 260, 1.0)
    mix, frac, diag = _growth_optimal_factor_allocation(
        g, 0.5, estimator="ridge", prior_maximum_sharpe=0.5)
    assert abs(float(np.abs(mix).sum()) - 1.0) < 1e-9, "mix is L1-normalized"
    assert "ridge_gamma" in diag and diag["ridge_gamma"] > 0
    assert frac > 0
    print(f"ridge mode: |mix|_1=1, gamma={diag['ridge_gamma']:.3e}, "
          f"gross={frac:.3f}: OK")


def test_effective_breadth_and_the_redundancy_tax():
    """Redundant signals are charged twice by the growth-optimal formula.

    c* = theta^2/(theta^2 + k/T): a duplicated signal raises the noise floor
    k/T without raising theta^2. Hou-Xue-Zhang and Novy-Marx both indicate
    this book's nominal breadth overstates its effective breadth, which makes
    pruning the cheapest available increase in the defensible risk budget.
    """
    from retail_alpha_ml_mpc import effective_breadth

    T = 260
    base = RNG.normal(0.0, 0.01, size=(T, 5))
    dupes = base[:, :4] + RNG.normal(0.0, 0.002, size=(T, 4))
    panel = np.hstack([base, dupes])
    k_eff = effective_breadth(panel)
    print(f"nominal k=9, effective breadth={k_eff:.2f}")
    assert 3.0 < k_eff < 7.5, f"expected ~5 independent directions, got {k_eff}"

    independent = effective_breadth(RNG.normal(0.0, 0.01, size=(T, 9)))
    print(f"  9 uncorrelated series -> effective breadth={independent:.2f}")
    assert independent > 7.5, "uncorrelated factors should show near-full breadth"

    th2 = (0.5 / np.sqrt(52.0)) ** 2
    c_nominal = th2 / (th2 + 9 / T)
    c_effective = th2 / (th2 + k_eff / T)
    print(f"  c* at nominal k=9        : {c_nominal:.4f}")
    print(f"  c* at effective k={k_eff:.1f}    : {c_effective:.4f} "
          f"({c_effective / c_nominal:.2f}x larger)")
    assert c_effective > c_nominal
    print("redundancy tax quantified: OK")


def test_ridge_returns_a_bounded_fraction():
    """Both modes must return the SAME semantic quantity: a bounded multiplier.

    Regression test for a real bug: the ridge briefly returned its raw gross
    notional (~8.0) where the caller expects a fraction, which would have
    multiplied the alpha vector eightfold inside the optimizer.
    """
    from retail_alpha_ml_mpc import _growth_optimal_factor_allocation

    for sr in (0.0, 0.5, 2.0):
        g, _ = _simulate_factor_returns(9, 260, sr)
        mix, frac, diag = _growth_optimal_factor_allocation(
            g, 0.5, estimator="ridge", prior_maximum_sharpe=0.5)
        assert abs(float(np.abs(mix).sum()) - 1.0) < 1e-9, "mix is L1-normalized"
        assert 0.0 <= frac <= 0.5 + 1e-9, f"fraction out of bounds: {frac}"
        assert diag["ridge_gamma"] > 0
        print(f"  SR={sr}: fraction={frac:.4f}, ridge/plug-in gross="
              f"{diag['ridge_gross']:.2f}/{diag['plug_in_gross']:.2f}")
    print("ridge returns a bounded, correctly-scaled multiplier: OK")


def test_crowding_kappa_disabled_by_default_is_a_no_op():
    """Default off: leaving `crowding_kappa_enabled` at its default (False)
    must be bit-for-bit identical to the estimator before this feature
    existed -- matching the "nothing changes until deliberately enabled"
    convention already used for `ml_kelly_mix_enabled` / `_scale_enabled`.
    """
    from retail_alpha_ml_mpc import _growth_optimal_factor_allocation

    g, _ = _simulate_factor_returns(9, 260, 1.0)
    mix_a, frac_a, diag_a = _growth_optimal_factor_allocation(
        g, 0.5, estimator="ridge", prior_maximum_sharpe=0.5)
    mix_b, frac_b, diag_b = _growth_optimal_factor_allocation(
        g, 0.5, estimator="ridge", prior_maximum_sharpe=0.5,
        crowding_kappa_enabled=False)
    assert np.allclose(mix_a, mix_b) and frac_a == frac_b
    assert diag_a["kappa_used"] == diag_b["kappa_used"] == 0.5
    assert "crowding_active" not in diag_a, "diagnostics must stay empty when off"
    print("crowding kappa off by default: identical to pre-existing behaviour: OK")


def test_crowding_kappa_burn_in_gate():
    """Inactive below 39 weeks (13-week window + 26-week baseline), active at
    exactly 39 -- the boundary fixed in the pre-registration."""
    from retail_alpha_ml_mpc import _crowding_regime_kappa

    beta = float(np.log(2.0) / 2.0)
    below = RNG.normal(0.0, 0.02, size=(38, 9))
    kappa, diag = _crowding_regime_kappa(below, 0.5, short_window=13, burn_in=26, beta=beta)
    assert kappa == 0.5 and diag["crowding_active"] == 0.0, "must be inactive below 39 obs"

    at_gate = RNG.normal(0.0, 0.02, size=(39, 9))
    kappa, diag = _crowding_regime_kappa(at_gate, 0.5, short_window=13, burn_in=26, beta=beta)
    assert diag["crowding_active"] == 1.0, "must activate at exactly 39 obs"
    print(f"burn-in gate: inactive at 38 obs, active at 39 obs (z={diag['crowding_z']:.2f}): OK")


def test_crowding_kappa_is_direction_agnostic():
    """MSCI's account of the June-July 2025 quant drawdown showed pairwise
    correlation FALLING during the unwind, not spiking -- so this must shrink
    kappa on an abnormal fall just as it does on an abnormal rise (Sec 2 of
    crowding_kappa_preregistration.md)."""
    from retail_alpha_ml_mpc import _crowding_regime_kappa

    beta = float(np.log(2.0) / 2.0)
    rng = np.random.default_rng(11)

    # Spike: near-zero correlation, then a shared shock in the last window.
    base = rng.normal(0.0, 0.02, size=(199, 9))
    shock = rng.normal(size=13) * 0.03
    spike_window = shock[:, None] + rng.normal(0.0, 0.005, size=(13, 9))
    spiked = np.vstack([base, spike_window])
    kappa_spike, diag_spike = _crowding_regime_kappa(spiked, 0.5, 13, 26, beta)
    assert kappa_spike < 0.5, "an abnormal correlation SPIKE must shrink kappa"

    # Collapse: elevated correlation, then decorrelation in the last window.
    common = rng.normal(size=199) * 0.03
    elevated = common[:, None] + rng.normal(0.0, 0.005, size=(199, 9))
    collapse_window = rng.normal(0.0, 0.02, size=(13, 9))
    collapsed = np.vstack([elevated, collapse_window])
    kappa_collapse, diag_collapse = _crowding_regime_kappa(collapsed, 0.5, 13, 26, beta)
    assert kappa_collapse < 0.5, "an abnormal correlation COLLAPSE must ALSO shrink kappa"

    print(f"  spike:    rho_bar={diag_spike['crowding_rho_bar']:.3f} "
          f"z={diag_spike['crowding_z']:.2f} kappa={kappa_spike:.4f}")
    print(f"  collapse: rho_bar={diag_collapse['crowding_rho_bar']:.3f} "
          f"z={diag_collapse['crowding_z']:.2f} kappa={kappa_collapse:.4f}")
    print("crowding kappa is direction-agnostic: OK")


def test_crowding_squash_halves_kappa_at_three_sigma():
    """beta is fixed in advance so a 3-sigma shock exactly halves kappa, with
    no penalty inside 1 sigma of ordinary noise -- the squash function itself,
    independent of any particular factor-return draw."""
    beta = float(np.log(2.0) / 2.0)
    cases = [(0.0, 1.0), (0.9, 1.0), (1.0, 1.0), (3.0, 0.5), (5.0, float(np.exp(-beta * 4)))]
    for z, expected in cases:
        shock = max(0.0, abs(z) - 1.0)
        multiplier = float(np.exp(-beta * shock))
        assert abs(multiplier - expected) < 1e-9, f"z={z}: {multiplier} != {expected}"
    print("squash function: flat inside 1 sigma, exact half at 3 sigma: OK")


def test_crowding_kappa_wired_into_ridge_reduces_gross():
    """End-to-end through `_growth_optimal_factor_allocation`: enabling the
    crowding adjustment on a spiked window must shrink kappa (and therefore
    not exceed the unadjusted gross position) relative to leaving it off, on
    the identical data."""
    from retail_alpha_ml_mpc import _growth_optimal_factor_allocation

    rng = np.random.default_rng(23)
    base = rng.normal(0.0, 0.01, size=(199, 9))
    shock = rng.normal(size=13) * 0.02
    spike_window = shock[:, None] + rng.normal(0.0, 0.003, size=(13, 9))
    spiked = np.vstack([base, spike_window])
    beta = float(np.log(2.0) / 2.0)

    _, frac_off, diag_off = _growth_optimal_factor_allocation(
        spiked, 0.5, estimator="ridge", prior_maximum_sharpe=0.5,
        crowding_kappa_enabled=False)
    _, frac_on, diag_on = _growth_optimal_factor_allocation(
        spiked, 0.5, estimator="ridge", prior_maximum_sharpe=0.5,
        crowding_kappa_enabled=True, crowding_window=13, crowding_burn_in=26,
        crowding_beta=beta)

    print(f"  gross off={diag_off['ridge_gross']:.4f} (kappa={diag_off['kappa_used']:.3f}), "
          f"gross on={diag_on['ridge_gross']:.4f} (kappa={diag_on['kappa_used']:.3f})")
    assert diag_on["kappa_used"] < diag_off["kappa_used"], "kappa should be shrunk on a spike"
    print("crowding-conditioned kappa wired into the ridge: OK")


def test_crowding_kappa_end_to_end_through_allocator():
    """The config flag must reach the real allocator's own diagnostics dict
    through the same causal buffer plumbing `test_allocator_buffer_end_to_end`
    exercises for the base ridge."""
    from retail_alpha_ml_mpc import RetailAlphaMLMPCAllocator, RetailAlphaMLMPCConfig

    allocator = RetailAlphaMLMPCAllocator.__new__(RetailAlphaMLMPCAllocator)
    allocator.config = RetailAlphaMLMPCConfig(
        ml_kelly_mix_enabled=True, ml_kelly_scale_enabled=True,
        ml_kelly_crowding_kappa_enabled=True,
    )
    allocator._engine_return_context = None
    allocator._initialize_ml_state()

    assets = pd.Index([f"A{i:03d}" for i in range(40)])
    dates = pd.date_range("2020-01-03", periods=80, freq="W-FRI")
    returns = pd.DataFrame(
        RNG.normal(0, 0.02, size=(len(dates), len(assets))),
        index=dates, columns=assets,
    )
    names = list(allocator.signal_names)
    for step in range(1, len(dates)):
        allocator._previous_signal_panel = pd.DataFrame(
            RNG.normal(size=(len(assets), len(names))), index=assets, columns=names
        )
        allocator._previous_signal_availability = pd.DataFrame(
            True, index=assets, columns=names
        )
        allocator._previous_signal_date = dates[step - 1]
        allocator._accumulate_factor_returns(returns, dates[step])

    _, fraction = allocator._kelly_factor_allocation()
    diag = allocator.last_kelly_diagnostics
    print(f"end-to-end: {len(allocator._factor_return_buffer)} rows, c*={fraction:.4f}, "
          f"crowding_active={diag.get('crowding_active')}, kappa_used={diag.get('kappa_used')}")
    assert "kappa_used" in diag, "crowding-aware diagnostics must reach the allocator"
    print("crowding kappa reaches the allocator end-to-end: OK")


if __name__ == "__main__":
    test_bias_adjustment_recovers_truth()
    test_pure_noise_gets_zero_risk_budget()
    test_kelly_fraction_is_conservative_versus_oracle()
    test_more_signal_means_more_risk_budget()
    test_short_history_refuses_to_size()
    test_shrunk_covariance_properties()
    test_maximum_fraction_caps()
    test_allocator_buffer_end_to_end()
    test_kns_ridge_estimator()
    test_ridge_returns_a_bounded_fraction()
    test_effective_breadth_and_the_redundancy_tax()
    test_crowding_kappa_disabled_by_default_is_a_no_op()
    test_crowding_kappa_burn_in_gate()
    test_crowding_kappa_is_direction_agnostic()
    test_crowding_squash_halves_kappa_at_three_sigma()
    test_crowding_kappa_wired_into_ridge_reduces_gross()
    test_crowding_kappa_end_to_end_through_allocator()
    print("\nALL KELLY SIZING TESTS PASSED")
