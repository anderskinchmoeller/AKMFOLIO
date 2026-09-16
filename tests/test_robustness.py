import numpy as np
import pandas as pd
import pytest

from akm_hrp.diagnostics.robustness import (
    block_bootstrap,
    causal_volatility_regimes,
    path_metrics,
    regime_hac,
    stratified_block_bootstrap,
    structural_break_regimes,
)


def _synthetic_paired(seed: int = 0, n: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-04", periods=n, freq="7D")
    bench = pd.Series(rng.normal(0.001, 0.01, n), index=dates)
    strat = bench + pd.Series(rng.normal(0.0002, 0.004, n), index=dates)
    return pd.concat([strat.rename("strategy"), bench.rename("benchmark")], axis=1)


def test_path_metrics_rejects_return_below_total_loss() -> None:
    with pytest.raises(ValueError):
        path_metrics(np.array([0.01, -1.5, 0.02]))


def test_causal_volatility_regimes_only_uses_past_information() -> None:
    rng = np.random.default_rng(1)
    n = 300
    dates = pd.bdate_range("2020-01-06", periods=n, freq="7D")
    calm = pd.Series(rng.normal(0.0, 0.005, n // 2), index=dates[: n // 2])
    stormy = pd.Series(rng.normal(0.0, 0.05, n - n // 2), index=dates[n // 2 :])
    series = pd.concat([calm, stormy])

    full_labels = causal_volatility_regimes(series)
    # Truncating the series after some date must not change any earlier
    # label -- a causal indicator can't be affected by data that comes later.
    cutoff = dates[200]
    truncated_labels = causal_volatility_regimes(series.loc[:cutoff])
    shared = full_labels.loc[:cutoff]
    assert (shared == truncated_labels).all()


def test_structural_break_regimes_is_causal_and_flags_a_real_shift() -> None:
    rng = np.random.default_rng(2)
    n = 400
    dates = pd.bdate_range("2018-01-05", periods=n, freq="7D")
    calm = rng.normal(0.001, 0.008, n // 2)
    stormy = rng.normal(-0.004, 0.035, n - n // 2)
    series = pd.Series(np.r_[calm, stormy], index=dates)

    labels = structural_break_regimes(series)
    assert (labels.iloc[:52] == "warmup").all()
    non_warmup = labels[labels != "warmup"]
    assert non_warmup.nunique() >= 2  # the real shift must produce a new segment

    # Causality: truncating the series must not change earlier labels.
    cutoff = dates[300]
    truncated = structural_break_regimes(series.loc[:cutoff])
    assert (labels.loc[:cutoff] == truncated).all()


def test_regime_hac_generalizes_beyond_low_high_vol_pair() -> None:
    rng = np.random.default_rng(3)
    n = 200
    dates = pd.bdate_range("2021-01-04", periods=n, freq="7D")
    active = pd.Series(rng.normal(0.0005, 0.01, n), index=dates)
    labels = pd.Series(
        np.where(np.arange(n) < n // 3, "segment_0",
                 np.where(np.arange(n) < 2 * n // 3, "segment_1", "segment_2")),
        index=dates, name="regime",
    )
    result = regime_hac(active, labels)
    assert set(result["regime"]) == {"segment_0", "segment_1", "segment_2"}
    # Three regimes -> no pairwise contrast row is added (only defined for exactly two).
    assert not any("minus" in r for r in result["regime"])


def test_regime_hac_keeps_original_high_minus_low_label() -> None:
    active = pd.Series(np.linspace(-0.01, 0.01, 40))
    labels = pd.Series(["low_vol"] * 20 + ["high_vol"] * 20, name="regime")
    result = regime_hac(active, labels)
    assert "high_minus_low" in set(result["regime"])


def test_stratified_bootstrap_gives_each_regime_a_share_of_every_draw() -> None:
    paired = _synthetic_paired(seed=7, n=300)
    # A rare regime with very few weeks nested inside a dominant one.
    regimes = pd.Series("common", index=paired.index, name="regime")
    regimes.iloc[100:106] = "rare"

    draws = stratified_block_bootstrap(paired, regimes, samples=50, block_weeks=8, seed=5)
    assert not draws.empty
    assert set(draws["model"]) == {"strategy", "benchmark", "active"}
    assert draws["draw"].nunique() == 50


def test_stratified_bootstrap_rejects_gaps_like_plain_bootstrap() -> None:
    paired = _synthetic_paired(seed=8, n=50)
    paired.iloc[10, 0] = np.nan
    regimes = pd.Series("common", index=paired.index)
    with pytest.raises(ValueError):
        stratified_block_bootstrap(paired, regimes, samples=10, block_weeks=4, seed=0)


def test_plain_and_stratified_bootstrap_agree_with_single_regime() -> None:
    """With one regime covering every row, stratified bootstrap should be
    statistically equivalent to the plain pooled bootstrap (same seed draws
    the same blocks from the same single group)."""
    paired = _synthetic_paired(seed=9, n=120)
    regimes = pd.Series("only_regime", index=paired.index)

    plain = block_bootstrap(paired, samples=30, block_weeks=6, seed=11)
    stratified = stratified_block_bootstrap(paired, regimes, samples=30, block_weeks=6, seed=11)

    plain_medians = plain.groupby("model")["sharpe"].median()
    stratified_medians = stratified.groupby("model")["sharpe"].median()
    pd.testing.assert_series_equal(
        plain_medians.sort_index(), stratified_medians.sort_index(), check_names=False
    )
