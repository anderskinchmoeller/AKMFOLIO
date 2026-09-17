"""Tests for the HRP-orthogonal structural signals."""

import numpy as np
import pandas as pd
import pytest

from akm_hrp.allocators.retail_alpha_ml_mpc import (
    RetailAlphaMLMPCAllocator,
    RetailAlphaMLMPCConfig,
)
from akm_hrp.data.microstructure_features import (
    MICROSTRUCTURE_FEATURE_COLUMNS,
    passive_event_intensity,
    weekly_microstructure_features,
)
from akm_hrp.signals.hrp_orthogonal import (
    OrthogonalizationConfig,
    build_basis,
    correlation_clusters,
    exposure_diagnostics,
    orthogonalize,
    shrunk_beta,
)
from akm_hrp.signals.regime_hmm import GaussianHMM, market_regime_features
from akm_hrp.signals.structural_alpha import (
    STRUCTURAL_SIGNALS,
    StructuralAlphaConfig,
)


def _factor_panel(n_assets=400, n_weeks=156, n_groups=5, seed=0):
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, n_groups, n_assets)
    beta = rng.uniform(0.5, 1.5, n_assets)
    market = rng.normal(0.001, 0.02, n_weeks)
    factors = rng.normal(0.0, 0.02, (n_weeks, n_groups))
    values = (
        market[:, None] * beta
        + factors[:, groups]
        + rng.normal(0.0, 0.02, (n_weeks, n_assets))
    )
    dates = pd.date_range("2015-01-02", periods=n_weeks, freq="W-FRI")
    frame = pd.DataFrame(values, index=dates, columns=[str(10_000 + i) for i in range(n_assets)])
    return frame, groups, beta


def test_orthogonalized_signal_has_no_cluster_or_beta_exposure():
    returns, groups, true_beta = _factor_panel()
    clusters = correlation_clusters(returns, OrthogonalizationConfig(max_clusters=10))
    beta = shrunk_beta(returns)
    assert np.corrcoef(beta, true_beta)[0, 1] > 0.8
    # Clusters recover the planted groups (high purity).
    purity = (
        pd.crosstab(clusters, groups).max(axis=1).sum() / len(groups)
    )
    assert purity > 0.9

    rng = np.random.default_rng(1)
    raw = pd.Series(3.0 * groups + 2.0 * true_beta + rng.normal(0, 1, len(groups)),
                    index=returns.columns)
    available = pd.Series(True, index=returns.columns)
    available.iloc[:20] = False
    basis = build_basis(returns.columns, beta, clusters)
    clean = orthogonalize(raw, available, basis)

    before = exposure_diagnostics(raw, available, clusters, beta)
    after = exposure_diagnostics(clean, available, clusters, beta)
    assert before["cluster_r2"] > 0.5
    assert after["cluster_r2"] < 1e-10
    assert abs(after["beta_corr"]) < 1e-10
    live = clean[available]
    assert abs(live.mean()) < 1e-12
    assert live.std(ddof=0) == pytest.approx(1.0)
    assert (clean[~available] == 0.0).all()


def test_passive_calendar_known_dates_and_no_future_snapping():
    dates = pd.bdate_range("2026-01-01", "2026-12-31")
    intensity = passive_event_intensity(dates)
    assert intensity[pd.Timestamp("2026-06-26")] >= 1.5  # Russell June (4th Fri)
    assert intensity[pd.Timestamp("2026-12-11")] >= 1.5  # Russell December
    assert intensity[pd.Timestamp("2026-03-20")] >= 1.0  # 3rd Friday of March
    assert intensity[pd.Timestamp("2026-06-19")] >= 1.0
    # A window that ends mid-month must not tag its last day as an event.
    partial = passive_event_intensity(pd.bdate_range("2026-03-01", "2026-03-10"))
    assert partial.iloc[-1] == 0.0
    # 2025 had no December reconstitution.
    old = passive_event_intensity(pd.bdate_range("2025-12-01", "2025-12-31"))
    assert old[pd.Timestamp("2025-12-12")] == 0.0


def _daily_bars(n_assets=6, start="2020-01-01", end="2020-12-31", seed=2):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    frames = []
    for p in range(n_assets):
        r = rng.normal(0, 0.02, len(dates))
        close = 30 * np.exp(np.cumsum(r))
        open_ = close / np.exp(rng.normal(0, 0.01, len(dates)))
        high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.01, len(dates))))
        low = np.minimum(open_, close) / np.exp(np.abs(rng.normal(0, 0.01, len(dates))))
        frames.append(pd.DataFrame({
            "permno": 20_000 + p, "date": dates,
            "ret": np.r_[0.0, np.exp(np.diff(np.log(close))) - 1.0],
            "volume": rng.lognormal(12, 0.4, len(dates)),
            "open": open_, "high": high, "low": low, "close": close,
        }))
    return pd.concat(frames, ignore_index=True)


def test_microstructure_features_are_causal():
    daily = _daily_bars()
    full = weekly_microstructure_features(daily)
    cut = pd.Timestamp("2020-08-14")
    truncated = weekly_microstructure_features(daily.loc[daily["date"] <= cut])
    merged = full.merge(truncated, on=["formation_date", "asset"], suffixes=("", "_t"))
    assert len(merged) == len(truncated)
    for column in MICROSTRUCTURE_FEATURE_COLUMNS:
        a, b = merged[column], merged[f"{column}_t"]
        assert np.allclose(a.fillna(-99), b.fillna(-99), atol=1e-6), column
    assert full["moc_close_dislocation_5d"].notna().mean() > 0.8
    assert full["vpin_proxy_20d"].dropna().between(0, 1).all()


def test_hmm_filter_does_not_use_future_observations():
    returns, _, _ = _factor_panel(n_assets=60, n_weeks=300)
    feats = market_regime_features(returns)
    x = ((feats - feats.mean()) / feats.std()).to_numpy()
    model = GaussianHMM(n_states=2, n_init=2).fit(x)
    full = model.filter(x)
    partial = model.filter(x[:150])
    assert np.allclose(full[:150], partial)
    # States are ordered by volatility.
    assert model.means_[0, 1] <= model.means_[1, 1]


def _allocator_inputs(n_assets=80, n_weeks=200, seed=5):
    rng = np.random.default_rng(seed)
    returns, groups, _ = _factor_panel(n_assets, n_weeks, n_groups=4, seed=seed)
    assets = returns.columns
    dates = returns.index
    # Plant a closing-dislocation reversal: next week's idiosyncratic return
    # is minus the current week's dislocation score.
    disloc = pd.DataFrame(rng.normal(0, 1, (n_weeks, n_assets)), index=dates, columns=assets)
    returns = returns + 0.004 * (-disloc.shift(1)).fillna(0.0)
    core = assets[:30]
    balanced_pit = pd.DataFrame(0, index=dates, columns=assets, dtype=np.int8)
    balanced_pit.loc[:, core] = 1
    rows = []
    caps = np.geomspace(3e8, 3e10, n_assets)
    for d in dates:
        frame = pd.DataFrame({
            "formation_date": d,
            "asset": assets,
            "market_cap_usd": caps,
            "price": 30.0,
            "dollar_volume_20d": np.geomspace(5e6, 8e7, n_assets),
            "amihud_20d": np.geomspace(0.02, 0.0002, n_assets),
            "zero_return_fraction_20d": 0.01,
            "book_to_market": np.linspace(0.2, 1.1, n_assets),
            "gross_profitability": np.linspace(0.08, 0.45, n_assets),
            "return_on_assets": np.linspace(0.01, 0.16, n_assets),
            "shareholder_carry": np.linspace(-0.02, 0.06, n_assets),
            "moc_close_dislocation_5d": disloc.loc[d].to_numpy(),
            "moc_event_dislocation_5d": 0.0,
        })
        for column in MICROSTRUCTURE_FEATURE_COLUMNS:
            if column not in frame:
                frame[column] = rng.normal(0, 1, n_assets)
        rows.append(frame)
    features = pd.concat(rows, ignore_index=True)
    sectors = pd.DataFrame({
        "permno": assets, "sec_info_start": "2000-01-01", "sec_info_end": "2030-01-01",
        "sector": [f"SECTOR_{n % 8}" for n in range(n_assets)],
    })
    return returns, balanced_pit, features, sectors


def _config(**overrides):
    base = dict(
        minimum_history_weeks=52,
        risk_lookback_weeks=78,
        ml_max_total_assets=40,
        maximum_added_assets=6,
        max_added_per_sector=3,
        max_weight=0.10,
        max_sector_weight=0.40,
        max_absolute_style_exposure=1.0,
        weekly_cvar_95_limit=0.30,
        portfolio_value=100_000.0,
        planning_horizon=2,
        optimizer_max_iterations=60,
        short_overlay_enabled=False,
        allow_cvar_floor_relaxation=True,
        ml_minimum_training_cross_sections=8,
    )
    base.update(overrides)
    return RetailAlphaMLMPCConfig(**base)


def test_disabled_structural_signals_keep_original_signal_set():
    returns, pit, features, sectors = _allocator_inputs(n_weeks=90)
    allocator = RetailAlphaMLMPCAllocator(
        pit, structural_features=features, sector_history=sectors, config=_config()
    )
    assert allocator.signal_names == RetailAlphaMLMPCAllocator.signal_names
    assert allocator._structural_engine is None
    with pytest.raises(ValueError):
        RetailAlphaMLMPCAllocator(
            pit, structural_features=features, sector_history=sectors,
            config=_config(structural_signals=("not_a_signal",)),
        )


def test_structural_signals_end_to_end():
    returns, pit, features, sectors = _allocator_inputs()
    structural = StructuralAlphaConfig(
        min_names=20,
        ml_min_training_cross_sections=6,
        ml_retrain_every_n_rebalances=2,
        ml_min_child_samples=20,
        ml_n_estimators=30,
        orthogonalization=OrthogonalizationConfig(max_clusters=6, min_cluster_size=5),
    )
    allocator = RetailAlphaMLMPCAllocator(
        pit,
        structural_features=features,
        sector_history=sectors,
        config=_config(structural_signals=STRUCTURAL_SIGNALS, structural_alpha=structural),
    )
    assert set(STRUCTURAL_SIGNALS) <= set(allocator.signal_names)
    for end in range(110, 200, 2):
        weights = allocator.allocate(returns.iloc[:end])
        assert np.isfinite(weights).all()
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        assert (weights >= -1e-12).all()

    history = pd.DataFrame(allocator._structural_engine.diagnostics_history)
    moc = history["moc_dislocation_reversal_cluster_r2_clean"].dropna()
    assert len(moc) and (moc < 1e-8).all()
    assert history["moc_dislocation_reversal_beta_corr_clean"].abs().max() < 1e-8
    assert history["moc_dislocation_reversal_coverage"].iloc[-1] > 0.9
    assert history["ml_training_cross_sections"].iloc[-1] >= 6
    assert history["microstructure_regime_ml_coverage"].iloc[-1] > 0.5
    # The planted reversal is picked up by the parent's online IC learner.
    ics = allocator._ic_sum / allocator._ic_weight.replace(0.0, np.nan)
    assert ics["moc_dislocation_reversal"] > 0.05
    assert np.isfinite(allocator.last_structural_diagnostics["regime_p0"])
    # Orthogonality checks and signal weights reach the diagnostics CSV.
    row = allocator.last_diagnostics.as_dict()
    assert "structural__moc_dislocation_reversal_cluster_r2_clean" in row
    assert "signal_weight__regime_conditional_momentum" in row


def test_stale_microstructure_file_switches_signals_off():
    returns, pit, features, sectors = _allocator_inputs(n_weeks=140)
    cutoff = returns.index[100]
    structural = StructuralAlphaConfig(
        signals=("moc_dislocation_reversal",),
        min_names=20,
        orthogonalization=OrthogonalizationConfig(max_clusters=6, min_cluster_size=5),
        microstructure_valid_through=cutoff,
    )
    allocator = RetailAlphaMLMPCAllocator(
        pit,
        structural_features=features,
        sector_history=sectors,
        config=_config(
            structural_signals=("moc_dislocation_reversal",),
            structural_alpha=structural,
        ),
    )
    allocator.allocate(returns.iloc[:95])
    assert allocator.last_structural_diagnostics["moc_dislocation_reversal_coverage"] > 0.9
    allocator.allocate(returns.iloc[:110])
    assert allocator.last_structural_diagnostics["moc_dislocation_reversal_coverage"] == 0.0


def test_betting_against_beta_is_low_beta_within_clusters():
    from akm_hrp.signals.structural_alpha import BAB_SIGNAL, StructuralAlphaEngine

    returns, groups, true_beta = _factor_panel(n_weeks=300)
    cfg = StructuralAlphaConfig(signals=(BAB_SIGNAL,), min_names=20)
    engine = StructuralAlphaEngine(cfg)
    raw, mask = engine._bab_raw(returns)
    assert mask.mean() > 0.95
    # Raw signal is minus beta: strongly negatively related to the true beta.
    assert np.corrcoef(raw[mask], true_beta[mask.to_numpy()])[0, 1] < -0.8

    clusters = correlation_clusters(returns, OrthogonalizationConfig(max_clusters=10))
    basis = build_basis(returns.columns, shrunk_beta(returns), clusters, None)
    clean = orthogonalize(raw, mask, basis.drop(columns=["BETA"]))
    live = mask & clean.ne(0.0)
    diag = exposure_diagnostics(clean, live, clusters, shrunk_beta(returns))
    assert diag["cluster_r2"] < 1e-10                # cluster-neutral
    assert abs(clean[live].mean()) < 1e-10 and clean[live].std(ddof=0) == pytest.approx(1.0)
    assert diag["beta_corr"] < -0.7                  # still a low-beta bet


def test_core_selection_mode_holds_whole_balanced_core():
    returns, pit, features, sectors = _allocator_inputs(n_weeks=120)
    core = pit.columns[pit.iloc[-1].astype(bool)]
    books = {}
    for mode in ("signal", "core"):
        allocator = RetailAlphaMLMPCAllocator(
            pit, structural_features=features, sector_history=sectors,
            config=_config(ml_selection_mode=mode, ml_max_total_assets=20,
                           ml_min_weight=0.05),
        )
        for end in (100, 110, 120):
            weights = allocator.allocate(returns.iloc[:end])
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
        books[mode] = weights[weights > 1e-12].index
    assert set(books["core"]) <= set(core)
    assert len(books["core"]) > 20            # no budget trim, no floor
    assert len(books["signal"]) <= 20
    with pytest.raises(ValueError):
        RetailAlphaMLMPCAllocator(
            pit, structural_features=features, sector_history=sectors,
            config=_config(ml_selection_mode="random"),
        )
