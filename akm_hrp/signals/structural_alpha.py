"""Three HRP-orthogonal alpha signals for the Retail Alpha ML MPC allocator.

``moc_dislocation_reversal``
    Structural microstructure / passive-flow imbalance. Scores the
    volume- and event-weighted gap between the close and the session VWAP
    proxy (plus a real closing-imbalance ratio when supplied) and bets on its
    short-horizon reversal, amplified in less liquid names where mechanical
    closing demand moves prices furthest from equilibrium.

``microstructure_regime_ml``
    Cross-sectional gradient-boosted model (LightGBM when installed, else
    scikit-learn's HistGradientBoosting) over order-flow toxicity, liquidity,
    price-path and optional ``sentiment_*`` / ``altdata_*`` features, each
    also interacted with the filtered HMM regime probabilities so the trees
    can learn e.g. that a volume spike means something different in a
    stress regime. Trained causally on forward returns that were already
    stripped of beta and HRP-cluster components, so the model is rewarded
    only for idiosyncratic forecasts; its prediction is additionally made
    orthogonal to the value/momentum/size/quality/low-vol styles.

``betting_against_beta``
    Low-beta anomaly (Black 1972; Frazzini & Pedersen 2014, "Betting Against
    Beta"): long low-beta names, underweight high-beta names. Beta is the
    Frazzini-Pedersen estimator -- correlation with the market over five
    years, volatility ratio over one year, shrunk 0.6/0.4 toward 1 -- on
    weekly returns. This is the one signal that cannot be made
    beta-neutral (beta *is* the signal), so it is residualised on the HRP
    clusters and industries only: it bets on low beta *within* each
    correlation cluster and industry (the industry-neutral BAB of Asness,
    Frazzini & Pedersen 2014), not on a defensive-sector rotation the HRP
    tree has already sized.

``regime_conditional_momentum``
    Momentum (12-1) and one-week reversal, each HRP-orthogonalised, blended
    with regime-specific information coefficients learned online and
    weighted by the forward-filtered HMM state probabilities. A signal whose
    IC in the current state is negative is reversed; a momentum-crash guard
    (bear market + stress state) switches momentum off. The output is scaled
    by a conviction factor in [0, 1], so the sleeve fades out when no
    regime-signal pair has a reliable edge.

Every output passes through :func:`hrp_orthogonal.orthogonalize`:
rank-normal -> residual on [1, beta, HRP clusters, industry(, styles)] ->
exact de-mean -> unit variance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from akm_hrp.data.microstructure_features import MICROSTRUCTURE_FEATURE_COLUMNS
from akm_hrp.signals.hrp_orthogonal import (
    ClusterCache,
    OrthogonalizationConfig,
    _rank_normal,
    build_basis,
    exposure_diagnostics,
    orthogonalize,
    shrunk_beta,
)
from akm_hrp.signals.regime_hmm import RegimeConfig, RegimeTracker

MOC_SIGNAL = "moc_dislocation_reversal"
ML_SIGNAL = "microstructure_regime_ml"
REGIME_SIGNAL = "regime_conditional_momentum"
BAB_SIGNAL = "betting_against_beta"
STRUCTURAL_SIGNALS = (MOC_SIGNAL, ML_SIGNAL, REGIME_SIGNAL, BAB_SIGNAL)

# Starting information coefficients for the parent's online IC learner.
STRUCTURAL_IC_PRIORS = {
    MOC_SIGNAL: 0.008,
    # Like the existing ML sleeve, the learned signal must earn its weight.
    ML_SIGNAL: 0.0,
    REGIME_SIGNAL: 0.008,
    # Small positive prior: decades of evidence, but the long-only book
    # captures the Sharpe side of BAB, not its levered return.
    BAB_SIGNAL: 0.008,
}

_EPS = 1e-12
_OPTIONAL_PREFIXES = ("sentiment_", "altdata_")
_LIQUIDITY_COLUMNS = (
    "amihud_20d",
    "turnover_20d",
    "realized_volatility_20d",
    "zero_return_fraction_20d",
    "dollar_volume_20d",
)
# Features that are interacted with the stress-state probability.
_REGIME_INTERACTED = (
    "volume_shock_5_60",
    "bvc_order_imbalance_5d",
    "vpin_proxy_20d",
    "moc_close_dislocation_5d",
    "ret_1w",
    "ret_4w",
)


@dataclass(frozen=True)
class StructuralAlphaConfig:
    signals: tuple[str, ...] = STRUCTURAL_SIGNALS
    orthogonalization: OrthogonalizationConfig = field(
        default_factory=OrthogonalizationConfig
    )
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    min_names: int = 30
    # Last formation date the microstructure feature file covers. The feature
    # store carries each asset's latest row forward indefinitely, so without
    # this a file ending in December would feed December's closing
    # dislocations into every later rebalance. After this date (+ one week of
    # grace) the microstructure columns are treated as missing.
    microstructure_valid_through: pd.Timestamp | None = None

    # --- MOC dislocation -------------------------------------------------
    moc_event_weight: float = 1.0
    moc_imbalance_weight: float = 1.0
    moc_illiquidity_amplification: float = 1.0

    # --- ML ----------------------------------------------------------------
    ml_backend: str = "auto"  # "auto" | "lightgbm" | "sklearn"
    ml_max_training_cross_sections: int = 156
    ml_min_training_cross_sections: int = 26
    ml_retrain_every_n_rebalances: int = 4
    ml_recency_halflife: float = 52.0
    ml_n_estimators: int = 200
    ml_learning_rate: float = 0.03
    ml_num_leaves: int = 15
    ml_min_child_samples: int = 200
    ml_l2: float = 5.0
    ml_target_clip: float = 2.5
    ml_random_state: int = 0

    # --- Regime-conditional momentum ---------------------------------------
    regime_ic_halflife: float = 52.0
    regime_ic_prior_strength: float = 26.0
    # Prior IC by state (calm -> stress) for (momentum, reversal). Momentum
    # is assumed to work in calm/normal markets and to crash in stress
    # (Daniel & Moskowitz 2016); short-term reversal is strongest when
    # liquidity provision is scarce (Nagel 2012).
    regime_prior_momentum: tuple[float, ...] = (0.03, 0.015, -0.01)
    regime_prior_reversal: tuple[float, ...] = (0.005, 0.01, 0.02)
    regime_conviction_reference_ic: float = 0.02
    regime_min_conviction: float = 0.10
    regime_crash_lookback_weeks: int = 104

    # --- Betting against beta (Frazzini & Pedersen 2014) --------------------
    bab_correlation_weeks: int = 260
    bab_volatility_weeks: int = 52
    bab_min_correlation_observations: int = 104
    bab_min_volatility_observations: int = 26
    bab_shrinkage: float = 0.6  # weight on the estimate; rest on beta = 1


def _ml_regressor(config: StructuralAlphaConfig):
    backend = config.ml_backend
    if backend in ("auto", "lightgbm"):
        try:
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=config.ml_n_estimators,
                learning_rate=config.ml_learning_rate,
                num_leaves=config.ml_num_leaves,
                min_child_samples=config.ml_min_child_samples,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=config.ml_l2,
                random_state=config.ml_random_state,
                n_jobs=1,
                verbose=-1,
            )
        except ImportError:
            if backend == "lightgbm":
                raise
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=config.ml_n_estimators,
        learning_rate=config.ml_learning_rate,
        max_leaf_nodes=config.ml_num_leaves,
        min_samples_leaf=config.ml_min_child_samples,
        l2_regularization=config.ml_l2,
        random_state=config.ml_random_state,
    )


def _col(snapshot: pd.DataFrame, name: str, assets: pd.Index) -> pd.Series:
    if name not in snapshot:
        return pd.Series(np.nan, index=assets, dtype=float)
    return pd.to_numeric(snapshot[name].reindex(assets), errors="coerce")


def _compound(returns: pd.DataFrame) -> pd.Series:
    return (1.0 + returns).prod(min_count=1) - 1.0


class StructuralAlphaEngine:
    """Stateful builder for the three structural signals."""

    def __init__(
        self,
        config: StructuralAlphaConfig | None = None,
        *,
        feature_columns: list[str] | tuple[str, ...] = (),
        macro: pd.DataFrame | None = None,
    ) -> None:
        self.config = config or StructuralAlphaConfig()
        unknown = set(self.config.signals) - set(STRUCTURAL_SIGNALS)
        if unknown:
            raise ValueError(f"Unknown structural signals: {sorted(unknown)}")
        self.clusters = ClusterCache(self.config.orthogonalization)
        self.regime = RegimeTracker(self.config.regime, macro=macro)
        optional = sorted(
            c for c in feature_columns if str(c).startswith(_OPTIONAL_PREFIXES)
        )
        base = (
            *MICROSTRUCTURE_FEATURE_COLUMNS,
            *_LIQUIDITY_COLUMNS,
            "log_size",
            "ret_1w",
            "ret_4w",
            "ret_13w",
            *optional,
        )
        # Date-constant columns carry no cross-sectional information.
        self.base_features = tuple(c for c in base if c != "moc_event_days_5d")
        k = self.config.regime.n_states
        self.ml_features = (
            *self.base_features,
            *[f"{c}_x_stress" for c in _REGIME_INTERACTED],
            *[f"regime_p{s}" for s in range(k)],
            "moc_event_days_5d",
        )
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.clusters.reset()
        self.regime.reset()
        k = self.config.regime.n_states
        self._ml_history: deque[tuple[np.ndarray, np.ndarray]] = deque(
            maxlen=self.config.ml_max_training_cross_sections
        )
        self._ml_model = None
        self._since_refit = 0
        self._ic_sum = np.zeros((k, 2))
        self._ic_weight = np.zeros((k, 2))
        self._previous: dict | None = None
        self.last_diagnostics: dict[str, float] = {}
        self.diagnostics_history: list[dict[str, float]] = []
        self.last_feature_importances = pd.Series(dtype=float)
        self.last_bab_beta = pd.Series(dtype=float)

    # -------------------------------------------------------------- learning
    def _learn(self, forward_source: pd.DataFrame, as_of: pd.Timestamp) -> None:
        prev = self._previous
        if prev is None or prev["date"] >= as_of:
            return
        rows = forward_source.loc[
            (forward_source.index > prev["date"]) & (forward_source.index <= as_of)
        ]
        if rows.empty:
            return
        forward = _compound(rows).reindex(prev["assets"])
        observed = forward.notna()
        if int(observed.sum()) < self.config.min_names:
            return
        # Residual (beta + cluster + industry neutral) forward return: the
        # only part of the return an HRP-orthogonal signal should predict.
        resid = orthogonalize(
            forward,
            observed,
            prev["basis"],
            weighting=self.config.orthogonalization.weighting,
            rank_transform=True,
            min_names=self.config.min_names,
        )
        names = prev["assets"][observed.to_numpy()]

        if prev.get("ml_x") is not None and ML_SIGNAL in self.config.signals:
            eligible = prev["ml_eligible"].reindex(names).fillna(False).to_numpy()
            if eligible.sum() >= self.config.min_names:
                x = prev["ml_x"].loc[names].to_numpy(dtype=np.float32)[eligible]
                y = resid.loc[names].clip(
                    -self.config.ml_target_clip, self.config.ml_target_clip
                ).to_numpy(dtype=np.float32)[eligible]
                self._ml_history.append((x, y))
                self._since_refit += 1
                if (
                    len(self._ml_history) >= self.config.ml_min_training_cross_sections
                    and (
                        self._ml_model is None
                        or self._since_refit >= self.config.ml_retrain_every_n_rebalances
                    )
                ):
                    self._refit()

        components = prev.get("regime_components")
        if components is not None and REGIME_SIGNAL in self.config.signals:
            decay = float(np.exp(np.log(0.5) / self.config.regime_ic_halflife))
            self._ic_sum *= decay
            self._ic_weight *= decay
            probs = prev["regime_probs"]
            fwd = resid.loc[names]
            for j, column in enumerate(components.columns):
                score = components[column].reindex(names)
                usable = score.ne(0.0) & score.notna()
                if usable.sum() < self.config.min_names:
                    continue
                ic = spearmanr(score[usable], fwd[usable]).statistic
                if not np.isfinite(ic):
                    continue
                ic = float(np.clip(ic, -0.2, 0.2))
                self._ic_sum[:, j] += probs * ic
                self._ic_weight[:, j] += probs

    def _refit(self) -> None:
        x = np.concatenate([e[0] for e in self._ml_history])
        y = np.concatenate([e[1] for e in self._ml_history])
        decay = float(np.exp(np.log(0.5) / self.config.ml_recency_halflife))
        weights = np.concatenate(
            [
                np.full(len(e[1]), decay**age, dtype=np.float32)
                for age, e in zip(
                    range(len(self._ml_history) - 1, -1, -1), self._ml_history
                )
            ]
        )
        model = _ml_regressor(self.config)
        model.fit(x, y, sample_weight=weights)
        self._ml_model = model
        self._since_refit = 0
        importances = getattr(model, "feature_importances_", None)
        if importances is not None and len(importances) == len(self.ml_features):
            self.last_feature_importances = pd.Series(
                importances, index=self.ml_features, dtype=float
            )

    # --------------------------------------------------------------- signals
    def _moc_raw(
        self, snapshot: pd.DataFrame, assets: pd.Index
    ) -> tuple[pd.Series, pd.Series]:
        cfg = self.config
        disloc = _col(snapshot, "moc_close_dislocation_5d", assets)
        event = _col(snapshot, "moc_event_dislocation_5d", assets)
        imbalance = _col(snapshot, "moc_imbalance_ratio", assets)
        parts = [_rank_normal(disloc)]
        weights = [1.0]
        if event.abs().sum() > _EPS and cfg.moc_event_weight > 0:
            parts.append(_rank_normal(event))
            weights.append(cfg.moc_event_weight)
        if imbalance.notna().mean() > 0.3 and cfg.moc_imbalance_weight > 0:
            parts.append(_rank_normal(imbalance))
            weights.append(cfg.moc_imbalance_weight)
        pressure = sum(w * p.fillna(0.0) for w, p in zip(weights, parts)) / sum(weights)
        illiquidity = _col(snapshot, "amihud_20d", assets)
        if illiquidity.notna().sum() < cfg.min_names:
            illiquidity = _col(snapshot, "cs_spread_20d", assets)
        illiq_pct = illiquidity.rank(pct=True).fillna(0.5)
        gate = 1.0 + cfg.moc_illiquidity_amplification * (illiq_pct - 0.5)
        raw = -pressure * gate
        mask = disloc.notna()
        return raw, mask

    def _ml_design(
        self,
        returns: pd.DataFrame,
        snapshot: pd.DataFrame,
        market_caps: pd.Series,
        probs: np.ndarray,
    ) -> tuple[pd.DataFrame, pd.Series]:
        assets = returns.columns
        raw: dict[str, pd.Series] = {}
        for column in self.base_features:
            if column == "log_size":
                values = np.log(market_caps.reindex(assets).clip(lower=1.0))
            elif column == "ret_1w":
                values = returns.iloc[-1]
            elif column == "ret_4w":
                values = _compound(returns.iloc[-4:])
            elif column == "ret_13w":
                values = _compound(returns.iloc[-13:])
            else:
                values = _col(snapshot, column, assets)
            # Per-date rank-normal: removes level drift across 30 years and
            # makes the trees learn cross-sectional, not time-series, splits.
            raw[column] = _rank_normal(values).where(values.notna())
        design = pd.DataFrame(raw, index=assets)
        stress = float(probs[-1])
        for column in _REGIME_INTERACTED:
            design[f"{column}_x_stress"] = design[column] * stress
        for s, p in enumerate(probs):
            design[f"regime_p{s}"] = float(p)
        design["moc_event_days_5d"] = _col(snapshot, "moc_event_days_5d", assets).fillna(0.0)
        design = design.reindex(columns=list(self.ml_features)).astype(np.float32)
        micro_cols = [c for c in MICROSTRUCTURE_FEATURE_COLUMNS if c in design]
        eligible = design[micro_cols].notna().sum(axis=1) >= max(3, len(micro_cols) // 3)
        return design, eligible

    def _ml_predict(self, design: pd.DataFrame, eligible: pd.Series) -> tuple[pd.Series, pd.Series]:
        assets = design.index
        if self._ml_model is None or int(eligible.sum()) < self.config.min_names:
            return pd.Series(np.nan, index=assets), pd.Series(False, index=assets)
        names = assets[eligible.to_numpy()]
        prediction = self._ml_model.predict(design.loc[names].to_numpy(dtype=np.float32))
        return pd.Series(prediction, index=names).reindex(assets), eligible

    def _bab_raw(self, observed: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Minus the Frazzini-Pedersen shrunk beta on the universe mean."""

        cfg = self.config
        assets = observed.columns
        window = observed.iloc[-cfg.bab_correlation_weeks :]
        market = window.mean(axis=1, skipna=True)
        count = window.notna().sum()
        # Correlation on the long window, volatilities on the short one.
        corr = window.corrwith(market, drop=False)
        recent = window.iloc[-cfg.bab_volatility_weeks :]
        vol = recent.std(ddof=1)
        vol_count = recent.notna().sum()
        market_vol = float(market.iloc[-cfg.bab_volatility_weeks :].std(ddof=1))
        if not np.isfinite(market_vol) or market_vol <= _EPS:
            return pd.Series(np.nan, index=assets), pd.Series(False, index=assets)
        beta_ts = corr * vol / market_vol
        beta = cfg.bab_shrinkage * beta_ts + (1.0 - cfg.bab_shrinkage) * 1.0
        mask = (
            beta.notna()
            & count.ge(cfg.bab_min_correlation_observations)
            & vol_count.ge(cfg.bab_min_volatility_observations)
            & observed.iloc[-1].notna()
        )
        self.last_bab_beta = beta.where(mask)
        return (-beta).where(mask), mask

    def _regime_components(self, returns: pd.DataFrame, observed: pd.DataFrame, basis: pd.DataFrame):
        assets = returns.columns
        window = returns.iloc[-52:-4] if len(returns) >= 52 else returns.iloc[:-4]
        momentum = _compound(window)
        need = max(1, int(0.9 * len(window)))
        mom_mask = observed.iloc[-len(window):].notna().sum().ge(need) if len(window) else pd.Series(False, index=assets)
        reversal = -observed.iloc[-1]
        rev_mask = observed.iloc[-1].notna()
        weighting = self.config.orthogonalization.weighting
        comps = pd.DataFrame(
            {
                "momentum": orthogonalize(momentum, mom_mask, basis, weighting=weighting,
                                          min_names=self.config.min_names),
                "reversal": orthogonalize(reversal, rev_mask, basis, weighting=weighting,
                                          min_names=self.config.min_names),
            },
            index=assets,
        )
        return comps, mom_mask & rev_mask

    def _regime_weights(self, probs: np.ndarray, market: pd.Series) -> tuple[np.ndarray, dict]:
        cfg = self.config
        priors = np.column_stack(
            [np.asarray(cfg.regime_prior_momentum), np.asarray(cfg.regime_prior_reversal)]
        )
        if priors.shape[0] != len(probs):
            priors = np.tile(priors.mean(axis=0), (len(probs), 1))
        strength = cfg.regime_ic_prior_strength
        posterior = (strength * priors + self._ic_sum) / (strength + self._ic_weight)
        weights = probs @ posterior
        horizon = market.iloc[-cfg.regime_crash_lookback_weeks:]
        bear = float((1.0 + horizon).prod() - 1.0) < 0.0
        crash_guard = bool(bear and probs[-1] > 0.5)
        if crash_guard:
            weights[0] = min(weights[0], 0.0)
        info = {
            "regime_ic_momentum": float(weights[0]),
            "regime_ic_reversal": float(weights[1]),
            "regime_crash_guard": float(crash_guard),
        }
        return weights, info

    # ------------------------------------------------------------------ main
    def build(
        self,
        returns: pd.DataFrame,
        snapshot: pd.DataFrame,
        exposures: pd.DataFrame,
        market_caps: pd.Series,
        observed_returns: pd.DataFrame,
        forward_source: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        cfg = self.config
        assets = returns.columns
        as_of = pd.Timestamp(returns.index[-1])
        observed = observed_returns.reindex(index=returns.index, columns=assets)
        if self._previous is not None and as_of < self._previous["date"]:
            # Rewound to an earlier date (a new backtest pass): start clean.
            self.reset()
        self._learn(forward_source if forward_source is not None else observed_returns, as_of)

        if cfg.microstructure_valid_through is not None and as_of > (
            pd.Timestamp(cfg.microstructure_valid_through) + pd.Timedelta(days=7)
        ):
            snapshot = snapshot.drop(
                columns=list(MICROSTRUCTURE_FEATURE_COLUMNS), errors="ignore"
            )

        orth = cfg.orthogonalization
        beta = shrunk_beta(observed, orth.beta_lookback_weeks, orth.beta_min_observations)
        clusters = self.clusters.labels(observed)
        basis = build_basis(assets, beta, clusters, exposures, include_industry=True)
        style_basis = build_basis(
            assets, beta, clusters, exposures, include_industry=True, include_styles=True
        )
        probs = self.regime.update(observed)
        signals = pd.DataFrame(0.0, index=assets, columns=list(STRUCTURAL_SIGNALS))
        masks = pd.DataFrame(False, index=assets, columns=list(STRUCTURAL_SIGNALS))
        diag: dict[str, float] = {
            "date": as_of.value / 1e9,
            "n_clusters": float(clusters.nunique()),
            **{f"regime_p{s}": float(p) for s, p in enumerate(probs)},
        }

        def finish(name: str, raw: pd.Series, mask: pd.Series, b: pd.DataFrame) -> None:
            clean = orthogonalize(
                raw, mask, b, market_caps=market_caps, weighting=orth.weighting,
                min_names=cfg.min_names,
            )
            live = mask & clean.ne(0.0)
            signals[name] = clean
            masks[name] = live
            before = exposure_diagnostics(_rank_normal(raw.where(mask)), mask, clusters, beta)
            after = exposure_diagnostics(clean, live, clusters, beta)
            diag[f"{name}_coverage"] = float(live.mean())
            diag[f"{name}_cluster_r2_raw"] = before["cluster_r2"]
            diag[f"{name}_cluster_r2_clean"] = after["cluster_r2"]
            diag[f"{name}_beta_corr_raw"] = before["beta_corr"]
            diag[f"{name}_beta_corr_clean"] = after["beta_corr"]

        state: dict = {"date": as_of, "assets": assets, "basis": basis, "regime_probs": probs}

        if MOC_SIGNAL in cfg.signals:
            raw, mask = self._moc_raw(snapshot, assets)
            finish(MOC_SIGNAL, raw, mask, style_basis)

        if ML_SIGNAL in cfg.signals:
            design, eligible = self._ml_design(returns, snapshot, market_caps, probs)
            prediction, ml_mask = self._ml_predict(design, eligible)
            finish(ML_SIGNAL, prediction, ml_mask & prediction.notna(), style_basis)
            state["ml_x"] = design
            state["ml_eligible"] = eligible
            diag["ml_training_cross_sections"] = float(len(self._ml_history))

        if REGIME_SIGNAL in cfg.signals:
            comps, comp_mask = self._regime_components(returns, observed, basis)
            weights, info = self._regime_weights(probs, observed.mean(axis=1).fillna(0.0))
            diag.update(info)
            conviction = float(
                np.clip(np.linalg.norm(weights) / cfg.regime_conviction_reference_ic, 0.0, 1.0)
            )
            diag["regime_conviction"] = conviction
            state["regime_components"] = comps
            if conviction >= cfg.regime_min_conviction:
                raw = comps @ weights
                # Components are already orthogonal; a linear blend stays
                # orthogonal, so only re-standardise (no rank transform that
                # could re-introduce cluster structure).
                clean = orthogonalize(
                    raw, comp_mask, basis, weighting=orth.weighting,
                    rank_transform=False, min_names=cfg.min_names,
                )
                live = comp_mask & clean.ne(0.0)
                signals[REGIME_SIGNAL] = clean * conviction
                masks[REGIME_SIGNAL] = live
                after = exposure_diagnostics(clean, live, clusters, beta)
                diag[f"{REGIME_SIGNAL}_coverage"] = float(live.mean())
                diag[f"{REGIME_SIGNAL}_cluster_r2_clean"] = after["cluster_r2"]
                diag[f"{REGIME_SIGNAL}_beta_corr_clean"] = after["beta_corr"]
            else:
                diag[f"{REGIME_SIGNAL}_coverage"] = 0.0

        if BAB_SIGNAL in cfg.signals:
            raw, mask = self._bab_raw(observed)
            # Beta is the signal, so the nuisance basis keeps clusters and
            # industries but drops the beta column.
            bab_basis = basis.drop(columns=["BETA"], errors="ignore")
            finish(BAB_SIGNAL, raw, mask, bab_basis)
            live = masks[BAB_SIGNAL]
            if int(live.sum()) >= 10:
                diag[f"{BAB_SIGNAL}_ff_beta_corr_clean"] = float(
                    np.corrcoef(signals.loc[live, BAB_SIGNAL],
                                self.last_bab_beta.loc[live])[0, 1]
                )

        self._previous = state
        self.last_diagnostics = diag
        self.diagnostics_history.append(diag)
        coverage = masks.mean(axis=0)
        return signals, coverage, masks
