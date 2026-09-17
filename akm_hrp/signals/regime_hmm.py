"""Causal market-regime detection with a Gaussian hidden Markov model.

Pure numpy (no hmmlearn dependency). The model is fit by Baum-Welch on the
trailing window only, and the current state is read from the *forward
filter* (``P(state_t | x_1..x_t)``), never from smoothed probabilities, so
no future observation leaks into the regime used at formation time.

States are relabelled after every fit in order of increasing market
volatility (0 = calm, last = stress), which keeps regime-conditional
statistics comparable across refits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_EPS = 1e-12


def market_regime_features(
    returns: pd.DataFrame,
    macro: pd.DataFrame | None = None,
    vol_window: int = 4,
    corr_window: int = 13,
) -> pd.DataFrame:
    """Weekly market-state observations from the eligible return panel.

    Columns: market return, log realised market vol, log cross-sectional
    dispersion, average-correlation proxy, and any macro columns supplied
    (aligned as-of, never forward-filled beyond the window end).
    The first column after ``mkt_ret`` is always the volatility feature used
    to order states.
    """

    mkt = returns.mean(axis=1, skipna=True)
    vol = np.log(mkt.rolling(vol_window, min_periods=vol_window).std() + 1e-4)
    dispersion = np.log(returns.std(axis=1, skipna=True) + 1e-4)
    var_mkt = mkt.rolling(corr_window, min_periods=corr_window).var()
    var_single = returns.rolling(corr_window, min_periods=corr_window).var().mean(axis=1)
    avg_corr = (var_mkt / (var_single + _EPS)).clip(1e-3, 1 - 1e-3)
    features = pd.DataFrame(
        {
            "mkt_ret": mkt,
            "mkt_log_vol": vol,
            "log_dispersion": dispersion,
            "avg_corr_logit": np.log(avg_corr / (1.0 - avg_corr)),
        },
        index=returns.index,
    )
    if macro is not None and not macro.empty:
        aligned = macro.sort_index().reindex(returns.index, method="ffill", limit=2)
        features = features.join(aligned.add_prefix("macro_"))
    return features.replace([np.inf, -np.inf], np.nan).dropna()


def _log_gauss(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """(T, K) diagonal-Gaussian log densities."""

    diff = x[:, None, :] - means[None, :, :]
    return -0.5 * (
        np.log(2.0 * np.pi * variances)[None, :, :] + diff**2 / variances[None, :, :]
    ).sum(axis=2)


@dataclass
class GaussianHMM:
    n_states: int = 3
    n_iter: int = 100
    n_init: int = 4
    tol: float = 1e-4
    variance_floor: float = 1e-3
    sticky_prior: float = 5.0
    random_state: int = 0

    def _forward(self, log_b: np.ndarray, start: np.ndarray, trans: np.ndarray):
        t_len, k = log_b.shape
        alpha = np.zeros((t_len, k))
        scale = np.zeros(t_len)
        b_max = log_b.max(axis=1, keepdims=True)
        b = np.exp(log_b - b_max)
        a = start * b[0]
        scale[0] = a.sum() + _EPS
        alpha[0] = a / scale[0]
        for t in range(1, t_len):
            a = (alpha[t - 1] @ trans) * b[t]
            scale[t] = a.sum() + _EPS
            alpha[t] = a / scale[t]
        loglik = float(np.log(scale).sum() + b_max.sum())
        return alpha, scale, b, loglik

    def _fit_once(self, x: np.ndarray, rng: np.random.Generator):
        t_len, d = x.shape
        k = self.n_states
        # k-means++-style init on random rows.
        idx = rng.choice(t_len, size=k, replace=False)
        means = x[idx] + 0.1 * rng.standard_normal((k, d))
        variances = np.tile(np.maximum(x.var(axis=0), self.variance_floor), (k, 1))
        trans = np.full((k, k), 0.1 / max(k - 1, 1))
        np.fill_diagonal(trans, 0.9)
        start = np.full(k, 1.0 / k)
        previous = -np.inf
        loglik = -np.inf
        for _ in range(self.n_iter):
            log_b = _log_gauss(x, means, variances)
            alpha, scale, b, loglik = self._forward(log_b, start, trans)
            beta = np.zeros_like(alpha)
            beta[-1] = 1.0
            for t in range(t_len - 2, -1, -1):
                beta[t] = trans @ (b[t + 1] * beta[t + 1]) / scale[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + _EPS
            xi = (
                alpha[:-1, :, None]
                * trans[None, :, :]
                * (b[1:] * beta[1:])[:, None, :]
                / scale[1:, None, None]
            )
            start = gamma[0] + 1e-3
            start /= start.sum()
            counts = xi.sum(axis=0) + self.sticky_prior * np.eye(k) + 1e-3
            trans = counts / counts.sum(axis=1, keepdims=True)
            weight = gamma.sum(axis=0) + _EPS
            means = (gamma.T @ x) / weight[:, None]
            second = (gamma.T @ (x**2)) / weight[:, None]
            variances = np.maximum(second - means**2, self.variance_floor)
            if abs(loglik - previous) < self.tol * max(1.0, abs(previous)):
                break
            previous = loglik
        return loglik, start, trans, means, variances

    def fit(self, x: np.ndarray) -> "GaussianHMM":
        x = np.asarray(x, dtype=float)
        if len(x) < 5 * self.n_states:
            raise ValueError("Too few observations to fit the HMM.")
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            candidate = self._fit_once(x, rng)
            if best is None or candidate[0] > best[0]:
                best = candidate
        _, start, trans, means, variances = best
        # Order states by the volatility feature (column 1).
        order = np.argsort(means[:, 1]) if x.shape[1] > 1 else np.argsort(means[:, 0])
        self.startprob_ = start[order]
        self.transmat_ = trans[np.ix_(order, order)]
        self.means_ = means[order]
        self.vars_ = variances[order]
        self.loglik_ = best[0]
        return self

    def filter(self, x: np.ndarray) -> np.ndarray:
        """Forward-filtered state probabilities, one row per observation."""

        log_b = _log_gauss(np.asarray(x, dtype=float), self.means_, self.vars_)
        alpha, _, _, _ = self._forward(log_b, self.startprob_, self.transmat_)
        return alpha


@dataclass(frozen=True)
class RegimeConfig:
    n_states: int = 3
    refit_every_calls: int = 13
    min_observations: int = 104
    fit_window_weeks: int = 520


class RegimeTracker:
    """Refits the HMM periodically and returns today's filtered regime."""

    def __init__(self, config: RegimeConfig | None = None, macro: pd.DataFrame | None = None):
        self.config = config or RegimeConfig()
        self.macro = macro
        self.reset()

    def reset(self) -> None:
        self.model: GaussianHMM | None = None
        self._center: pd.Series | None = None
        self._scale: pd.Series | None = None
        self._columns: list[str] = []
        self._calls = 0
        self._history: list[pd.DataFrame] = []
        self.last_probabilities = self.neutral()
        self.last_features: pd.Series | None = None

    def neutral(self) -> np.ndarray:
        return np.full(self.config.n_states, 1.0 / self.config.n_states)

    def _observations(self, returns: pd.DataFrame) -> pd.DataFrame:
        feats = market_regime_features(returns, self.macro)
        # The allocator window is finite (e.g. 260 weeks). Keep an
        # accumulating store of past observations so refits can use a longer
        # history than one window -- each row was computed from data
        # available at its own date, so nothing leaks.
        if self._history:
            stored = self._history[-1]
            feats = pd.concat([stored, feats[~feats.index.isin(stored.index)]])
        feats = feats.sort_index().iloc[-self.config.fit_window_weeks :]
        self._history = [feats]
        return feats

    def update(self, returns: pd.DataFrame) -> np.ndarray:
        if self._history and pd.Timestamp(returns.index[-1]) < self._history[-1].index[-1]:
            # Rewound (new backtest pass): start clean.
            self.reset()
        feats = self._observations(returns)
        if len(feats) < self.config.min_observations:
            self.last_probabilities = self.neutral()
            return self.last_probabilities
        refit = (
            self.model is None
            or self._calls >= self.config.refit_every_calls
            or list(feats.columns) != self._columns
        )
        if refit:
            self._columns = list(feats.columns)
            self._center = feats.median()
            self._scale = (feats - self._center).abs().median() * 1.4826 + 1e-6
            x = ((feats - self._center) / self._scale).clip(-6, 6).to_numpy()
            try:
                self.model = GaussianHMM(n_states=self.config.n_states).fit(x)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                self.model = None
                self.last_probabilities = self.neutral()
                return self.last_probabilities
            self._calls = 0
        self._calls += 1
        x = ((feats - self._center) / self._scale).clip(-6, 6).to_numpy()
        probs = self.model.filter(x)[-1]
        if not np.isfinite(probs).all():
            probs = self.neutral()
        self.last_probabilities = probs / probs.sum()
        self.last_features = feats.iloc[-1]
        return self.last_probabilities
