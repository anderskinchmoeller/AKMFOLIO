"""Daily-bar microstructure and closing-auction features (CRSP CIZ OHLC).

There is no intraday tape here, so every quantity is a documented daily-bar
proxy:

VWAP
    Typical price ``(high + low + close) / 3``. A true VWAP needs TAQ; the
    typical price is the standard daily-bar stand-in and keeps the sign of the
    close-vs-session dislocation.
Close dislocation
    ``ln(close / typical)`` scaled by Parkinson (high-low) volatility, so a
    one-tick move in a quiet name and a large move in a volatile name are
    comparable.
Expected MOC (market-on-close) imbalance
    No imbalance feed is available, so expected passive closing demand is
    proxied by (a) a calendar of mechanical index-rebalance closes and
    (b) abnormal volume on the day. If you later obtain NYSE/Nasdaq closing
    imbalance data, add a weekly ``moc_imbalance_ratio`` column
    (imbalance shares / ADV, signed) to the feature file; the signal uses it
    in place of the proxy automatically.
Order-flow toxicity
    Bulk-volume classification (Easley, Lopez de Prado & O'Hara 2012) on the
    open-to-close move gives a buy/sell split; VPIN proxy = volume-weighted
    mean |order imbalance|. Kyle's lambda from a rolling regression of
    returns on BVC-signed dollar volume. Corwin-Schultz (2012) high-low
    spread and, where CRSP supplies them, closing quoted spreads.

Timing: a row stamped Friday ``t`` uses daily data through the close of
``t`` and is **not** shifted by default. That is the same information set as
the week-``t`` return the allocator already sees at formation, so trades are
assumed to execute no earlier than the next session. Use ``lag_weeks=1`` for
the more conservative convention of ``build_structural_alpha_features``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

OHLC_COLUMNS = ("open", "high", "low", "close")

MICROSTRUCTURE_FEATURE_COLUMNS = (
    "moc_close_dislocation_5d",
    "moc_close_dislocation_1d",
    "moc_event_dislocation_5d",
    "moc_abnormal_volume_5d",
    "moc_event_days_5d",
    "intraday_return_5d",
    "overnight_return_5d",
    "bvc_order_imbalance_5d",
    "vpin_proxy_20d",
    "kyle_lambda_60d",
    "cs_spread_20d",
    "quoted_spread_20d",
    "parkinson_vol_20d",
    "volume_shock_5_60",
    "trade_count_shock_5d",
)


# ---------------------------------------------------------------------------
# Passive-flow calendar
# ---------------------------------------------------------------------------


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (weekday - first.weekday()) % 7
    return first + pd.Timedelta(days=offset + 7 * (n - 1))


@dataclass(frozen=True)
class PassiveCalendarConfig:
    # S&P 500 / Nasdaq-100 quarterly rebalances and quad witching:
    # third Friday of Mar/Jun/Sep/Dec, effective after the close.
    quarterly_weight: float = 1.0
    # Russell US reconstitution: fourth Friday of June (approximation of the
    # historical rule), plus the second Friday of December from 2026 when
    # FTSE Russell moved to semi-annual reconstitution (11 Dec 2026).
    russell_weight: float = 1.5
    russell_semiannual_from_year: int = 2026
    # MSCI quarterly/semi-annual index reviews: last business day of
    # Feb/May/Aug/Nov.
    msci_weight: float = 1.0
    # Month-end benchmark and pension rebalancing flow (weaker).
    month_end_weight: float = 0.5


def passive_event_intensity(
    trading_dates: pd.DatetimeIndex,
    config: PassiveCalendarConfig | None = None,
) -> pd.Series:
    """Expected mechanical closing-auction pressure per trading date (>= 0).

    Nominal event dates falling on a holiday map to the previous trading day.
    """

    cfg = config or PassiveCalendarConfig()
    dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).normalize().unique().sort_values()
    intensity = pd.Series(0.0, index=dates)
    if len(dates) == 0:
        return intensity

    def snap(target: pd.Timestamp) -> pd.Timestamp | None:
        # Never map a future nominal date onto the last available day.
        if target > dates[-1]:
            return None
        pos = int(dates.searchsorted(target, side="right")) - 1
        if pos < 0:
            return None
        found = dates[pos]
        # Do not snap across more than a week (data gap, not a holiday).
        return found if (target - found).days <= 6 else None

    def add(target: pd.Timestamp, weight: float) -> None:
        day = snap(target)
        if day is not None and weight > 0:
            intensity.loc[day] += weight

    for year in range(dates[0].year, dates[-1].year + 1):
        for month in (3, 6, 9, 12):
            add(_nth_weekday(year, month, 4, 3), cfg.quarterly_weight)
        add(_nth_weekday(year, 6, 4, 4), cfg.russell_weight)
        if year >= cfg.russell_semiannual_from_year:
            add(_nth_weekday(year, 12, 4, 2), cfg.russell_weight)
        for month in (2, 5, 8, 11):
            add(pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0),
                cfg.msci_weight)
        if cfg.month_end_weight > 0:
            for month in range(1, 13):
                add(
                    pd.Timestamp(year=year, month=month, day=1) + pd.offsets.BMonthEnd(0),
                    cfg.month_end_weight,
                )
    return intensity


# ---------------------------------------------------------------------------
# Daily -> weekly features
# ---------------------------------------------------------------------------


def _rolling(grouped, column: str, window: int, how: str, min_periods: int | None = None):
    mp = min_periods if min_periods is not None else max(2, window // 2)
    roll = grouped[column].rolling(window, min_periods=mp)
    out = getattr(roll, how)()
    return out.reset_index(level=0, drop=True)


def daily_microstructure(
    daily: pd.DataFrame,
    calendar: PassiveCalendarConfig | None = None,
) -> pd.DataFrame:
    """Per-security daily microstructure quantities and rolling aggregates.

    ``daily`` needs permno, date, ret, volume and open/high/low/close (price
    is used when close is absent). bid, ask, num_trades are optional.
    """

    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["permno"] = frame["permno"].astype(str)
    if "close" not in frame and "price" in frame:
        frame["close"] = frame["price"]
    missing = [c for c in ("ret", "volume", *OHLC_COLUMNS) if c not in frame]
    if missing:
        raise ValueError(
            f"Daily data is missing {missing}. Re-pull CRSP with OHLC fields "
            "(akm_hrp.cli.build_microstructure_features --download)."
        )
    for column in ("ret", "volume", "open", "high", "low", "close", "bid", "ask", "num_trades"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # CRSP marks no-trade days with a negative (bid/ask midpoint) price.
    for column in OHLC_COLUMNS:
        frame[column] = frame[column].where(frame[column] > 0)
    frame = frame.drop_duplicates(["permno", "date"], keep="last")
    frame = frame.sort_values(["permno", "date"]).reset_index(drop=True)

    o, h, l, c = (frame[k] for k in OHLC_COLUMNS)
    valid_bar = (h >= l) & (h >= c) & (l <= c) & o.notna() & c.notna()
    volume = frame["volume"].clip(lower=0.0)
    frame["dollar_volume"] = volume * c

    typical = (h + l + c) / 3.0
    frame["disloc"] = np.log(c / typical).where(valid_bar)
    frame["park_var"] = (np.log(h / l) ** 2 / (4.0 * np.log(2.0))).where(valid_bar)
    frame["intraday"] = np.log(c / o).where(valid_bar)
    gross = 1.0 + frame["ret"]
    frame["overnight"] = (np.log(gross.where(gross > 0)) - frame["intraday"])

    g = frame.groupby("permno", sort=False)
    park_vol = np.sqrt(_rolling(g, "park_var", 20, "mean", 5))
    frame["parkinson_vol_20d"] = park_vol
    # Scale by *yesterday's* volatility estimate so the scaler is not
    # contaminated by today's bar.
    park_prev = park_vol.groupby(frame["permno"]).shift(1)
    frame["disloc_z"] = (frame["disloc"] / park_prev.where(park_prev > 1e-6)).clip(-5, 5)

    frame["vol_med_20"] = (
        _rolling(g, "volume", 20, "median", 5).groupby(frame["permno"]).shift(1)
    )
    abn = (volume / frame["vol_med_20"].where(frame["vol_med_20"] > 0)).clip(0.0, 20.0)
    frame["log_abn_vol"] = np.log1p(abn)

    intensity = passive_event_intensity(pd.DatetimeIndex(frame["date"].unique()), calendar)
    frame["event"] = frame["date"].map(intensity).fillna(0.0)
    weight = abn.fillna(1.0) * (1.0 + frame["event"])
    frame["w"] = weight.where(frame["disloc_z"].notna(), 0.0)
    frame["w_disloc"] = (frame["w"] * frame["disloc_z"]).fillna(0.0)
    frame["event_disloc"] = (
        frame["event"] * abn.fillna(1.0) * frame["disloc_z"]
    ).fillna(0.0)

    # Bulk-volume classification on the open-to-close move.
    sigma_intraday = (
        _rolling(g, "intraday", 60, "std", 20).groupby(frame["permno"]).shift(1)
    )
    z = frame["intraday"] / sigma_intraday.where(sigma_intraday > 1e-6)
    frame["oi"] = 2.0 * norm.cdf(z.clip(-8, 8)) - 1.0
    frame.loc[z.isna(), "oi"] = np.nan
    frame["oi_v"] = (frame["oi"] * volume).fillna(0.0)
    frame["abs_oi_v"] = (frame["oi"].abs() * volume).fillna(0.0)
    frame["v_oi"] = volume.where(frame["oi"].notna(), 0.0).fillna(0.0)
    frame["signed_dv"] = frame["oi"] * frame["dollar_volume"] / 1e6

    # Rolling Kyle lambda: cov(ret, signed $vol) / var(signed $vol).
    frame["x"] = frame["signed_dv"]
    frame["y"] = frame["ret"].where(frame["x"].notna())
    frame["xy"] = frame["x"] * frame["y"]
    frame["xx"] = frame["x"] ** 2
    g = frame.groupby("permno", sort=False)
    mx = _rolling(g, "x", 60, "mean", 20)
    my = _rolling(g, "y", 60, "mean", 20)
    mxy = _rolling(g, "xy", 60, "mean", 20)
    mxx = _rolling(g, "xx", 60, "mean", 20)
    var_x = mxx - mx**2
    frame["kyle_lambda_60d"] = ((mxy - mx * my) / var_x.where(var_x > 1e-12)).clip(-1, 1)

    # Corwin-Schultz two-day high-low spread estimator.
    prev_h = g["high"].shift(1)
    prev_l = g["low"].shift(1)
    beta_cs = np.log(h / l) ** 2 + np.log(prev_h / prev_l) ** 2
    gamma_cs = np.log(np.maximum(h, prev_h) / np.minimum(l, prev_l)) ** 2
    k = 3.0 - 2.0 * np.sqrt(2.0)
    alpha_cs = (np.sqrt(2.0 * beta_cs) - np.sqrt(beta_cs)) / k - np.sqrt(gamma_cs / k)
    spread = 2.0 * (np.exp(alpha_cs) - 1.0) / (1.0 + np.exp(alpha_cs))
    frame["cs"] = spread.clip(lower=0.0).where(valid_bar)
    if {"bid", "ask"}.issubset(frame.columns):
        bid, ask = frame["bid"], frame["ask"]
        ok = (bid > 0) & (ask >= bid)
        frame["qs"] = (2.0 * (ask - bid) / (ask + bid)).where(ok)
    else:
        frame["qs"] = np.nan
    if "num_trades" in frame:
        frame["ntrd"] = np.log1p(frame["num_trades"].clip(lower=0.0))
    else:
        frame["ntrd"] = np.nan

    g = frame.groupby("permno", sort=False)
    out = frame[["permno", "date", "parkinson_vol_20d", "kyle_lambda_60d"]].copy()
    sum_w = _rolling(g, "w", 5, "sum", 1)
    out["moc_close_dislocation_5d"] = _rolling(g, "w_disloc", 5, "sum", 1) / sum_w.where(sum_w > 0)
    out["moc_close_dislocation_1d"] = frame["disloc_z"]
    out["moc_event_dislocation_5d"] = _rolling(g, "event_disloc", 5, "sum", 1)
    out["moc_abnormal_volume_5d"] = _rolling(g, "log_abn_vol", 5, "mean", 2)
    out["moc_event_days_5d"] = _rolling(g, "event", 5, "sum", 1)
    out["intraday_return_5d"] = _rolling(g, "intraday", 5, "sum", 3)
    out["overnight_return_5d"] = _rolling(g, "overnight", 5, "sum", 3)
    sv5 = _rolling(g, "v_oi", 5, "sum", 1)
    out["bvc_order_imbalance_5d"] = _rolling(g, "oi_v", 5, "sum", 1) / sv5.where(sv5 > 0)
    sv20 = _rolling(g, "v_oi", 20, "sum", 5)
    out["vpin_proxy_20d"] = _rolling(g, "abs_oi_v", 20, "sum", 5) / sv20.where(sv20 > 0)
    out["cs_spread_20d"] = _rolling(g, "cs", 20, "mean", 5)
    out["quoted_spread_20d"] = _rolling(g, "qs", 20, "mean", 5)
    v5 = _rolling(g, "volume", 5, "mean", 3)
    v60 = _rolling(g, "volume", 60, "mean", 20)
    out["volume_shock_5_60"] = np.log((v5 + 1.0) / (v60 + 1.0))
    n5 = _rolling(g, "ntrd", 5, "mean", 3)
    n60 = _rolling(g, "ntrd", 60, "mean", 20)
    out["trade_count_shock_5d"] = n5 - n60
    return out


def weekly_microstructure_features(
    daily: pd.DataFrame,
    *,
    calendar: PassiveCalendarConfig | None = None,
    lag_weeks: int = 0,
    week_frequency: str = "W-FRI",
) -> pd.DataFrame:
    """Last trading day of each week -> one row per (formation_date, asset)."""

    rolled = daily_microstructure(daily, calendar)
    rolled["formation_date"] = (
        rolled["date"].dt.to_period(week_frequency).dt.end_time.dt.normalize()
    )
    weekly = rolled.sort_values(["permno", "date"]).groupby(
        ["formation_date", "permno"], as_index=False
    ).tail(1)
    columns = list(MICROSTRUCTURE_FEATURE_COLUMNS)
    if lag_weeks:
        weekly[columns] = weekly.groupby("permno")[columns].shift(int(lag_weeks))
    weekly = weekly.rename(columns={"permno": "asset"})
    weekly = weekly[["formation_date", "asset", *columns]]
    weekly[columns] = weekly[columns].replace([np.inf, -np.inf], np.nan).astype("float32")
    return weekly.sort_values(["formation_date", "asset"]).reset_index(drop=True)


# Trailing trading days a chunked builder must carry into the next chunk so
# every rolling window above (60-day sigma feeding a 60-day Kyle lambda)
# is complete at the chunk boundary.
REQUIRED_TRAILING_DAYS = 130
