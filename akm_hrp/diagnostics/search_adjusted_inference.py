"""Search-adjusted performance inference: PSR, DSR-EO, DSR-L, MinTRL.

Implements Bailey & Lopez de Prado (2014) and the exact order-statistic form
from Lopez de Prado & Porcu (2026). Deliberately a separate script from the
backtest: it consumes a saved return series and cannot influence the strategy
configuration that produced it.

Key relations used (per-observation units throughout, never annualized inside
the formulas):

  PSR(SR*) = Phi[ (SRhat - SR*) * sqrt(T-1)
                  / sqrt(1 - g3*SRhat + ((g4-1)/4)*SRhat^2) ]

  DSR-EO   = Phi(t)^K = (1 - p)^K        (Supp. Eq. 21; Sidak-equivalent)
  reject at alpha  <=>  t >= Phi^-1((1-alpha)^(1/K))

  mu_K     = Int x*K*phi(x)*Phi(x)^(K-1) dx           (Supp. Eq. 23)
  sigma_K^2= Int x^2*K*phi(x)*Phi(x)^(K-1) dx - mu_K^2 (Supp. Eq. 24)

The 2026 paper shows the 2014 location-only rule is strongly conservative
(0.05% actual rejection at a nominal 5% when K=1000), so DSR-EO is the headline
and DSR-L is reported for continuity.
"""

import numpy as np
from scipy import integrate, stats

EULER_MASCHERONI = 0.5772156649015329


def sharpe_statistics(returns: np.ndarray) -> dict[str, float]:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    mean, sd = float(r.mean()), float(r.std(ddof=1))
    sharpe = mean / sd if sd > 0 else 0.0
    return {
        "T": float(T),
        "sharpe_per_obs": sharpe,
        "sharpe_annualized": sharpe * np.sqrt(52.0),
        "skew": float(stats.skew(r, bias=False)),
        "kurtosis_raw": float(stats.kurtosis(r, fisher=False, bias=False)),
        "mean": mean,
        "volatility": sd,
    }


def psr_statistic(sharpe: float, T: float, skew: float, kurtosis_raw: float,
                  benchmark: float = 0.0) -> float:
    """The PSR z-statistic: (SRhat - SR*) / SE(SRhat), skew/kurtosis corrected."""
    variance = 1.0 - skew * sharpe + 0.25 * (kurtosis_raw - 1.0) * sharpe**2
    variance = max(variance, 1e-12)
    return (sharpe - benchmark) * np.sqrt(max(T - 1.0, 1.0)) / np.sqrt(variance)


def expected_max_moments(trials: int) -> tuple[float, float]:
    """Exact mean and sd of the max of `trials` iid standard normals."""
    K = int(trials)
    if K <= 1:
        return 0.0, 1.0

    def density(x):
        return K * stats.norm.pdf(x) * stats.norm.cdf(x) ** (K - 1)

    m1, _ = integrate.quad(lambda x: x * density(x), -12, 12, limit=400)
    m2, _ = integrate.quad(lambda x: x * x * density(x), -12, 12, limit=400)
    return float(m1), float(np.sqrt(max(m2 - m1**2, 0.0)))


def expected_max_euler(trials: int) -> float:
    """The 2014 Euler-Mascheroni approximation to E[max] (Eq. 1)."""
    K = max(int(trials), 2)
    return float(
        (1.0 - EULER_MASCHERONI) * stats.norm.ppf(1.0 - 1.0 / K)
        + EULER_MASCHERONI * stats.norm.ppf(1.0 - 1.0 / (K * np.e))
    )


def evaluate(returns: np.ndarray, trial_counts=(1, 4, 10, 50, 200, 1000),
             alpha: float = 0.05) -> dict:
    stat = sharpe_statistics(returns)
    t = psr_statistic(stat["sharpe_per_obs"], stat["T"], stat["skew"],
                      stat["kurtosis_raw"])
    p = float(stats.norm.sf(t))
    standard_error = (stat["sharpe_per_obs"] / t) if t != 0 else np.nan

    rows = []
    for K in trial_counts:
        mu_K, sigma_K = expected_max_moments(K)
        critical_t = float(stats.norm.ppf((1.0 - alpha) ** (1.0 / max(K, 1))))
        dsr_eo = float(stats.norm.cdf(t) ** K)
        # Sharpe (annualized) that would be required to clear this hurdle.
        required_annual = critical_t * standard_error * np.sqrt(52.0)
        # MinTRL: observations needed for the observed SR to clear mu_K.
        edge = stat["sharpe_per_obs"] - mu_K * standard_error
        if edge > 0:
            variance = (1.0 - stat["skew"] * stat["sharpe_per_obs"]
                        + 0.25 * (stat["kurtosis_raw"] - 1.0)
                        * stat["sharpe_per_obs"] ** 2)
            min_trl = 1.0 + variance * (stats.norm.ppf(1 - alpha) / edge) ** 2
        else:
            min_trl = float("inf")
        rows.append({
            "K": K,
            "mu_K": mu_K,
            "sigma_K": sigma_K,
            "mu_K_euler": expected_max_euler(K) if K > 1 else 0.0,
            "critical_t": critical_t,
            "dsr_eo": dsr_eo,
            "significant": bool(dsr_eo >= 1.0 - alpha),
            "required_sharpe_annualized": float(required_annual),
            "min_track_record_weeks": float(min_trl),
        })

    return {
        "statistics": stat,
        "psr_t_statistic": float(t),
        "psr": float(stats.norm.cdf(t)),
        "p_value": p,
        "sharpe_standard_error_per_obs": float(standard_error),
        "trials": rows,
    }


def format_report(name: str, evaluation: dict, alpha: float = 0.05) -> str:
    s = evaluation["statistics"]
    lines = [
        f"--- {name} ---",
        f"  T={int(s['T'])} obs   SR={s['sharpe_annualized']:.4f} annualized "
        f"({s['sharpe_per_obs']:.5f}/wk)   skew={s['skew']:.3f}  "
        f"kurtosis={s['kurtosis_raw']:.3f}",
        f"  PSR(SR*=0) = {evaluation['psr']:.4f}   "
        f"p = {evaluation['p_value']:.4f}   t = {evaluation['psr_t_statistic']:.3f}",
        f"  {'K':>6} {'DSR-EO':>9} {'signif':>7} {'crit t':>8} "
        f"{'req SR ann':>11} {'MinTRL wk':>10}",
    ]
    for row in evaluation["trials"]:
        trl = row["min_track_record_weeks"]
        trl_text = "inf" if not np.isfinite(trl) else f"{trl:,.0f}"
        lines.append(
            f"  {row['K']:>6} {row['dsr_eo']:>9.4f} "
            f"{'YES' if row['significant'] else 'no':>7} "
            f"{row['critical_t']:>8.3f} {row['required_sharpe_annualized']:>11.3f} "
            f"{trl_text:>10}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "kelly_results.json"
    results = json.loads(open(path).read())

    print("Search-adjusted inference (Lopez de Prado & Porcu 2026, DSR-EO)")
    print("=" * 78)
    evaluations = {}
    for name, payload in results.items():
        series = np.array(list(payload["returns"].values()), dtype=float)
        if len(series) < 10:
            print(f"--- {name} --- too few observations ({len(series)})")
            continue
        evaluations[name] = evaluate(series)
        print(format_report(name, evaluations[name]))
        print()

    # Paired difference vs the baseline: the quantity that actually matters,
    # since the arms share a universe, window and cost model.
    if "A_baseline" in results:
        import pandas as pd

        base = pd.Series(results["A_baseline"]["returns"], dtype=float)
        print("Paired difference vs baseline (same weeks, same universe)")
        print("-" * 78)
        for name, payload in results.items():
            if name == "A_baseline":
                continue
            other = pd.Series(payload["returns"], dtype=float)
            common = base.index.intersection(other.index)
            diff = (other.reindex(common) - base.reindex(common)).dropna()
            if len(diff) < 10:
                continue
            ev = evaluate(diff.to_numpy())
            st = ev["statistics"]
            print(f"  {name} - A_baseline: mean diff/wk = {st['mean']:+.6f}, "
                  f"t = {ev['psr_t_statistic']:+.3f}, p = {ev['p_value']:.4f}, "
                  f"n = {int(st['T'])}")
