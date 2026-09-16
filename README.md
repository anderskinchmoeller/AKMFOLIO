Factor risk decomposition + DCC-style residual correlation (retail_alpha_mpc.py) — this is a scaled-down Barra
Bayesian IC-shrinkage for signal weighting, decaying realized ICs toward priors rather than trusting raw backtest Sharpe — exactly how real multi-signal books size conviction
Walk-forward-validated ML with weight scaled by out-of-sample IC (hrp_alpha_v2.py) — most retail quant code skips this and just overfits in-sample
Explicit execution-cost modeling (spread + temporary + permanent impact) baked into the optimizer, not bolted on after
Regime-adaptive clustering (ra_hrp_v2_allocator.py's stability-spike gating) — this is a genuinely sophisticated idea most public HRP implementations don't have


 

# AKM HRP

## UNIVERSE

Du skal started med at vœlge dit universe af assetes i dowload_tick.py. 

## Factor model 

Små ting der kan give en lillle lillle fordel !

kort... reel mulighed for alpha, da vi lettere kan reallokere i stedet for at sœlge og derfor betale skal eller betale mere i omkostninger hvis vi vil købe og sœlge for at udligne.
Kort kan give høj negativ korreleret asset. 
Men det er ikke gratis. og alle fees skal huskes at medregnes.
kunne give en kant ved at short høje beta aktier (mindre rente) og lange lave beta.


inspo [text](artikler/ssrn_id1020543_code623849.pdf)
anomly inspo [text](<../Downloads/houxuezhang2020rfs (1).pdf>)
factor inspo [text](artikler/ssrn-4695086.pdf)
[text](artikler/1-s2.0-S2405844020303613-main.pdf)
[text](artikler/ssrn-2319861.pdf)
[text](artikler/w20984.pdf)
[text](../Downloads/ssrn-2499205.pdf)  delta small large eu caps
tilt towards f-13filings largest buys rentec, jane street, kengriffin, millium, de shaw, agq, bridgewater. 

Hvis vi skal generere alpha skal det komme fra factor modellen. Der er kendte faktorer der giver lille alpha, så vi kan altid lave et fall back på sådan en. (momentum/low beta).

Vi mister også hurtigt det hele igen, hvis vi introducerer for meget støj i hvores covariance matrix. Derfor vil vi gerne holde factor modellen relavtiv simple og lave statistisk test på error estimation af more matrix. 
Få indsigt i problemet her [text](../Downloads/1-s2.0-S2405918826000048-main.pdf)
                            [text](../Downloads/2507.01918v3.pdf)
## VI BRUGER Advanced AR-HRP (RMT + OLO) 

Vil kigge i kroge hvor de store funde har sin begrœnsninger.  (ingen gearing, stor position i mag7 for at nå benchmark, ingen mikro caps under 300M$ og at jeg ikke automatisk skal købe aktier der bliver tilføjet indexe, kan måske gøre det inden, hvis jeg får en god ide. 

vil også lave en vurdering af de vorskellige aktiers valuta og om den er god eller dårlig, baseret på carry trades og derefter tilt lidt i en retning.

vil også undersøge kort om det bliver mening at gå lang mikro caps og kort bluechip og hvilken risiko det skaber. 

stat arb skal vœre en del at modellen, hvis jeg ikke selv kan finde på noget finder jeg en aqr fund.

btc lang minus guld. lidt risky, men måske en ide. (måske ikke lige guld, men en asset der bliver udvandet kontinuerligt ligsom guld, som er modsat btc)

vi bruger ra-hrp model til at finde vœgte 

vil vi ender med lav beta og ønsker at opjusterer gør vi det gennem billige gearet spy future pga. fractuanally kelly crit. 



Få disse data ned i en fil der hedder weekly_returns.csv og har det rigtige format. GL.

Påstår ikke at kunne slå market (s&p) men med en hvis sandsynlighed slå det på et risk/return basis. Derfor hvis du vil slå skal du på en måde geare ved at få fat i billige lån i DK som er den edge som skal udnyttes. (krœver man har en vis startskapital)



Få et godt overblik over alpha: (close must read)
[text](artikler/w28432.appendix.pdf)

Lœs alt research fra DE Shaw og AQR og (bridge water)
de eneste top funds der ikke er helt stille.

## AR_HRP

Vores model baserer sig hovedsagligt på HRP (SHUR-HRP)
[text](artikler/ssrn-53706246.pdf)
[text](artikler/2606.12612v1.pdf)
[text](../Downloads/ssrn-4623991.pdf)
rene beviser kan man bare tage for gode vare. 

## MACHINE LEARNING

Når vi konstruerer en model skal vi altid teste i hoved og røv. F.eks. hvis jeg ville implementerer en ny ting ville jeg nok opstille nogle H_0 hypoteser, så man tilføjer en parameter på et stistics grundlag og hver kan man tilføjer, tilføjer man også støj til Covariancen. Så må man renge ud om det giver et højerer forventet afkast end hvad støjen fjerner af forventet afkast igen inden for et statistisk signifikant niveau (og det bliver meget uprœcist fordi det er notorisk svœrt at estimere afkast ), så der skulle nok vœre generelle højere krav for at acceptere sådan en parameter. 

## OG NU DEN UMULIGE

UNDGå AL BAIS og OVERFITTING. Jeg tœnker, at hver kan jeg kører min model og retter noget, laver jeg selv en overfitting fordi jeg optimerer nogle parametre for at få et bedre resultat. Optimalt skrev vi opdellen og kørte den en gang for at se at den virkede. Du kan lade vœr med at udskrive noget fra din model, eller som alle andre køre out of sample tests over tilfœldige intervaller som du ikke kender når du er fœrdig.
Derudover skal den stresstestes under devires konstrueret (monte carlo) eller tidligere markedets crasheds og 

## BACKTEST
[text](artikler/ssrn-2460551.pdf)

brug bootstrap og monte carlo
[text](../Downloads/ssrn-7333578.pdf)




Og her er den generelle struktur på vores model:

1.set contraints:
    Både max assets og max/min weights. (min(min.weights))=100/max.Nassets

2. Covariance matrix:
implement    (maybe Classic Barra-style with EWMA factor covariance, structural shrinkage on specific risk)

            Implement predictive specific-risk model (highest leverage) 
        WMA + Shrinkage + Residual DCC-GARCH:Factor Covariance ($\Omega_{f,t}$): Modeled using exponentially weighted moving averages (EWMA) blended with Ledoit-Wolf shrinkage to ensure positive semi-definiteness and numerical stability.Residual Covariance ($\Sigma_{\epsilon,t}$): Idiosyncratic returns are stripped of systematic factor exposure via WLS/OLS. The remaining residual correlation matrix is modeled using a Multi-Asset DCC process to capture time-varying, dynamic cross-asset residual tail risk.Predictive Specific Risk ($\hat{\sigma}_{i, \epsilon, t+1}^2$):Uses a multi-horizon autoregressive predictive structure (Daily, Weekly, Monthly components ala HAR-GARCH) to forecast future idiosyncratic variance per asset, incorporating dynamic volatility clustering.Multi-Scale Factor Volatility Dynamics:Adjusts factor returns dynamically across short-term ($5$d), medium-term ($21$d), and long-term ($63$d) lookback windows to capture regime shifts and multi-horizon volatility spillover. with these factor signals

            . factor model:
   
   1. robust fundamental investment signal (good carry)

    2. quality-decomposed skewness signal (good vs. bad carry filter)

    3. small-minus-big cap

    4. winsorize → rank→normal → neutralize → combine → grinold → rank-ic)5

    5. higher-moment tilted multi-factor allocation
        instead of simple equal-weighting across factors, apply the vsk tilting optimization from boudt et al. (2020) to ar-hrp.

     # --- 3. Optimizer -------------------------------------------------------
    print(f"\n[3] Optimizing (multi-period, {LOOKAHEAD_PERIODS}-period lookahead, "
          f"CVaR + tax-aware + vol-scaled turnover cost)...")
    current_weights = pd.Series(1.0 / len(prices.columns), index=prices.columns)  # e.g. starting equal-weight
    tax_rate_per_name = pd.Series(RNG.choice([0.0, 0.37, 0.15], size=len(prices.columns),
                                              p=[0.3, 0.4, 0.3]), index=prices.columns)
    # fraction of each existing position's value that's an unrealized gain
    # (0 = no gain/at a loss, e.g. a recently-opened lot; closer to 1 = a
    # long-held position that's appreciated a lot) -- this is what the tax
    # cost term below actually scales by, not the tax rate alone
    unrealized_gain_frac = pd.Series(RNG.uniform(0, 0.75, len(prices.columns)), index=prices.columns)
 
    w_final, planned_path = multiperiod_optimize(
        alpha_paths, full_cov, current_weights, tax_rate_per_name, unrealized_gain_frac,
        position_cap=0.12, cvar_limit=0.06, discount=DISCOUNT_FACTOR,
    )
 
    # HRP kept only as a diversification benchmark to compare against, not
    # as the live allocator -- see the module docstring above §3.
   w_hrp = hrp_weights(full_cov)


    7. implement transaction ost Temporary + Permanent impact

    8. implement tail risk with Parametric CVaR only

    Clean Rank-Normalization: Cross-sectional percentile ranking mapped onto a standard Gaussian distribution $N(0, 1)$ with tail winsorization to remove outliers while preserving monotonic relative ordering.Multi-Factor Neutralization: Weighted Least Squares (WLS) regression against risk exposures (e.g., Sector, Market Beta, Size) to extract pure residual signal streams.Dynamic IC Estimation: Rolling Information Coefficient (IC) tracking using rank correlation (Spearman) adjusted for signal persistence and volatility.Grinold Fundamental Law Alpha Combination: Combining factors using dynamic IC weights scaled by residual volatility ($\sigma_\epsilon$) via Grinold's Fundamental Law of Active Management:$$\alpha_i = \text{IC}_i \times \text{Score}_{i, \text{neut}} \times \sigma_{\epsilon, i}$$




flowchart TD
    A["Compustat PIT by GVKEY"] --> C["CCM historical link"]
    B["IBES history by IBES ticker"] --> D["IBES–CRSP historical link"]
    C --> E["Weekly PERMNO feature store"]
    D --> E
    E --> F["Alpha forecasts"]
    F --> G["HRP active sleeve"]
    H["CRSP PIT universe"] --> G

(.venv) ~/hrp/akm_hrp/data
» cp ~/Portfolio/point_in_time_universe_audit.csv ./point_in_time_universe_audit.csv

(.venv) ~/hrp/akm_hrp/data
» python generate_pit_mask.py
PIT mask saved to ~/hrp/akm_hrp/data/pit_universe.csv

Factor risk decomposition + DCC-style residual correlation (retail_alpha_mpc.py) — this is a scaled-down Barra
Bayesian IC-shrinkage for signal weighting, decaying realized ICs toward priors rather than trusting raw backtest Sharpe — exactly how real multi-signal books size conviction
Walk-forward-validated ML with weight scaled by out-of-sample IC (hrp_alpha_v2.py) — most retail quant code skips this and just overfits in-sample
Explicit execution-cost modeling (spread + temporary + permanent impact) baked into the optimizer, not bolted on after
Regime-adaptive clustering (ra_hrp_v2_allocator.py's stability-spike gating) — this is a genuinely sophisticated idea most public HRP implementations don't have


# AKM HRP

## UNIVERSE

Du skal started med at vœlge dit universe af assetes i dowload_tick.py. 

## Factor model 

Små ting der kan give en lillle lillle fordel !

kort... reel mulighed for alpha, da vi lettere kan reallokere i stedet for at sœlge og derfor betale skal eller betale mere i omkostninger hvis vi vil købe og sœlge for at udligne.
Kort kan give høj negativ korreleret asset. 
Men det er ikke gratis. og alle fees skal huskes at medregnes.
kunne give en kant ved at short høje beta aktier (mindre rente) og lange lave beta.


inspo [text](artikler/ssrn_id1020543_code623849.pdf)
anomly inspo [text](<../Downloads/houxuezhang2020rfs (1).pdf>)
factor inspo [text](artikler/ssrn-4695086.pdf)
[text](artikler/1-s2.0-S2405844020303613-main.pdf)
[text](artikler/ssrn-2319861.pdf)
[text](artikler/w20984.pdf)
[text](../Downloads/ssrn-2499205.pdf)  delta small large eu caps
tilt towards f-13filings largest buys rentec, jane street, kengriffin, millium, de shaw, agq, bridgewater. 

Hvis vi skal generere alpha skal det komme fra factor modellen. Der er kendte faktorer der giver lille alpha, så vi kan altid lave et fall back på sådan en. (momentum/low beta).

Vi mister også hurtigt det hele igen, hvis vi introducerer for meget støj i hvores covariance matrix. Derfor vil vi gerne holde factor modellen relavtiv simple og lave statistisk test på error estimation af more matrix. 
Få indsigt i problemet her [text](../Downloads/1-s2.0-S2405918826000048-main.pdf)
                            [text](../Downloads/2507.01918v3.pdf)
## VI BRUGER Advanced AR-HRP (RMT + OLO) 

Vil kigge i kroge hvor de store funde har sin begrœnsninger.  (ingen gearing, stor position i mag7 for at nå benchmark, ingen mikro caps under 300M$ og at jeg ikke automatisk skal købe aktier der bliver tilføjet indexe, kan måske gøre det inden, hvis jeg får en god ide. 

vil også lave en vurdering af de vorskellige aktiers valuta og om den er god eller dårlig, baseret på carry trades og derefter tilt lidt i en retning.

vil også undersøge kort om det bliver mening at gå lang mikro caps og kort bluechip og hvilken risiko det skaber. 

stat arb skal vœre en del at modellen, hvis jeg ikke selv kan finde på noget finder jeg en aqr fund.

btc lang minus guld. lidt risky, men måske en ide. (måske ikke lige guld, men en asset der bliver udvandet kontinuerligt ligsom guld, som er modsat btc)

vi bruger ra-hrp model til at finde vœgte 

vil vi ender med lav beta og ønsker at opjusterer gør vi det gennem billige gearet spy future pga. fractuanally kelly crit. 



Få disse data ned i en fil der hedder weekly_returns.csv og har det rigtige format. GL.

Påstår ikke at kunne slå market (s&p) men med en hvis sandsynlighed slå det på et risk/return basis. Derfor hvis du vil slå skal du på en måde geare ved at få fat i billige lån i DK som er den edge som skal udnyttes. (krœver man har en vis startskapital)



Research and portfolio-allocation framework built around Hierarchical Risk Parity (HRP), adaptive covariance estimation, alpha overlays, point-in-time universes, combinatorial purged cross-validation (CPCV), Rustuna hyperparameter tuning, diagnostics, and portfolio weight exports.

> **Package namespace:** the active package is `akm_hrp`.  
> Older project notes may refer to `institutional_hrp`; those names are obsolete.

## What the project does

The current pipeline combines:

- point-in-time universe filtering;
- rolling covariance estimation;
- adaptive Ledoit-Wolf / EWMA / PCA covariance blending;
- downside covariance scenarios and regret-aware consensus HRP;
- optional Numba acceleration around covariance kernels and block bootstrap generation;
- ensemble HRP trees;
- cluster-stability gating;
- Bayesian shrinkage of combined alpha scores;
- signal-budget and covariance-inverse overlays;
- portfolio bounds and rebalance controls;
- CPCV-based Rustuna tuning;
- walk-forward reporting;
- target-weight and execution exports;
- a multi-panel research dashboard.

The project can build `weekly_returns.csv` and a matching PIT universe from
WRDS CRSP CIZ. Existing CSV inputs remain supported.

## Main workflow

Run commands from the project root, the directory containing `akm_hrp/`:

```bash
cd ~/hrp
source .venv/bin/activate
```

### 0. Build CRSP inputs from WRDS (optional)

Configure WRDS authentication locally, then run:

```bash
python -m akm_hrp.cli.build_wrds_dataset \
  --start 1990-01-01 \
  --end 2026-08-21 \
  --output-dir wrds_data \
  --chunk-months 12 \
  --top-n 1500
```

The command uses CRSP CIZ `DlyRet`, compounds it to weekly returns, keys
securities by permanent `PERMNO`, and creates a date-effective eligibility
mask. CIZ delisting returns are already included and are not multiplied again.
The connector auto-discovers `crsp` first and falls back to `crsp_a_stock`,
matching the schemas available in this WRDS account. For normalized current
CIZ tables, it joins `stkSecurityInfoHist` by its effective-date interval so
the common-stock classification is point-in-time.

Downloads are cached in restartable, Friday-aligned chunks. At each month-end,
eligible stocks are ranked by `DlyCap`; the top-N selection becomes effective
one month later. This lag prevents formation-month information from entering
the same month's eligibility mask. Use `--no-resume` to refresh cached chunks
or `--top-n 0` to deliberately disable market-cap ranking.

### Build a cross-exposure HRP universe from CRSP ETFs

The stock-only WRDS bundle excludes ETFs by design. To reproduce a liquid
multi-asset ETF template from CRSP CIZ instead of Yahoo data, run:

```bash
python -m akm_hrp.cli.build_crsp_hrp_universe \
  --start 2018-01-01 \
  --end 2025-12-26 \
  --output-dir wrds_hrp_etf \
  --minimum-median-dollar-volume 1000000
```

The default template contains SPY, QQQ, IWM, EFA, EEM, SHY, IEF, TLT, TIP,
LQD, HYG, GLD, DBC, VNQ, and UUP. The WRDS query admits only point-in-time CRSP
`FUND` securities classified as `ETF` or `ETV`, then exports weekly PERMNO
returns, a PIT liquidity mask, the economic cluster labels, complete aligned
returns, raw Spearman correlation, a Ledoit-Wolf-denoised rank correlation,
and its angular-distance matrix.

Use `--universe-file my_universe.csv` for a custom CSV containing `ticker` and
optional `cluster` and `role` columns. These are exchange-traded wrappers for
cross-asset exposures; they are not the same as joining CRSP's separate cash
Treasury, index, or real-estate databases.

### Build a balanced CRSP stock + Treasury + factor universe

For a universe based on individual common stocks rather than ETF wrappers,
build the point-in-time sector history and the CRSP Treasury sleeves first:

```bash
python -m akm_hrp.cli.build_crsp_sector_history \
  --start 1990-01-01 \
  --end 2025-12-26 \
  --output wrds_full_clean/crsp_sector_history.csv.gz

python -m akm_hrp.cli.build_crsp_treasury_sleeves \
  --start 1990-01-01 \
  --end 2025-12-26 \
  --output wrds_full_clean/crsp_treasury_weekly_returns.csv

python -m akm_hrp.cli.build_structural_alpha_features \
  --daily-cache-dir wrds_full_clean/daily_cache \
  --start 1990-01-01 \
  --end 2025-12-26 \
  --output wrds_full_clean/structural_alpha_features.csv.gz
```

The Treasury command requests CRSP fixed-term index `TREASNOX` values 2000003,
2000005, 2000007, and 2000009 (1Y, 5Y, 10Y, and 30Y). It uses the licensed
`TFZ_DLY_FT` table, converts `TDRETADJ` percentages to decimal returns, and
geometrically compounds daily returns to Friday weeks. If the WRDS account has
no CRSP Treasury entitlement, omit `--macro-returns` in the final command.

To add a standard release-aware value factor, rebuild the Compustat features
with lagged CRSP market capitalization. This creates `book_to_market`; it does
not substitute book-to-assets for value:

If the raw extract lacks `dvy`, `prstkcy`, and `sstky`, the command still
builds `book_to_market`; it only leaves the optional shareholder-carry columns
empty and prints a warning.

```bash
python build_compustat_pit_features.py \
  --fundamentals wrds_full_clean/compustat_quarterly_raw.csv \
  --returns wrds_full_clean/weekly_returns.csv \
  --pit wrds_full_clean/pit_universe_delisting_safe.csv \
  --crsp-cache-dir wrds_full_clean/daily_cache \
  --output wrds_full_clean/compustat_pit_features_long.csv.gz
```

Then build the mixed universe:

```bash
python -m akm_hrp.cli.build_crsp_balanced_universe \
  --returns wrds_full_clean/weekly_returns.csv \
  --base-pit wrds_full_clean/pit_universe_delisting_safe.csv \
  --structural-features wrds_full_clean/structural_alpha_features.csv.gz \
  --sector-history wrds_full_clean/crsp_sector_history.csv.gz \
  --value-features wrds_full_clean/compustat_pit_features_long.csv.gz \
  --macro-returns wrds_full_clean/crsp_treasury_weekly_returns.csv \
  --stocks-per-sector 5 \
  --minimum-size-percentile 0.80 \
  --minimum-price 5 \
  --minimum-median-dollar-volume 1000000 \
  --output-dir wrds_balanced_hrp
```

Every formation month uses date-effective UES/ICB/SIC classifications,
selects at most five liquid stocks per sector from the top market-cap quintile,
requires at least six represented sectors, and activates the selection one
month later. The base PIT mask is intersected afterward, retaining its
common-stock, history, suspension, and delisting protections.

`crsp_style_factor_returns.csv` contains causal small-minus-big, high-minus-low
book-to-market (when supplied), 12-1 momentum, and low-minus-high volatility
returns. They are zero-investment explanatory factors for Mapper/factor NCO,
not cash assets. Use them like this:

```bash
python -m akm_hrp.cli.compare_models \
  --returns wrds_balanced_hrp/weekly_returns.csv \
  --pit wrds_balanced_hrp/pit_universe.csv \
  --factor-returns wrds_balanced_hrp/crsp_style_factor_returns.csv \
  --models mapper_factor_nco ra_hrp_v2 equal_weight \
  --weights-output outputs/balanced_current_weights.csv \
  --weights-png outputs/balanced_current_weights.png \
  --dashboard-pdf outputs/dashboard/balanced_hrp_dashboard.pdf
```

Only use `--include-factor-proxies` on the universe-builder command when the
long and short legs can actually be financed and traded; it is deliberately
off by default.

#### Run the same build for `wrds_top2500`

The repository includes a runnable wrapper:

```bash
./build_wrds_top2500_balanced.sh
```

Its expanded commands are:

```bash
python -m akm_hrp.cli.build_crsp_sector_history \
  --start 1990-01-01 \
  --end 2025-12-26 \
  --output wrds_top2500/crsp_sector_history.csv.gz

python -m akm_hrp.cli.build_crsp_treasury_sleeves \
  --start 1990-01-01 \
  --end 2025-12-26 \
  --output wrds_top2500/crsp_treasury_weekly_returns.csv

python -m akm_hrp.cli.build_structural_alpha_features \
  --daily-cache-dir wrds_full_clean/daily_cache \
  --start 1990-01-01 \
  --end 2025-12-26 \
  --output wrds_top2500/structural_alpha_features.csv.gz

python build_compustat_pit_features.py \
  --fundamentals wrds_full_clean/compustat_quarterly_raw.csv \
  --returns wrds_top2500/weekly_returns.csv \
  --pit wrds_top2500/pit_universe.csv \
  --crsp-cache-dir wrds_full_clean/daily_cache \
  --output wrds_top2500/compustat_pit_features_long.csv.gz

python -m akm_hrp.cli.build_crsp_balanced_universe \
  --returns wrds_top2500/weekly_returns.csv \
  --base-pit wrds_top2500/pit_universe.csv \
  --structural-features wrds_top2500/structural_alpha_features.csv.gz \
  --sector-history wrds_top2500/crsp_sector_history.csv.gz \
  --value-features wrds_top2500/compustat_pit_features_long.csv.gz \
  --asset-metadata wrds_top2500/crsp_security_metadata.csv \
  --macro-returns wrds_top2500/crsp_treasury_weekly_returns.csv \
  --output-dir wrds_top2500/balanced_hrp
```

The CRSP daily cache and raw Compustat extract are shared inputs, so this build
reuses them from `wrds_full_clean` rather than copying approximately 785 MB.
The returns, PIT mask, features, metadata, and final balanced bundle are all
specific to `wrds_top2500`. Because that bundle currently has no separate
`pit_universe_delisting_safe.csv`, the wrapper uses its native
`pit_universe.csv`.

### 1. Smoke-test the research pipeline

```bash
python -m akm_hrp.cli.run_research \
  --returns weekly_returns.csv \
  --pit akm_hrp/data/pit_universe.csv \
  --as-of 2026-08-21 \
  --storage rustuna_studies.db \
  --trials 1 \
  --baseline-trials 1 \
  --jobs 1
```

### 2. Run full tuning

```bash
python -m akm_hrp.cli.run_research \
  --returns weekly_returns.csv \
  --pit akm_hrp/data/pit_universe.csv \
  --storage rustuna_studies.db \
  --trials 40 \
  --baseline-trials 30 \
  --jobs 4
```

### 3. Generate the final report and target weights

```bash
python generate_outputs.py \
  --returns weekly_returns.csv \
  --pit akm_hrp/data/pit_universe.csv \
  --storage rustuna_studies.db
```

### Compare the allocator with hard baselines

```bash
python -m akm_hrp.cli.compare_models \
  --returns wrds_data/weekly_returns.csv \
  --pit wrds_data/pit_universe.csv \
  --as-of 2026-08-21 \
  --tc-bps 10 \
  --models ra_hrp_v2 \
  --progress-every-rebalances 1 \
  --weights-output outputs/ra_hrp_current_weights.csv \
  --weights-png outputs/ra_hrp_current_weights.png
```

This writes a net-of-cost comparison of equal weight, inverse volatility,
regularized minimum variance, Return-Adjusted HRP (`ra_hrp`), robust consensus
Return-Adjusted HRP (`ra_hrp_v2`), Mapper-conditioned factor NCO
(`mapper_factor_nco`), retail-oriented ensemble HRP with momentum/trend
(`hrp_alpha_v1`), capacity-aware systematic/ML and structural alpha
(`hrp_alpha_v2`), legacy ensemble HRP, the regret-aware core, and the bounded-
overlay model. Use `--models` to
run a subset and `--data-start` to define a shorter warm-up sample for focused
comparisons.

### Run Retail Alpha MPC

`retail_alpha_mpc` starts with the balanced PIT universe and admits up to 30
liquid candidates from the broad top-2500 file. It combines 12-1 momentum,
post-earnings drift, quality/value/carry, liquidity-conditioned reversal, and
a smaller-but-tradeable retail-capacity sleeve. Signal weights update from
strictly subsequent rank ICs. A HAR-style specific-risk forecast and a
three-period receding-horizon optimizer jointly account for risk, spread,
temporary impact, permanent impact, participation, sector, style, CVaR, and
position constraints.

```bash
python -m akm_hrp.cli.compare_models \
  --returns wrds_top2500/weekly_returns.csv \
  --pit wrds_top2500/pit_universe.csv \
  --models retail_alpha_mpc equal_weight \
  --data-start 1996-01-01 \
  --evaluation-start 2001-01-01 \
  --retail-mpc-horizon 3 \
  --retail-max-added-assets 30 \
  --dynamic-portfolio-value 1000000 \
  --progress-every-rebalances 1 \
  --output outputs/retail_alpha_mpc/model_comparison.csv \
  --diagnostics-output outputs/retail_alpha_mpc/diagnostics.csv \
  --weights-output outputs/retail_alpha_mpc/current_weights.csv \
  --weights-png outputs/retail_alpha_mpc/current_weights.png \
  --dashboard-pdf outputs/retail_alpha_mpc/dashboard.pdf \
  --dashboard-png outputs/retail_alpha_mpc/dashboard.png \
  --dashboard-focus-model retail_alpha_mpc
```

The model uses `balanced_hrp/pit_universe.csv`, the structural and Compustat
feature files, and `crsp_sector_history.csv.gz` beside the broad returns file
unless the corresponding `--dynamic-*` paths are supplied. Its nonlinear
execution-cost estimate is deducted in the walk-forward result and shown in
the dashboard; `--tc-bps` remains the fallback for the comparison models.
Balanced-only columns are loaded automatically from
`balanced_hrp/weekly_returns.csv`; override that location with
`--dynamic-balanced-returns` when using a custom bundle layout.

### Run Retail Edge MPC

`retail_edge_mpc` builds on Retail Alpha MPC but restricts learned alpha to a
two-sided capacity niche: a position must be implementable for the configured
retail portfolio while a minimally meaningful position for a multi-billion-
dollar fund would require excessive trading time or ownership. It combines
five predeclared PIT signals: factor-residual 12-1 momentum, neglected earnings
drift, patient quality/value, conservative investment, and short-term liquidity
provision only when the return dislocation clears a round-trip spread hurdle.
A stricter entry score than hold score implements a buy/hold spread to reduce
costly churn.

`retail_edge_ml_mpc` adds a deliberately small nonlinear reliability layer to
that model. Two causal, exponentially weighted ridge learners use fixed signal
interactions; an ML score is admitted only when the fast and slow learners
agree on its sign. The ML signal has a zero IC prior, a 12-rebalance warm-up,
and is automatically turned off by the existing online IC gate when it fails
to add out-of-sample information. It does not fit weights or tune architecture
on the backtest.

The single predeclared 2019–2025 top-2500 evaluation produced a 0.820 Sharpe,
15.43% CAGR, 20.03% volatility, and -36.29% maximum drawdown, modestly better
than the frozen `retail_edge_mpc` result (0.811 Sharpe and 15.31% CAGR). This is
not statistically established alpha: with 51 declared research trials, the
Deflated Sharpe probability was 37.2% (p=0.628). At the final formation date,
the ML signal's causal rank IC was +0.0081 but its 18.35% coverage remained
below the 20% activation gate, so the model assigned it zero current weight.

```bash
python -m akm_hrp.cli.compare_models \
  --returns wrds_top2500/weekly_returns.csv \
  --pit wrds_top2500/pit_universe.csv \
  --models retail_edge_mpc equal_weight \
  --data-start 2017-01-01 \
  --evaluation-start 2019-01-01 \
  --rebalance-every-weeks 4 \
  --deflated-sharpe-trials 50 \
  --significance-benchmark equal_weight \
  --output outputs/retail_edge/model_comparison.csv \
  --diagnostics-output outputs/retail_edge/current_diagnostics.csv \
  --weights-output outputs/retail_edge/current_weights.csv \
  --weights-png outputs/retail_edge/current_weights.png \
  --dashboard-pdf outputs/retail_edge/dashboard.pdf \
  --dashboard-focus-model retail_edge_mpc
```

`--deflated-sharpe-trials` must include every model or parameter variant that
was searched, including discarded attempts. Output includes Probabilistic
Sharpe, Deflated Sharpe probability and p-value, its selection-bias benchmark,
and Newey-West HAC active-return statistics versus
`--significance-benchmark`. A raw Sharpe improvement is not evidence of alpha
when these diagnostics fail. The 2019-2025 top-2500 run completed in this
project improved Sharpe from 0.645 to 0.811 and CAGR from 13.00% to 15.31%, but
did **not** establish significant alpha: 50-trial Deflated Sharpe probability
was 36.7% and equal-weight active-alpha HAC p-value was 0.671.

The capacity conditioning and cost controls follow the implementability logic
in Novy-Marx and Velikov (2016), the selection adjustment follows Bailey and
López de Prado (2014), and the direct net-of-cost portfolio framing follows
Jensen, Kelly, Malamud, and Pedersen (2026). These references motivate the
design; they do not guarantee future alpha.

When `--weights-output` is supplied, the command also writes the latest
post-close target weights using all data through `--as-of`. The export is in
tidy form with ticker symbols in `asset`, the original CRSP identifier in
`permno`, and the allocation in `weight`. The ticker lookup is automatically
loaded from `crsp_security_metadata.csv` beside the returns file. Use
`--asset-metadata` to provide a different lookup. `--weights-png` writes a
readable chart of the largest positions and aggregates the remainder.

### Run Mapper Factor NCO

This strategy estimates expected returns and covariance from a low-rank factor
model, identifies the current state with a causal Mapper graph, then applies
hierarchical quasi-diagonalization and two-level Nested Clustered Optimization.
It learns PCA factors automatically when `--factor-returns` is omitted.

```bash
python -m akm_hrp.cli.compare_models \
  --returns wrds_full_clean/weekly_returns.csv \
  --pit wrds_full_clean/pit_universe_top500.csv \
  --models mapper_factor_nco \
  --progress-every-rebalances 5 \
  --output outputs/mapper_factor_nco_metrics.csv \
  --diagnostics-output outputs/mapper_factor_nco_diagnostics.csv \
  --weights-output outputs/mapper_factor_nco_weights.csv \
  --weights-png outputs/mapper_factor_nco_weights.png
```

An optional external factor CSV must have a date first column and one weekly,
decimal-return factor per remaining column. Add
`--factor-returns data/weekly_factor_returns.csv`. See
[Mapper Factor NCO](docs/mapper-factor-nco.md) for the design, data contract,
diagnostics, research references, and limitations.

### Run HRP Alpha v1

`hrp_alpha_v1` implements a simpler retail-oriented candidate: blended
long/short Ledoit–Wolf covariance, average/complete/Ward HRP tree bagging,
smooth 6/12-month momentum, a soft negative-trend penalty, bounded weights, and
a per-position no-trade band.

```bash
python -m akm_hrp.cli.compare_models \
  --returns data/multi_asset_weekly_returns.csv \
  --models hrp_alpha_v1 \
  --hrp-alpha-max-weight 0.20 \
  --hrp-alpha-no-trade-band 0.02 \
  --output outputs/hrp_alpha_v1_metrics.csv \
  --diagnostics-output outputs/hrp_alpha_v1_diagnostics.csv \
  --weights-output outputs/hrp_alpha_v1_weights.csv \
  --weights-png outputs/hrp_alpha_v1_weights.png
```

Optional unlevered volatility targeting requires an actual cash/T-bill return
column. For a column named `BIL`, add `--hrp-alpha-target-vol 0.10` and
`--hrp-alpha-cash-asset BIL`. See
[HRP Alpha v1](docs/hrp-alpha-v1.md) for the exact formulas and limitations.

### Run HRP Alpha v2

V2 adds purged chronological ridge forecasts, weekly reversal/low-volatility
microstructure proxies, lagged CRSP size/liquidity features, release-aware
fundamentals, and portfolio-value/ADV capacity caps. ML is automatically given
zero weight when its chronological validation IC is non-positive.

```bash
python -m akm_hrp.cli.build_structural_alpha_features \
  --daily-cache-dir wrds_full_clean/daily_cache \
  --start 1990-01-01 \
  --end 2025-12-26 \
  --output wrds_full_clean/structural_alpha_features.csv.gz

python -m akm_hrp.cli.compare_models \
  --returns wrds_full_clean/weekly_returns.csv \
  --pit wrds_full_clean/pit_universe_top500.csv \
  --models hrp_alpha_v2 \
  --hrp-alpha-portfolio-value 100000 \
  --hrp-alpha-max-adv-participation 0.01 \
  --hrp-alpha-structural-features \
    wrds_full_clean/structural_alpha_features.csv.gz \
    wrds_full_clean/compustat_pit_features_long.csv.gz \
  --diagnostics-output outputs/hrp_alpha_v2_diagnostics.csv \
  --weights-output outputs/hrp_alpha_v2_weights.csv \
  --weights-png outputs/hrp_alpha_v2_weights.png \
  --dashboard-pdf outputs/dashboard/hrp_alpha_v2_dashboard.pdf \
  --dashboard-png outputs/dashboard/hrp_alpha_v2_dashboard.png \
  --dashboard-focus-model hrp_alpha_v2
```

Merger, CEF, warrant, and option sleeves activate only when real point-in-time
event fields are supplied. See [HRP Alpha v2](docs/hrp-alpha-v2.md).

The dashboard is calculated from the same walk-forward results as the metrics.
In addition to the requested full-period files, export writes a file for each
calendar year present in the focus model after `--evaluation-start` filtering,
including partial years. For example, `dashboard.pdf` also produces
`dashboard_2024.pdf`, `dashboard_2025.pdf`, etc.; PNG follows the same convention.
For dashboards during a run, add `--dashboard-every-year`. Each completed
calendar year is exported immediately, and the final partial year is exported
at the end. Run comparison models before the focus model to include them in
these live exports. The full weekly retail-alpha run is:

```bash
bash /Users/anderskinch/AKMFOLIO/run_full_scale_retail_alpha_ml_mpc.sh
```

The launcher uses the installed Portfolio Python environment, evaluates from
1996-01-05, and saves a timestamped run directory with live terminal output,
`run.log`, annual PDF/PNG dashboards, and final full-period results.

Both full-scale retail-alpha launchers enable
`--retail-alpha-ml-allow-exposure-limit-relaxation`. If the sector/style caps
jointly exclude a fully invested portfolio, the allocator finds the minimum
common additive increase across those caps and all planning steps. For example,
an increase of 0.01 changes a 25% sector cap to 26% and a 0.25 absolute style
limit to 0.26. Each event is logged with the formation date and recorded as
`exposure_limit_relaxation` in diagnostics; the increase resets on each rebalance.
Full investment, per-name bounds, trading capacity, and forced exits remain hard
constraints. CVaR follows its separate existing relaxation flag. Direct CLI and
Python usage remain strict unless exposure relaxation is explicitly enabled.
This changes the portfolio policy, so retain the relaxation diagnostics alongside
backtest results. It does not restore state from a failed run.

Annual panels contain only that year's observations, so a multi-year rolling
Sharpe window has no values in an annual dashboard.
It compares every model named by `--models` in the equity, drawdown, and rolling
Sharpe panels; transaction costs and the ticker-labelled weight heatmap use
`--dashboard-focus-model`. For a benchmark comparison, run `--models
hrp_alpha_v2 ra_hrp_v2 equal_weight` and keep `hrp_alpha_v2` as the focus.

### Run Frontier Alpha HRP on the full sample

`frontier_alpha_hrp.py` keeps the robust consensus HRP portfolio as the core
and adds a small, tracking-error-capped active sleeve. Price, hierarchy, and
release-aware fundamental signals are evaluated as 1-, 4-, and 13-week
experts. Their weights learn only from returns observed after publication and
each expert is charged for its own turnover before it earns influence.

```bash
python frontier_alpha_hrp.py \
  --returns wrds_full_clean/weekly_returns.csv \
  --pit wrds_full_clean/pit_universe_top500.csv \
  --fundamental-features wrds_full_clean/compustat_pit_features_long.csv.gz \
  --ticker-metadata wrds_full_clean/crsp_security_metadata.csv \
  --data-start 1990-01-01 \
  --evaluation-start 1992-01-03 \
  --as-of 2025-12-26 \
  --research-trials 32 \
  --bootstrap-samples 2000 \
  --progress-every-rebalances 5 \
  --output-dir outputs/frontier_alpha_hrp_full_corrected
```

The output directory contains net return and turnover histories, all absolute
and active statistical-test tables, DSR/HAC/bootstrap/Reality Check/SPA/PBO
gates, risk metrics, diagnostics, current ticker-labelled weights, and four
PNG charts. A failed gate is a research failure, even when a headline return
or Sharpe ratio looks attractive. The supplied Compustat store is
release-date-aware; unless its upstream vendor snapshot is vintage-safe, it
must still be treated as potentially exposed to later restatements.

## Tests and data audit

Run the regression suite from the project root:

```bash
python -m pytest -q
```

To classify missing returns for securities that could still be held, run:

```bash
python audit_missing_held_returns.py
```

The audit reads the cleaned WRDS files by default and writes
`wrds_full_clean/missing_held_return_audit.csv`. It exits with status 0 when
the report is empty and status 1 when reviewable gaps are found. Use
`--returns`, `--pit`, and `--output` to override those paths.

### 4. Generate the research dashboard

```bash
python export_dashboard.py \
  --returns weekly_returns.csv \
  --pit akm_hrp/data/pit_universe.csv \
  --storage rustuna_studies.db \
  --tc-bps 10 \
  --rolling-sharpe-years 3 \
  --heatmap-assets 15
```

## Documentation

The full documentation lives under `docs/` and can be served with MkDocs:

```bash
mkdocs serve
```

Then open the local URL printed by MkDocs.

Start with:

- [Installation](docs/installation.md)
- [Workflow](docs/workflow.md)
- [Data contracts](docs/data.md)
- [Model architecture](docs/model.md)
- [Configuration](docs/configuration.md)
- [Research and tuning](docs/research.md)
- [Reports and exports](docs/outputs.md)
- [CLI reference](docs/cli.md)
- [Module/API map](docs/api.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Reproducibility](docs/reproducibility.md)
- [Known limitations](docs/known-limitations.md)

## Expected project layout

```text
hrp/
├── README.md
├── mkdocs.yml
├── requirements.txt
├── weekly_returns.csv
├── rustuna_studies.db
├── generate_outputs.py
├── export_dashboard.py
├── outputs/
└── akm_hrp/
    ├── __init__.py
    ├── config.py
    ├── data/
    │   ├── io.py
    │   ├── returns.py
    │   └── pit_universe.py
    ├── cov/
    │   ├── ensemble.py
    │   ├── ewma.py
    │   ├── pca.py
    │   ├── synthetic_factors.py
    │   ├── bootstrap.py
    │   └── jit_kernels.py
    ├── signals/
    │   ├── alpha_stack.py
    │   └── shrinkage.py
    ├── hrp/
    │   ├── trees.py
    │   ├── allocation.py
    │   └── stability.py
    ├── overlay/
    │   ├── overlay.py
    │   ├── signal_budget.py
    │   ├── cov_inverse.py
    │   └── bounds.py
    ├── allocators/
    │   └── hrp_overlay_allocator.py
    ├── backtest/
    │   ├── cpcv.py
    │   └── engine.py
    ├── tuning/
    │   └── rustuna_integration.py
    ├── diagnostics/
    │   ├── metrics.py
    │   ├── plots.py
    │   └── reports.py
    ├── export/
    │   └── weight_exporter.py
    └── cli/
        └── run_research.py
```

## Important interpretation note

This is research software. The documentation distinguishes between:

1. **implemented behavior**;
2. **intended research design**; and
3. **known limitations that should be reviewed before treating results as production-grade out-of-sample evidence**.

See [Known limitations](docs/known-limitations.md).
