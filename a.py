import pathlib

path = pathlib.Path("akm_hrp/cli/build_compustat_pit_features.py")
content = path.read_text()

# --- Fix 1: BuildConfig gains market_cap_scale --------------------------
old_config = """@dataclass(frozen=True)
class BuildConfig:
    rdq_fallback_days: int = 90
    base_information_lag_days: int = 1
    extra_information_lag_days: int = 0
    maximum_staleness_days: int = 150
    minimum_ttm_quarters: int = 4
    sue_history_quarters: int = 8
    sue_minimum_quarters: int = 4
    output_buffer_assets: int = 100"""
new_config = """@dataclass(frozen=True)
class BuildConfig:
    rdq_fallback_days: int = 90
    base_information_lag_days: int = 1
    extra_information_lag_days: int = 0
    maximum_staleness_days: int = 150
    minimum_ttm_quarters: int = 4
    sue_history_quarters: int = 8
    sue_minimum_quarters: int = 4
    output_buffer_assets: int = 100
    # Raw CRSP DlyCap-style market cap inputs are in thousands of dollars;
    # matches build_structural_alpha_features.py's --market-cap-scale
    # default so the two builders agree on what "market cap" means.
    market_cap_scale: float = 1000.0"""
assert content.count(old_config) == 1, "BuildConfig anchor not found or not unique"
content = content.replace(old_config, new_config)

# --- Fix 2: align_to_weekly_universe gains an optional market_caps arg --
old_sig = """def align_to_weekly_universe(
    quarterly: pd.DataFrame,
    weekly_index: pd.DatetimeIndex,
    pit: pd.DataFrame,
    destination: Path,
    config: BuildConfig,
) -> dict[str, object]:"""
new_sig = """def align_to_weekly_universe(
    quarterly: pd.DataFrame,
    weekly_index: pd.DatetimeIndex,
    pit: pd.DataFrame,
    destination: Path,
    config: BuildConfig,
    market_caps: pd.DataFrame | None = None,
) -> dict[str, object]:"""
assert content.count(old_sig) == 1, "align_to_weekly_universe signature anchor not found or not unique"
content = content.replace(old_sig, new_sig)

# --- Fix 3: compute book_to_market per asset when market_caps is given --
old_block = """        selected = [
            "formation_date", "permno", "gvkey", "datadate", "available_date",
            "age_days", "rdq_fallback", *FEATURE_COLUMNS,
        ]
        aligned = aligned[selected]"""
new_block = """        if market_caps is not None and asset in market_caps.columns:
            # book_to_market mixes Compustat book equity (ceqq, $ millions)
            # with a lagged CRSP market cap observed as of each formation
            # date -- i.e. the market cap known/PIT-eligible at the time,
            # not a same-quarter Compustat-only proxy. market_caps carries
            # raw CRSP-style units (thousands), same convention as
            # build_structural_alpha_features.py's market_cap_usd.
            raw_cap = aligned["formation_date"].map(market_caps[asset])
            market_cap_usd = raw_cap.astype(float) * config.market_cap_scale
            book_equity_usd = aligned["ceqq"].astype(float) * 1_000_000.0
            aligned["book_to_market"] = _safe_divide(book_equity_usd, market_cap_usd)
        else:
            aligned["book_to_market"] = np.nan

        selected = [
            "formation_date", "permno", "gvkey", "datadate", "available_date",
            "age_days", "rdq_fallback", "book_to_market", *FEATURE_COLUMNS,
        ]
        aligned = aligned[selected]"""
assert content.count(old_block) == 1, "selected-columns anchor not found or not unique"
content = content.replace(old_block, new_block)

path.write_text(content)
print("All 3 fixes applied successfully.")
