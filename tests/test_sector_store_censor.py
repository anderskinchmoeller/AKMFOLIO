"""_SectorStore carries the last sector past the extract's censoring date."""

import pandas as pd

from akm_hrp.allocators.dynamic_barra_alpha import _SectorStore


def _history():
    return pd.DataFrame(
        {
            "permno": [1, 2, 2, 3],
            "sec_info_start": ["2000-01-01", "2000-01-01", "2010-01-01", "2000-01-01"],
            "sec_info_end": ["2025-12-31", "2009-12-31", "2025-12-31", "2015-06-30"],
            "sector": ["TECH", "FINL", "HEALTH", "ENRG"],
        }
    )


def test_censored_interval_carries_forward():
    store = _SectorStore(_history())
    out = store.lookup(pd.Timestamp("2026-05-15"), pd.Index(["1", "2", "3"]))
    assert out.to_dict() == {"1": "TECH", "2": "HEALTH", "3": "UNKNOWN"}


def test_dates_inside_extract_unchanged():
    store = _SectorStore(_history())
    assets = pd.Index(["1", "2", "3"])
    assert store.lookup(pd.Timestamp("2005-01-07"), assets).to_dict() == {
        "1": "TECH", "2": "FINL", "3": "ENRG"
    }
    assert store.lookup(pd.Timestamp("2020-01-03"), assets)["3"] == "UNKNOWN"


def test_real_interior_end_still_closes():
    store = _SectorStore(_history())
    assert store.lookup(pd.Timestamp("2009-12-31"), pd.Index(["2"]))["2"] == "FINL"
    assert store.lookup(pd.Timestamp("2010-01-01"), pd.Index(["2"]))["2"] == "HEALTH"
