
import hashlib

import pandas as pd


def load_pit_universe_mask(path: str, dates: pd.Index, assets: list[str]) -> pd.DataFrame:
    """
    Load PIT universe mask from CSV and align it to the returns index/columns.
    Ensures the PIT mask is a proper DataFrame with a datetime index.
    """

    # Load CSV
    pit = pd.read_csv(path)

    # --- VALIDATION: must contain a date column ---
    date_col = None
    for col in pit.columns:
        if col.lower() in ("date", "timestamp", "time", "week", "period"):
            date_col = col
            break

    if date_col is None:
        raise ValueError(
            f"PIT file '{path}' does not contain a date column. "
            f"Did you accidentally pass the audit file instead of the PIT mask?"
        )

    # Parse dates
    pit[date_col] = pd.to_datetime(pit[date_col], errors="raise")
    pit = pit.set_index(date_col)

    # --- VALIDATION: ensure no duplicate dates ---
    if pit.index.has_duplicates:
        raise ValueError(
            f"PIT file '{path}' contains duplicate date rows. "
            f"Cannot reindex safely."
        )

    # CSV headers are strings, while callers may supply numeric identifiers.
    # Validate before reindexing: reindex would otherwise turn a completely
    # mismatched universe into an all-zero mask and silently skip every trade.
    pit.columns = pit.columns.astype(str)
    requested_assets = pd.Index([str(asset) for asset in assets])
    overlap = pit.columns.intersection(requested_assets)
    minimum_overlap = min(2, len(requested_assets))
    if len(overlap) < minimum_overlap:
        pit_examples = pit.columns[:3].tolist()
        return_examples = requested_assets[:3].tolist()
        raise ValueError(
            f"PIT file '{path}' has only {len(overlap)} asset columns in common "
            f"with the returns data; at least {minimum_overlap} are required. "
            f"PIT examples: {pit_examples}; returns examples: {return_examples}. "
            "Use a PIT universe built for the same identifier scheme."
        )

    # --- ALIGN TO RETURNS ---
    pit = pit.reindex(index=dates, columns=requested_assets).fillna(0).astype(int)

    return pit


def pit_fingerprint(pit_mask: pd.DataFrame) -> str:
    """
    Compute a SHA256 fingerprint of the PIT mask.
    Ensures the PIT mask is a DataFrame, not a string or malformed object.
    """

    if not isinstance(pit_mask, pd.DataFrame):
        raise TypeError(
            "pit_mask must be a pandas DataFrame. "
            "You likely passed the audit file instead of the PIT mask."
        )

    h = hashlib.sha256()
    h.update("\n".join(map(str, pit_mask.index)).encode("utf-8"))
    h.update("\n".join(map(str, pit_mask.columns)).encode("utf-8"))
    h.update(pit_mask.to_numpy().tobytes())
    return h.hexdigest()
