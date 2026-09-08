import pandas as pd
import numpy as np
from .io import load_returns_wide

def clean_returns(df: pd.DataFrame, min_history_weeks: int) -> pd.DataFrame:
    df = df.copy().astype(float).sort_index()
    for col in df.columns:
        non_zero_mask = df[col].fillna(0.0) != 0.0
        if not non_zero_mask.any():
            df[col] = np.nan
            continue
        first_valid = non_zero_mask.idxmax()
        df.loc[df.index < first_valid, col] = np.nan

    all_nan = df.columns[df.isna().all()].tolist()
    if all_nan:
        df = df.drop(columns=all_nan)

    history = df.notna().sum()
    insufficient = history[history < min_history_weeks].index.tolist()
    if insufficient:
        df = df.drop(columns=insufficient)

    return df

def load_and_clean_returns(path: str, min_history_weeks: int) -> pd.DataFrame:
    df = load_returns_wide(path)
    return clean_returns(df, min_history_weeks)

