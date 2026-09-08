from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CPCVConfig:
    n_groups: int = 6
    test_groups: int = 2
    group_weeks: int = 52
    purge_weeks: int = 1
    embargo_weeks: int = 1


def _validate_config(cfg: CPCVConfig) -> None:
    if cfg.n_groups < 2:
        raise ValueError("n_groups must be >= 2.")

    if not 1 <= cfg.test_groups < cfg.n_groups:
        raise ValueError(
            "test_groups must satisfy 1 <= test_groups < n_groups."
        )

    if cfg.group_weeks <= 0:
        raise ValueError("group_weeks must be positive.")

    if cfg.purge_weeks < 0:
        raise ValueError("purge_weeks must be non-negative.")

    if cfg.embargo_weeks < 0:
        raise ValueError("embargo_weeks must be non-negative.")


def build_cpcv_splits(
    index: pd.Index,
    cfg: CPCVConfig,
) -> list[dict[str, np.ndarray]]:
    """
    Build combinatorial purged cross-validation splits using vectorized
    NumPy masks.

    The CPCV region is the most recent:

        n_groups * group_weeks

    observations. It is partitioned into n_groups contiguous groups of
    exactly group_weeks observations. Every combination of test_groups
    groups is used as one CPCV path.

    Purging:
        remove purge_weeks observations immediately BEFORE each test block
        from the training set.

    Embargo:
        remove embargo_weeks observations immediately AFTER each test block
        from the training set.

    Returns
    -------
    list[dict]
        Each item contains flat integer arrays:

            {
                "train_idx": np.ndarray[int],
                "test_idx": np.ndarray[int],
            }

        These selectors are directly consumable by engine.py and avoid
        pandas-Series masks entirely.
    """
    _validate_config(cfg)

    n_obs = len(index)
    required = cfg.n_groups * cfg.group_weeks

    if n_obs < required:
        raise ValueError(
            f"CPCV requires at least {required} observations "
            f"({cfg.n_groups} groups x {cfg.group_weeks} weeks), "
            f"but received {n_obs}."
        )

    # Match the fixed-size CPCV design: use the most recent complete region.
    offset = n_obs - required
    local_n = required

    positions = np.arange(local_n, dtype=np.int64)
    group_ids = positions // cfg.group_weeks

    # Shape: (n_splits, test_groups)
    combos = np.asarray(
        list(combinations(range(cfg.n_groups), cfg.test_groups)),
        dtype=np.int64,
    )

    if combos.ndim != 2 or combos.shape[0] != comb(cfg.n_groups, cfg.test_groups):
        raise RuntimeError("Failed to construct CPCV test-group combinations.")

    # Vectorized test mask:
    #   group_ids: (weeks,)
    #   combos:    (splits, test_groups)
    # -> comparison: (splits, weeks, test_groups)
    # -> any:        (splits, weeks)
    test_mask = (
        group_ids[None, :, None] == combos[:, None, :]
    ).any(axis=2)

    # Block boundaries for every chosen test group in every split.
    starts = combos * cfg.group_weeks
    ends = starts + cfg.group_weeks - 1

    pos = positions[None, :, None]

    if cfg.purge_weeks:
        purge_mask = (
            (pos >= (starts[:, None, :] - cfg.purge_weeks))
            & (pos < starts[:, None, :])
        ).any(axis=2)
    else:
        purge_mask = np.zeros_like(test_mask)

    if cfg.embargo_weeks:
        embargo_mask = (
            (pos > ends[:, None, :])
            & (pos <= (ends[:, None, :] + cfg.embargo_weeks))
        ).any(axis=2)
    else:
        embargo_mask = np.zeros_like(test_mask)

    train_mask = ~(test_mask | purge_mask | embargo_mask)

    # Convert local mask positions back to absolute row positions.
    splits: list[dict[str, np.ndarray]] = []

    for row in range(combos.shape[0]):
        train_idx = np.flatnonzero(train_mask[row]).astype(np.int64) + offset
        test_idx = np.flatnonzero(test_mask[row]).astype(np.int64) + offset

        splits.append(
            {
                "train_idx": train_idx,
                "test_idx": test_idx,
            }
        )

    return splits
