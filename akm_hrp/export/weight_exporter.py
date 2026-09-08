from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd


_EPS = 1e-12


@dataclass(frozen=True)
class ExportConfig:
    """
    Portfolio-weight / execution export configuration.
    """

    round_to: int = 6
    output_dir: str = "exports"

    # Portfolio constraints applied before export.
    min_weight: float = 0.0
    max_weight: float = 1.0
    ensure_sum_to_one: bool = True

    # LSE/Yahoo convention:
    # Yahoo commonly quotes London-listed prices in GBp. Multiplying
    # execution shares by 100 is equivalent to dividing a GBp price by 100
    # before computing shares. This must NOT be applied to portfolio weights.
    multiply_lse_shares: bool = True
    lse_suffixes: tuple[str, ...] = (".L",)

    portfolio_value: float = 100_000.0


def _sanitize(
    weights: pd.Series,
    cfg: ExportConfig,
) -> pd.Series:
    """
    Validate and normalize portfolio weights.
    """
    if not isinstance(weights, pd.Series):
        weights = pd.Series(weights, dtype=float)

    w = (
        weights.copy()
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )

    if w.isna().any():
        bad = w.index[w.isna()].tolist()
        raise ValueError(
            f"Non-finite weights for assets: {bad}"
        )

    if cfg.min_weight < 0:
        raise ValueError("min_weight must be non-negative.")

    if cfg.max_weight <= 0:
        raise ValueError("max_weight must be positive.")

    if cfg.min_weight > cfg.max_weight:
        raise ValueError(
            "min_weight cannot exceed max_weight."
        )

    w = w.clip(
        lower=cfg.min_weight,
        upper=cfg.max_weight,
    )

    if cfg.ensure_sum_to_one:
        total = float(w.sum())

        if total <= _EPS:
            raise ValueError(
                "Portfolio has zero total weight."
            )

        w /= total

    return w


def _rounded_weights(
    weights: pd.Series,
    cfg: ExportConfig,
) -> pd.Series:
    """
    Round weights while keeping the exported vector summing to one.

    Any residual rounding error is assigned to the largest position.
    """
    w = _sanitize(weights, cfg)
    rounded = w.round(cfg.round_to)

    if cfg.ensure_sum_to_one and len(rounded):
        residual = 1.0 - float(rounded.sum())

        if abs(residual) > 0:
            largest = rounded.idxmax()
            rounded.loc[largest] += residual
            rounded = rounded.round(cfg.round_to)

    return rounded


def latest_weights(
    weights: pd.DataFrame | pd.Series,
) -> pd.Series:
    """
    Return a Series of current weights from either a weight history DataFrame
    or an already-current weight Series.
    """
    if isinstance(weights, pd.Series):
        return weights.astype(float)

    if not isinstance(weights, pd.DataFrame):
        raise TypeError(
            "weights must be a pandas Series or DataFrame."
        )

    if weights.empty:
        raise ValueError("weights is empty.")

    return weights.iloc[-1].astype(float)


def export_csv(
    weights: pd.Series,
    cfg: ExportConfig,
    filename: str = "weights.csv",
) -> Path:
    w = _rounded_weights(weights, cfg)

    path = Path(cfg.output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    w.rename("weight").to_csv(path)
    return path


def export_json(
    weights: pd.Series,
    cfg: ExportConfig,
    filename: str = "weights.json",
) -> Path:
    w = _rounded_weights(weights, cfg)

    path = Path(cfg.output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        str(asset): float(weight)
        for asset, weight in w.items()
    }

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def export_yaml(
    weights: pd.Series,
    cfg: ExportConfig,
    filename: str = "weights.yaml",
) -> Path:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Install PyYAML to export YAML."
        ) from exc

    w = _rounded_weights(weights, cfg)

    path = Path(cfg.output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {
                str(asset): float(weight)
                for asset, weight in w.items()
            },
            fh,
            sort_keys=True,
        )

    return path


def export_execution_shares(
    weights: pd.Series,
    prices: pd.Series,
    cfg: ExportConfig,
    filename: str = "execution_shares.csv",
    *,
    allow_fractional: bool = True,
) -> Path:
    """
    Export execution quantities from target weights and prices.

    For `.L` symbols with multiply_lse_shares=True, shares are multiplied by
    100 to account for prices quoted in GBp while portfolio value is in GBP.

    Crucially, the LSE multiplier is applied to SHARES only, never weights.
    """
    w = _sanitize(weights, cfg)

    p = (
        pd.Series(prices, dtype=float)
        .reindex(w.index)
        .replace([np.inf, -np.inf], np.nan)
    )

    missing = p.index[p.isna()].tolist()
    if missing:
        raise ValueError(
            f"Missing prices for assets: {missing}"
        )

    non_positive = p.index[p <= 0].tolist()
    if non_positive:
        raise ValueError(
            f"Non-positive prices for assets: {non_positive}"
        )

    notional = w * float(cfg.portfolio_value)
    shares = notional / p

    if cfg.multiply_lse_shares:
        is_lse = pd.Series(
            [
                any(
                    str(asset).endswith(suffix)
                    for suffix in cfg.lse_suffixes
                )
                for asset in shares.index
            ],
            index=shares.index,
        )

        shares.loc[is_lse] *= 100.0

    if not allow_fractional:
        shares = np.floor(shares)

    export = pd.DataFrame(
        {
            "ticker": w.index.astype(str),
            "weight": w.values,
            "price": p.values,
            "target_notional": notional.values,
            "shares": shares.values,
        }
    )

    path = Path(cfg.output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(path, index=False)

    return path


def export_yahoo_upload(
    weights: pd.Series,
    prices: pd.Series,
    cfg: ExportConfig,
    filename: str = "yahoo_upload.csv",
) -> Path:
    """
    Yahoo-friendly two-column execution export.
    """
    detailed_path = export_execution_shares(
        weights,
        prices,
        cfg,
        filename="_tmp_execution_export.csv",
        allow_fractional=True,
    )

    detailed = pd.read_csv(detailed_path)
    detailed_path.unlink(missing_ok=True)

    yahoo = detailed.loc[:, ["ticker", "shares"]]

    path = Path(cfg.output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    yahoo.to_csv(path, index=False)

    return path


def export_all_weights(
    weights: pd.DataFrame | pd.Series,
    cfg: ExportConfig,
    *,
    prefix: str = "portfolio",
    prices: pd.Series | None = None,
) -> dict[str, Path]:
    """
    Export the latest target weights in all supported formats.
    """
    current = latest_weights(weights)

    paths = {
        "csv": export_csv(
            current,
            cfg,
            f"{prefix}_weights.csv",
        ),
        "json": export_json(
            current,
            cfg,
            f"{prefix}_weights.json",
        ),
        "yaml": export_yaml(
            current,
            cfg,
            f"{prefix}_weights.yaml",
        ),
    }

    if prices is not None:
        paths["execution"] = export_execution_shares(
            current,
            prices,
            cfg,
            f"{prefix}_execution.csv",
        )
        paths["yahoo"] = export_yahoo_upload(
            current,
            prices,
            cfg,
            f"{prefix}_yahoo_upload.csv",
        )

    return paths
