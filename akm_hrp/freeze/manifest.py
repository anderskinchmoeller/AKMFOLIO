from dataclasses import dataclass, asdict, is_dataclass
import json
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ModelSignature:
    """
    Immutable signature of a research specification.
    """
    model_spec_version: str
    min_weight_policy_version: str
    search_space_version: str
    pit_fingerprint: str | None
    config_hash: str
    returns_hash: str


@dataclass(frozen=True)
class FrozenManifest:
    """
    Frozen model specification manifest.
    """
    signature: ModelSignature
    freeze_date: str
    last_data_date: str


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_config(config: Any) -> str:
    """
    Hash a config object (dataclass or dict).
    """
    if is_dataclass(config):
        obj = asdict(config)
    elif isinstance(config, dict):
        obj = config
    elif hasattr(config, "__dict__"):
        obj = vars(config)
    else:
        obj = config
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return _hash_bytes(payload)


def hash_returns(returns: pd.DataFrame) -> str:
    """
    Hash the return matrix (index + columns + values).
    """
    idx_bytes = json.dumps(list(map(str, returns.index)), sort_keys=True).encode("utf-8")
    col_bytes = json.dumps(list(map(str, returns.columns)), sort_keys=True).encode("utf-8")
    val_bytes = returns.values.tobytes()
    return _hash_bytes(idx_bytes + col_bytes + val_bytes)


def save_manifest(manifest: FrozenManifest, path: str) -> None:
    """
    Save frozen manifest to JSON.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "signature": asdict(manifest.signature),
                "freeze_date": manifest.freeze_date,
                "last_data_date": manifest.last_data_date,
            },
            fh,
            indent=2,
            sort_keys=True,
        )


def load_manifest(path: str) -> FrozenManifest | None:
    """
    Load frozen manifest from JSON, or None if missing.
    """
    p = Path(path)
    if not p.exists():
        return None

    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    sig = ModelSignature(**data["signature"])
    return FrozenManifest(signature=sig, freeze_date=data["freeze_date"], last_data_date=data["last_data_date"])

