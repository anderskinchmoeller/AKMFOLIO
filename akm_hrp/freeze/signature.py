import datetime as dt
from dataclasses import dataclass

import pandas as pd

from .manifest import (
    ModelSignature,
    FrozenManifest,
    hash_config,
    hash_returns,
    save_manifest,
    load_manifest,
)


@dataclass(frozen=True)
class FreezeConfig:
    model_spec_version: str = "5.0-regret-aware"
    min_weight_policy_version: str = "dynamic-feasible-floor-v1"
    search_space_version: str = "institutional-v3"
    manifest_path: str = "frozen_model_spec.json"
    allow_exploratory_freeze: bool = False
    forward_test_min_weeks: int = 26


def build_signature(
    returns: pd.DataFrame,
    config: object,
    pit_fingerprint: str | None,
    freeze_cfg: FreezeConfig,
) -> ModelSignature:
    """
    Build a model signature from data + config + PIT fingerprint.
    """
    cfg_hash = hash_config(config)
    ret_hash = hash_returns(returns)

    return ModelSignature(
        model_spec_version=freeze_cfg.model_spec_version,
        min_weight_policy_version=freeze_cfg.min_weight_policy_version,
        search_space_version=freeze_cfg.search_space_version,
        pit_fingerprint=pit_fingerprint,
        config_hash=cfg_hash,
        returns_hash=ret_hash,
    )


def freeze_specification(
    returns: pd.DataFrame,
    config: object,
    pit_fingerprint: str | None,
    freeze_cfg: FreezeConfig,
) -> FrozenManifest:
    """
    Create and persist a frozen model specification manifest.
    """
    sig = build_signature(returns, config, pit_fingerprint, freeze_cfg)
    last_date = str(returns.index.max().date())
    freeze_date = str(dt.date.today())

    manifest = FrozenManifest(
        signature=sig,
        freeze_date=freeze_date,
        last_data_date=last_date,
    )
    save_manifest(manifest, freeze_cfg.manifest_path)
    return manifest


def validate_forward_run(
    returns: pd.DataFrame,
    config: object,
    pit_fingerprint: str | None,
    freeze_cfg: FreezeConfig,
) -> FrozenManifest:
    """
    Load existing manifest and validate that:
      - signature matches (config, PIT, search space)
      - current data strictly extends last_data_date
      - forward-test region has enough weeks
    """
    manifest = load_manifest(freeze_cfg.manifest_path)
    if manifest is None:
        raise RuntimeError("No frozen manifest found; run freeze_specification() first.")

    current_sig = build_signature(returns, config, pit_fingerprint, freeze_cfg)

    if current_sig.model_spec_version != manifest.signature.model_spec_version:
        raise RuntimeError("Model spec version mismatch with frozen manifest.")

    if current_sig.min_weight_policy_version != manifest.signature.min_weight_policy_version:
        raise RuntimeError("Min-weight policy version mismatch with frozen manifest.")

    if current_sig.search_space_version != manifest.signature.search_space_version:
        raise RuntimeError("Search-space version mismatch with frozen manifest.")

    if current_sig.config_hash != manifest.signature.config_hash:
        raise RuntimeError("Config hash mismatch with frozen manifest.")

    if current_sig.pit_fingerprint != manifest.signature.pit_fingerprint:
        raise RuntimeError("PIT fingerprint mismatch with frozen manifest.")

    # Historical data used at freeze time must not be revised.
    last_data = pd.Timestamp(manifest.last_data_date)
    historical = returns.loc[returns.index <= last_data]
    if hash_returns(historical) != manifest.signature.returns_hash:
        raise RuntimeError("Historical returns changed since the model was frozen.")

    # forward-test region: strictly after last_data_date
    forward_mask = returns.index > last_data
    forward_ret = returns.loc[forward_mask]

    if len(forward_ret) < freeze_cfg.forward_test_min_weeks:
        raise RuntimeError(
            f"Forward-test region too short: {len(forward_ret)} weeks "
            f"(min {freeze_cfg.forward_test_min_weeks})."
        )

    return manifest
