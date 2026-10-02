import pytest
import torch

from market_maker.features import pipeline
from market_maker.features.pipeline import (build_variant_features, check_transformer_binding, ensure_features,
                                            load_meta, transformer_checkpoint_path)


def _extra(art):
    return torch.load(transformer_checkpoint_path(art), map_location="cpu", weights_only=False)["extra"]


def test_checkpoint_stores_binding_and_matching_passes(wp2_copy):
    cfg, art = wp2_copy
    extra, meta = _extra(art), load_meta(art)
    assert extra["base_feature_hash"] == meta["lob_hash"] and extra["scaler_sha256"]
    assert check_transformer_binding(meta, extra) is False
    ensure_features(cfg, art, "SAC_FULL")
    assert "transformer_stale" not in load_meta(art)


def test_changed_base_config_with_old_checkpoint_raises_then_escape_hatch(wp2_copy):
    cfg, art = wp2_copy
    cfg2 = cfg.model_copy(deep=True)
    cfg2.features.vol_window = cfg.features.vol_window + 7  # same column names, new distribution
    pipeline.build_base_features(cfg2, art, stub_sentiment=True)   # same checkpoint, new base features
    with pytest.raises(RuntimeError, match="train_transformer.py --config"):
        build_variant_features(cfg2, art, ["SAC_FULL"], device="cpu")
    with pytest.raises(RuntimeError, match="train_transformer.py"):
        ensure_features(cfg2, art, "SAC_FULL")
    res = build_variant_features(cfg2, art, ["SAC_FULL"], device="cpu", allow_stale_transformer=True)
    assert res["built"] == ["SAC_FULL"]
    assert load_meta(art)["transformer_stale"] is True


def test_legacy_checkpoint_without_hash_raises(wp2_copy):
    cfg, art = wp2_copy
    p = transformer_checkpoint_path(art)
    payload = torch.load(p, map_location="cpu", weights_only=False)
    payload["extra"].pop("base_feature_hash"); payload["extra"].pop("scaler_sha256")
    torch.save(payload, p)
    with pytest.raises(RuntimeError, match="legacy.*train_transformer.py"):
        build_variant_features(cfg, art, ["SAC_FULL"], device="cpu")
    with pytest.raises(RuntimeError, match="legacy"):
        ensure_features(cfg, art, "SAC_FULL")
    build_variant_features(cfg, art, ["SAC_FULL"], device="cpu", allow_stale_transformer=True)
    assert load_meta(art)["transformer_stale"] is True
