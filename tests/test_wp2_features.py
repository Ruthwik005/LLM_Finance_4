import hashlib
import json
import os
import numpy as np
import pandas as pd
import torch

from market_maker.data.sample_data import generate_synthetic_lob
from market_maker.features.lob_features import compute_lob_features
from market_maker.features.pipeline import (load_columns, SPLITS, SAC_VARIANTS, base_features_path, columns_path, load_meta,
                                             lob_split_path, transformer_checkpoint_path, variant_path)
from market_maker.features.temporal_dataset import (IGNORE_LABEL, WindowDataset, build_windows,
                                                     compute_horizon_labels)
from market_maker.models.embedding import embed_all, load_frozen_transformer
from market_maker.models.transformer_encoder import TemporalTransformer


def test_window_dataset_matches_build_windows():
    df = generate_synthetic_lob(n_rows=1500, depth=5, symbol="X", seed=3)
    df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="10s", tz="UTC")
    feats = compute_lob_features(df, depth=5, return_lags=[1, 5])
    cols = [c for c in feats.columns if c != "timestamp"]
    L, horizons = 20, [1, 5]
    X_ref, y_ref = build_windows(feats, cols, L, horizons, tick_threshold=0.0005)
    ds = WindowDataset(feats[cols].to_numpy(np.float32), feats["timestamp"], feats["midprice"].to_numpy(), L,
                       horizons, threshold_bps=5.0, stride=1)
    full = (ds.labels != IGNORE_LABEL).all(axis=1)  # build_windows keeps only fully labelled windows
    assert full.sum() == len(X_ref) > 0
    idx = np.nonzero(full)[0]
    got_x = np.stack([ds[i][0].numpy() for i in idx])
    np.testing.assert_array_equal(got_x, X_ref)
    for j, h in enumerate(horizons):
        np.testing.assert_array_equal(ds.labels[idx, j], y_ref[h])
    # partially-labelled tail windows exist and use the ignore label
    assert (ds.labels == IGNORE_LABEL).any()


def test_window_dataset_stride_and_label_masking():
    n = 200
    ts = pd.date_range("2024-01-01", periods=n, freq="30s", tz="UTC")
    mid = np.linspace(100, 101, n)
    ds = WindowDataset(np.zeros((n, 3), np.float32), ts, mid, 10, [1, 5], 5.0, stride=7)
    assert (np.diff(ds.ends) == 7).all()
    lab = compute_horizon_labels(ts, mid, [5], 5.0)[5]
    assert (lab[-10:] == IGNORE_LABEL).all() and (lab[:-11] >= 0).all()  # 5 min = 10 rows of 30s


def test_scaler_fit_on_train_only(wp2_built):
    cfg, art = wp2_built
    meta = load_meta(art)
    lob = pd.read_parquet(cfg.data.lob_path)
    from market_maker.data.validators import validate_lob
    lob, _ = validate_lob(lob)
    feats = compute_lob_features(lob, depth=cfg.data.depth, return_lags=cfg.features.return_lags,
                                 vol_window=cfg.features.vol_window)
    cols = meta["lob_feature_columns"]
    n_tr = meta["splits"]["train"]["rows"]
    np.testing.assert_allclose(meta["feature_mean"], feats[cols].to_numpy()[:n_tr].mean(axis=0), rtol=1e-9, atol=1e-12)
    assert not np.allclose(meta["feature_mean"], feats[cols].to_numpy().mean(axis=0), rtol=1e-6)
    # train block is standardized by its own stats (mean ~0 unless clipped), other splits are not
    tr = np.load(base_features_path(art, "train"))
    assert "midprice" not in cols and "microprice" not in cols and "mid_vs_ma" in cols  # level-free inputs only
    j = cols.index("mid_vs_ma")
    assert abs(tr[:, j].mean()) < 0.05
    assert abs(np.load(base_features_path(art, "test"))[:, j].mean()) > 0.0


def test_variant_artifacts_composition_finite_clipped(wp2_built):
    cfg, art = wp2_built
    meta = load_meta(art)
    n_lob = len(meta["lob_feature_columns"])
    sent = json.load(open(os.path.join(art, "features", "sentiment_columns.json")))
    d = cfg.transformer.d_model
    n_probs = 3 * len(cfg.transformer.horizons_minutes)
    expected = {"SAC_LOB_ONLY": n_lob, "SAC_LOB_SENTIMENT": n_lob + len(sent),
                "SAC_LOB_TRANSFORMER": n_lob + d + n_probs + 1, "SAC_FULL": n_lob + len(sent) + d + n_probs + 1}
    for v in SAC_VARIANTS:
        cols = load_columns(art, v)
        assert cols[:n_lob] == meta["lob_feature_columns"]  # LOB block identical -> AS indices valid everywhere
        assert len(cols) == expected[v]
        assert ("transformer_signal_valid" in cols) == (v in ("SAC_LOB_TRANSFORMER", "SAC_FULL"))
        assert any(c.startswith("sentiment_") for c in cols) == (v in ("SAC_LOB_SENTIMENT", "SAC_FULL"))
        for s in SPLITS:
            a = np.load(variant_path(art, v, s))
            n = len(pd.read_parquet(lob_split_path(art, s), columns=["timestamp"]))
            assert a.shape == (n, expected[v]) and a.dtype == np.float32
            assert np.isfinite(a).all() and np.abs(a).max() <= cfg.features.clip_value + 1e-6


def test_extras_standardized_with_train_stats(wp2_built):
    cfg, art = wp2_built
    sc = json.load(open(os.path.join(art, "features", "extras_scaler.json")))
    assert sc["sentiment"]["fit"] == "train only" and sc["transformer"]["fit"] == "train only"
    cols = load_columns(art, "SAC_FULL")
    a = np.load(variant_path(art, "SAC_FULL", "train"))
    j = cols.index("transformer_embed_0")
    assert abs(a[:, j].mean()) < 0.1 and abs(a[:, j].std() - 1) < 0.2


def test_embed_all_alignment_shape_and_frozen(wp2_built):
    cfg, art = wp2_built
    ckpt = transformer_checkpoint_path(art)
    sha = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
    model, extra = load_frozen_transformer(ckpt, "cpu")
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    L = extra["sequence_length"]
    X = np.load(base_features_path(art, "validation"))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    emb, probs, valid = embed_all(model, X, L, batch_size=37, device="cpu")
    assert emb.shape == (len(X), cfg.transformer.d_model) and valid.shape == (len(X),)
    assert valid[:L - 1].sum() == 0 and valid[L - 1:].all()
    for h, p in probs.items():
        assert p.shape == (len(X), 3) and np.allclose(p.sum(1), 1, atol=1e-5)
    # row i equals the model applied to the window ending at row i (alignment), and batch size is irrelevant
    i = L + 5
    with torch.no_grad():
        pooled, _ = model(torch.from_numpy(X[i - L + 1:i + 1][None]))
    np.testing.assert_allclose(emb[i], pooled[0].numpy(), atol=1e-5)
    emb2, _, _ = embed_all(model, X, L, batch_size=1000, device="cpu")
    np.testing.assert_allclose(emb, emb2, atol=1e-5)
    # edge padding: row 0 equals the window made of row 0 repeated
    with torch.no_grad():
        p0, _ = model(torch.from_numpy(np.repeat(X[:1], L, axis=0)[None]))
    np.testing.assert_allclose(emb[0], p0[0].numpy(), atol=1e-5)
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert hashlib.sha256(open(ckpt, "rb").read()).hexdigest() == sha


def test_checkpoint_unchanged_by_sac_data_build(wp2_copy):
    from market_maker.features.pipeline import build_variant_features
    cfg, art = wp2_copy
    ckpt = transformer_checkpoint_path(art)
    sha = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
    build_variant_features(cfg, art, ["SAC_FULL"], device="cpu")
    assert hashlib.sha256(open(ckpt, "rb").read()).hexdigest() == sha


def test_transformer_training_curve_and_checkpoint_contents(wp2_built):
    cfg, art = wp2_built
    payload = torch.load(transformer_checkpoint_path(art), map_location="cpu", weights_only=False)
    ex = payload["extra"]
    assert ex["sequence_length"] == cfg.transformer.sequence_length and ex["horizons_minutes"] == [1, 5]
    assert ex["feature_cols"] == load_meta(art)["lob_feature_columns"] and "best_epoch" in ex
