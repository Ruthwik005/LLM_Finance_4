"""Tests for the WP2 review fixes (stale resume, feature cache hash, level-free features, memory path, label gaps)."""
import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from market_maker.data.sample_data import generate_synthetic_lob
from market_maker.features import pipeline
from market_maker.features.fusion import build_fused_dataset
from market_maker.features.lob_features import compute_lob_features, model_feature_columns
from market_maker.features.pipeline import (SPLITS, build_base_features, ensure_features, load_columns, load_meta,
                                             variant_path)
from market_maker.features.temporal_dataset import IGNORE_LABEL, WindowDataset, compute_horizon_labels
from market_maker.training import experiment as exp
from market_maker.training import train_sac
from market_maker.training.env_factory import evenly_spaced_starts, make_env
from market_maker.training.experiment import (check_run_completed, compute_config_hash, default_experiment_id,
                                               run_experiment)


# ---------------- I4: labels across session gaps ----------------
def test_labels_ignored_across_overnight_gap():
    day1 = pd.date_range("2024-01-01 20:00", periods=120, freq="30s", tz="UTC")       # until 20:59:30
    day2 = pd.date_range("2024-01-02 08:00", periods=120, freq="30s", tz="UTC")       # next morning
    ts = day1.append(day2)
    mid = np.linspace(100, 101, len(ts))
    lab = compute_horizon_labels(ts, mid, [5], 5.0)[5]
    # rows in the last 5 minutes of day 1 would jump over the gap -> IGNORE, not a label
    assert (lab[110:120] == IGNORE_LABEL).all()
    assert (lab[:110] >= 0).all()
    assert (lab[120:230] >= 0).all() and (lab[230:] == IGNORE_LABEL).all()
    # permissive tolerance restores the (wrong) cross-gap labels
    lab2 = compute_horizon_labels(ts, mid, [5], 5.0, max_gap_factor=1e9)[5]
    assert (lab2[110:120] >= 0).all()
    ds = WindowDataset(np.zeros((len(ts), 2), np.float32), ts, mid, 10, [5], 5.0, max_gap_factor=1.5)
    assert (ds.labels != IGNORE_LABEL).all()  # windows whose only label is ignored are dropped


# ---------------- I2: level-free model inputs ----------------
def _lob(n=400, seed=1, scale=1.0):
    df = generate_synthetic_lob(n_rows=n, depth=5, symbol="X", seed=seed)
    for c in df.columns:
        if "price" in c:
            df[c] = df[c] * scale
    df["timestamp"] = pd.date_range("2024-01-01", periods=n, freq="10s", tz="UTC")
    return df


def test_model_columns_exclude_levels_and_new_feature_is_scale_free():
    f1 = compute_lob_features(_lob(), 5, [1, 5], price_ma_window=50)
    f2 = compute_lob_features(_lob(scale=1000.0), 5, [1, 5], price_ma_window=50)
    assert "midprice" in f1.columns  # kept for labels / backtests
    cols = model_feature_columns(f1.columns)
    assert "midprice" not in cols and "microprice" not in cols and "timestamp" not in cols
    assert "mid_vs_ma" in cols and "microprice_offset" in cols
    np.testing.assert_allclose(f1["mid_vs_ma"], f2["mid_vs_ma"], atol=1e-9)  # no absolute level in it
    assert model_feature_columns(["a", "midprice", "b"], exclude=["a"]) == ["midprice", "b"]
    # backward looking: changing the future does not change past values
    df = _lob()
    f_a = compute_lob_features(df, 5, [1], price_ma_window=20)
    df2 = df.copy()
    df2.loc[300:, "bid_price_1"] += 5
    f_b = compute_lob_features(df2, 5, [1], price_ma_window=20)
    np.testing.assert_allclose(f_a["mid_vs_ma"][:300], f_b["mid_vs_ma"][:300])
    # volume window is config driven
    d = _lob()
    d["trade_size"] = 1.0
    assert compute_lob_features(d, 5, [1], volume_window=3)["rolling_traded_volume"].iloc[10] == 3.0


# ---------------- I1/I2c: feature cache hash + clip fractions (shares the session fixture) ----------------
def test_meta_has_hashes_and_clip_fractions(wp2_built):
    cfg, art = wp2_built
    meta = load_meta(art)
    assert meta["lob_hash"] and meta["sentiment"]["hash"]
    assert set(meta["clip_fractions"]) == set(SPLITS)
    assert set(meta["clip_fractions"]["train"]) == set(meta["lob_feature_columns"])
    assert "extras_clip_fractions" in meta and "transformer" in meta["extras_clip_fractions"]
    for v in pipeline.SAC_VARIANTS:
        with open(pipeline.columns_path(art, v)) as f:
            assert json.load(f)["feature_hash"]
        assert load_columns(art, v)[:len(meta["lob_feature_columns"])] == meta["lob_feature_columns"]
    assert "midprice" not in meta["lob_feature_columns"]


def test_clip_warning_logged(tmp_path, monkeypatch):
    msgs = []
    monkeypatch.setattr(pipeline.logger, "warning", lambda m, *a, **k: msgs.append(m % a))
    pipeline._warn_clipping({"train": {"a": 0.5, "b": 0.0}}, 0.01, "LOB")
    assert len(msgs) == 1 and "'a'" in msgs[0]


def test_ensure_features_reuses_rebuilds_or_raises(wp2_copy):
    cfg, art = wp2_copy
    p = variant_path(art, "SAC_FULL", "train")
    t0 = os.path.getmtime(p)
    ensure_features(cfg, art, "SAC_FULL")                       # unchanged -> reused
    assert os.path.getmtime(p) == t0
    # variant-level change (transformer label settings) -> variant rebuilt automatically
    cfg2 = cfg.model_copy(deep=True)
    cfg2.transformer.label_max_gap_factor = 3.0
    ensure_features(cfg2, art, "SAC_FULL")
    assert os.path.getmtime(p) > t0
    assert pipeline.feature_staleness(cfg2, art, "SAC_FULL") is None
    # base-level change (feature config) -> clear error, rebuild with force
    cfg3 = cfg.model_copy(deep=True)
    cfg3.features.return_lags = [1, 3]
    with pytest.raises(RuntimeError, match="Stale base features"):
        ensure_features(cfg3, art, "SAC_LOB_ONLY")
    ensure_features(cfg3, art, "SAC_LOB_ONLY", force=True, stub_sentiment=True)
    assert "return_lag_3" in load_meta(art)["lob_feature_columns"]
    assert pipeline.feature_staleness(cfg3, art, "SAC_LOB_ONLY") is None
    # other variants built under the old config are now flagged stale
    assert pipeline.feature_staleness(cfg3, art, "SAC_FULL")[0] in ("base", "variant", "missing")
    # LOB data fingerprint is part of the hash
    assert pipeline.lob_feature_hash(cfg3) == load_meta(art)["lob_hash"]


# ---------------- I3: memmap path == simple path ----------------
def test_memmap_variant_write_equals_build_fused_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_CHUNK", 7)  # force several chunks
    rng = np.random.default_rng(0)
    n, clip = 50, 3.0
    lob = rng.normal(size=(n, 6)).astype(np.float32) * 2
    lob = np.clip(lob, -clip, clip)
    sent = rng.normal(size=(n, 3)).astype(np.float32)
    emb, probs = rng.normal(size=(n, 4)).astype(np.float32), {1: rng.random((n, 3)).astype(np.float32),
                                                              5: rng.random((n, 3)).astype(np.float32)}
    valid = (np.arange(n) > 4).astype(np.float32)
    block = np.concatenate([emb, probs[1], probs[5], valid[:, None]], axis=1)
    for use_s, use_t in [(False, False), (True, False), (False, True), (True, True)]:
        path = str(tmp_path / f"v{use_s}{use_t}.npy")
        pipeline._write_variant_split(path, lob, sent if use_s else None, block if use_t else None, clip)
        lf = pd.DataFrame(lob, columns=list("abcdef"))
        sdf = pd.DataFrame(sent, columns=["s0", "s1", "s2"]) if use_s else pd.DataFrame(index=range(n))
        if use_t:
            ref, _ = build_fused_dataset(lf, sdf, emb, probs, 0, signal_valid=valid, clip_value=clip)
        else:
            ref, _ = build_fused_dataset(lf, sdf, np.zeros((n, 0), np.float32), {}, 0, clip_value=clip)
        np.testing.assert_array_equal(np.load(path), ref)


def test_standardize_inplace_matches_float64_formula(monkeypatch):
    monkeypatch.setattr(pipeline, "_CHUNK", 11)
    rng = np.random.default_rng(1)
    blk = (rng.normal(size=(100, 5)) * [1, 10, 0.1, 5, 0.0] + 3).astype(np.float32)
    mean, std = pipeline._fit_std(blk)
    ref_mean, ref_std = blk.astype(np.float64).mean(axis=0), blk.astype(np.float64).std(axis=0)
    np.testing.assert_allclose(mean, ref_mean)
    np.testing.assert_allclose(std, np.where(ref_std < 1e-8, 1.0, ref_std))
    expected = np.clip((blk.astype(np.float64) - mean) / std, -2, 2).astype(np.float32)
    fr = pipeline._standardize_inplace(blk, mean, std, 2.0, list("abcde"))
    np.testing.assert_array_equal(blk, expected)
    assert blk.dtype == np.float32 and fr["a"] > 0


# ---------------- C1: resume correctness ----------------
def _run(cfg, art, db, **kw):
    kw.setdefault("variants", ["SAC_LOB_ONLY"])
    kw.setdefault("seeds", [0])
    kw.setdefault("fast", True)
    return run_experiment(cfg, artifacts_dir=art, db_path=db, data_mode="synthetic_smoke_test", **kw)


def _mt(art, rel):
    return os.path.getmtime(os.path.join(art, rel))


def test_resume_skips_matching_and_retrains_on_any_mismatch(wp2_copy, tmp_path):
    cfg, art = wp2_copy
    db = str(tmp_path / "t.db")
    _run(cfg, art, db, experiment_id="e1")
    md = json.load(open(os.path.join(art, "evaluation", "sac_lob_only_seed0", "metadata.json")))
    for k in ("experiment_id", "fast_mode", "total_timesteps", "eval_freq", "n_test_windows", "seed", "variant",
              "config_hash", "feature_hash"):
        assert k in md, k
    assert md["fast_mode"] is True and md["total_timesteps"] == exp.FAST_BUDGET["total_timesteps"]
    model, base = "rl/sac_lob_only/seed0/best_model.zip", "evaluation/heuristic/metrics.json"
    m0, b0 = _mt(art, model), _mt(art, base)

    _run(cfg, art, db, experiment_id="e1", resume=True)                   # identical -> skipped
    assert _mt(art, model) == m0 and _mt(art, base) == b0

    os.remove(os.path.join(art, model))                                    # best_model.zip missing -> retrain
    _run(cfg, art, db, experiment_id="e1", resume=True)
    assert os.path.exists(os.path.join(art, model))
    m1 = _mt(art, model)
    assert m1 > m0 and _mt(art, base) == b0                                # baselines still reused

    _run(cfg, art, db, experiment_id="e2", resume=True)                    # other experiment id -> retrain
    m2 = _mt(art, model)
    assert m2 > m1

    cfg2 = cfg.model_copy(deep=True)
    cfg2.sac.learning_rate = 1e-4                                          # config hash differs -> retrain
    _run(cfg2, art, db, experiment_id="e2", resume=True)
    m3 = _mt(art, model)
    assert m3 > m2

    _run(cfg2, art, db, experiment_id="e2", fast=False, resume=True)       # real run after smoke run -> retrain
    m4 = _mt(art, model)
    assert m4 > m3
    md = json.load(open(os.path.join(art, "evaluation", "sac_lob_only_seed0", "metadata.json")))
    assert md["fast_mode"] is False and md["total_timesteps"] == cfg2.sac.total_timesteps

    # without --resume a run always retrains (latest wins)
    _run(cfg2, art, db, experiment_id="e2", fast=False)
    assert _mt(art, model) > m4


def test_check_run_completed_unit(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "metrics.json").write_text("{}")
    expected = {"experiment_id": "x", "fast_mode": False, "total_timesteps": 10, "config_hash": "h",
                "feature_hash": "f"}
    assert not check_run_completed(str(d), expected)[0]                    # metadata missing
    (d / "metadata.json").write_text(json.dumps(expected))
    model = tmp_path / "best_model.zip"
    assert not check_run_completed(str(d), expected, str(model))[0]       # model missing
    model.write_text("x")
    assert check_run_completed(str(d), expected, str(model)) == (True, "ok")
    for k, v in [("experiment_id", "y"), ("fast_mode", True), ("total_timesteps", 11), ("config_hash", "z"),
                 ("feature_hash", "q")]:
        ok, why = check_run_completed(str(d), {**expected, k: v}, str(model))
        assert not ok and k in why


def test_config_hash_sensitive_to_sections(wp2_built):
    cfg, _ = wp2_built
    h = lambda c, f="fh", **k: compute_config_hash(c, c.sac, 2, 3, 120, f, **k)  # noqa: E731
    base = h(cfg)
    assert base == h(cfg)
    assert base != h(cfg, "other")
    for section, field, val in [("sac", "learning_rate", 0.5), ("features", "vol_window", 7),
                                ("transformer", "d_model", 99), ("simulator", "max_inventory", 5.0),
                                ("reward", "inventory_penalty", 0.5), ("evaluation", "adverse_horizon_steps", 99)]:
        c = cfg.model_copy(deep=True)
        setattr(getattr(c, section), field, val)
        assert h(c) != base, section
    assert h(cfg, baselines=True) != base


def test_fast_and_real_summaries_do_not_collide(wp2_copy, tmp_path):
    cfg, art = wp2_copy
    db = str(tmp_path / "t.db")
    assert default_experiment_id(cfg, True) == cfg.experiment_id + "_fast" != default_experiment_id(cfg, False)
    s = _run(cfg, art, db)  # fast, no explicit id
    ev = os.path.join(art, "evaluation")
    assert s["experiment_id"] == cfg.experiment_id + "_fast"
    assert os.path.exists(os.path.join(ev, f"{cfg.experiment_id}_fast_summary.json"))
    assert not os.path.exists(os.path.join(ev, f"{cfg.experiment_id}_summary.json"))
    s2 = _run(cfg, art, db, fast=False)
    assert s2["experiment_id"] == cfg.experiment_id
    assert os.path.exists(os.path.join(ev, f"{cfg.experiment_id}_summary.json"))
    assert os.path.exists(os.path.join(ev, f"{cfg.experiment_id}_fast_summary.json"))


# ---------------- minor ----------------
def test_checkpoint_eval_fn_uses_validation_env(wp2_copy, tmp_path, monkeypatch):
    cfg, art = wp2_copy
    meta = load_meta(art)
    seen = []
    real = train_sac.evaluate_on_windows

    def spy(model, env, starts):
        seen.append((env, tuple(starts)))
        return real(model, env, starts)

    monkeypatch.setattr(train_sac, "evaluate_on_windows", spy)
    tr_env = make_env(cfg, "SAC_LOB_ONLY", "train", art, "train", seed=1)
    val_env = make_env(cfg, "SAC_LOB_ONLY", "validation", art, "eval")
    test_env = make_env(cfg, "SAC_LOB_ONLY", "test", art, "eval")
    starts = evenly_spaced_starts(len(val_env.raw_lob_df), cfg.episode_steps, 2)
    train_sac.train_sac_agent(tr_env, val_env, cfg.sac, seed=1, out_dir=str(tmp_path / "m"), val_starts=starts)
    assert seen and all(e is val_env and s == tuple(starts) for e, s in seen)
    assert val_env.raw_lob_df["timestamp"].min() == pd.Timestamp(meta["splits"]["validation"]["start"])
    assert all(e is not test_env for e, _ in seen)

    # and run_experiment wires the validation split env (not test) into train_sac_agent
    captured = {}
    real_train = exp.train_sac_agent

    def spy_train(train_env, val_env_, *a, **k):
        captured["val_start"] = val_env_.raw_lob_df["timestamp"].min()
        captured["val_windows"] = k["val_starts"]
        return real_train(train_env, val_env_, *a, **k)

    monkeypatch.setattr(exp, "train_sac_agent", spy_train)
    _run(cfg, art, str(tmp_path / "t.db"), experiment_id="w")
    assert captured["val_start"] == pd.Timestamp(meta["splits"]["validation"]["start"])
    assert captured["val_start"] != pd.Timestamp(meta["splits"]["test"]["start"])


def test_fallback_overwrites_stale_best_model(wp2_copy, tmp_path, monkeypatch):
    cfg, art = wp2_copy
    out = tmp_path / "m"
    out.mkdir()
    (out / "best_model.zip").write_bytes(b"stale")
    monkeypatch.setattr(train_sac, "evaluate_on_windows",
                        lambda m, e, s: {"val_reward_mean": float("nan"), "val_pnl_mean": 0.0, "val_map": 0.0})
    tr_env = make_env(cfg, "SAC_LOB_ONLY", "train", art, "train", seed=1)
    val_env = make_env(cfg, "SAC_LOB_ONLY", "validation", art, "eval")
    res = train_sac.train_sac_agent(tr_env, val_env, cfg.sac, seed=1, out_dir=str(out), val_starts=[0])
    assert res.best_step is None
    assert (out / "best_model.zip").read_bytes() != b"stale"
    from stable_baselines3 import SAC
    SAC.load(str(out / "best_model.zip"), device="cpu")
