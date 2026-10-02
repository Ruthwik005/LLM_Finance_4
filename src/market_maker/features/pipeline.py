"""Scalable feature pipeline: raw LOB (+ news) -> scaled, split, per-variant feature artifacts.

Two stages (both re-runnable, artifacts live under ``<artifacts_dir>/features``):

1. ``build_base_features``  (needs no Transformer)
   load LOB -> validate -> compute LOB features ONCE on the full time-sorted data (every feature only looks at the
   past, so slicing by split afterwards is leak-free and avoids cold-start artifacts at split starts) ->
   chronological split -> StandardScaler fit on TRAIN rows only -> apply to all splits, clip to +-clip_value, float32.
   Also aligns FinBERT sentiment to the market rows (vectorized as-of join) and stores the RAW sentiment columns.
   Writes: lob_<split>.parquet, lob_features_<split>.npy, sentiment_raw_<split>.npy (+ columns json),
           scaler.joblib, feature_meta.json.

2. ``build_variant_features``  (needs artifacts/transformer/checkpoint.pt for the Transformer variants)
   frozen-Transformer embeddings + horizon probabilities for every row (per split; edge padding at the split start,
   ``transformer_signal_valid`` flag), then fusion per variant. Embedding / probability / sentiment columns are
   standardized with statistics fitted on the TRAIN split only (extras_scaler.json) and clipped, exactly like the
   LOB columns. Writes <variant>_<split>.npy (float32) and <variant>_columns.json (variant names lowercase).

Column order in every variant array: [LOB | sentiment | embedding | probs | signal_valid]. The LOB block is
identical in every variant, so the Avellaneda-Stoikov indices/scale in feature_meta.json are valid for all of them.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from market_maker.data.lob_loader import load_lob
from market_maker.data.news_loader import load_news
from market_maker.data.splits import chronological_split
from market_maker.data.validators import validate_lob, validate_news
from market_maker.features.fusion import build_fused_dataset
from market_maker.features.lob_features import compute_lob_features
from market_maker.features.news_alignment import align_sentiment_fast
from market_maker.features.sentiment import (FinbertScorer, cache_sentiment, load_cached_sentiment,
                                             score_news_dataframe)
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)

SPLITS = ["train", "validation", "test"]
SAC_VARIANTS = ["SAC_LOB_ONLY", "SAC_LOB_TRANSFORMER", "SAC_LOB_SENTIMENT", "SAC_FULL"]
VARIANT_USES = {  # variant -> (uses_sentiment, uses_transformer)
    "SAC_LOB_ONLY": (False, False),
    "SAC_LOB_TRANSFORMER": (False, True),
    "SAC_LOB_SENTIMENT": (True, False),
    "SAC_FULL": (True, True),
}


def slug(variant: str) -> str:
    return variant.lower()


def features_dir(artifacts_dir: str) -> str:
    return os.path.join(artifacts_dir, "features")


def variant_path(artifacts_dir: str, variant: str, split: str) -> str:
    return os.path.join(features_dir(artifacts_dir), f"{slug(variant)}_{split}.npy")


def columns_path(artifacts_dir: str, variant: str) -> str:
    return os.path.join(features_dir(artifacts_dir), f"{slug(variant)}_columns.json")


def lob_split_path(artifacts_dir: str, split: str) -> str:
    return os.path.join(features_dir(artifacts_dir), f"lob_{split}.parquet")


def base_features_path(artifacts_dir: str, split: str) -> str:
    return os.path.join(features_dir(artifacts_dir), f"lob_features_{split}.npy")


def transformer_checkpoint_path(artifacts_dir: str) -> str:
    return os.path.join(artifacts_dir, "transformer", "checkpoint.pt")


def load_meta(artifacts_dir: str) -> dict:
    with open(os.path.join(features_dir(artifacts_dir), "feature_meta.json")) as f:
        return json.load(f)


def _dump_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Sentiment source
# ---------------------------------------------------------------------------
def stub_sentiment_scores(news_df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Deterministic placeholder scores (NOT FinBERT) for smoke tests only; source column = 'synthetic_stub'."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet([1.0, 1.0, 1.0], size=len(news_df))
    out = news_df.reset_index(drop=True).copy()
    out["p_positive"], out["p_negative"], out["p_neutral"] = p[:, 0], p[:, 1], p[:, 2]
    out["sentiment_score"] = out["p_positive"] - out["p_negative"]
    out["sentiment_source"] = "synthetic_stub"
    return out


def resolve_device(pref: str = "auto") -> str:
    from market_maker.models.transformer_train import resolve_device as _rd
    return _rd(pref)


def load_scored_news(cfg, artifacts_dir: str, *, stub: bool = False, news_path: str | None = None,
                     device: str = "auto") -> tuple[pd.DataFrame, dict]:
    """Scored news (published_at, sentiment_score, ...) and a provenance dict.

    Order: explicit stub (labelled, written to a separate *_stub cache so it never poisons the real cache)
    -> cached FinBERT parquet -> run FinBERT on the news file. Never silently falls back to a stub."""
    cache = os.path.join(artifacts_dir, "sentiment", "news_sentiment.parquet")
    news_path = news_path or cfg.data.news_path
    if stub:
        news = load_news(news_path)
        news, _ = validate_news(news)
        scored = stub_sentiment_scores(news, seed=cfg.project.seed)
        cache_sentiment(scored, os.path.join(artifacts_dir, "sentiment", "news_sentiment_stub.parquet"))
        logger.warning("USING STUB SENTIMENT (synthetic_stub) - smoke test only, not FinBERT.")
        return scored, {"source": "synthetic_stub", "n_news": int(len(scored))}
    cached = load_cached_sentiment(cache)
    if cached is not None:
        logger.info("Using cached FinBERT sentiment %s (%d rows)", cache, len(cached))
        cached["published_at"] = pd.to_datetime(cached["published_at"], utc=True)
        return cached, {"source": "finbert_cache", "path": cache, "n_news": int(len(cached))}
    if not os.path.exists(news_path):
        raise FileNotFoundError(
            f"No cached sentiment at {cache} and no news file at {news_path}. Provide news, or run with "
            f"--no-sentiment, or --stub-sentiment for a labeled smoke-test stub.")
    news = load_news(news_path)
    news, _ = validate_news(news)
    scorer = FinbertScorer(model_name=cfg.sentiment.model_name, batch_size=cfg.sentiment.batch_size,
                           device=resolve_device(device if device != "auto" else cfg.project.device))
    scored = score_news_dataframe(news, scorer)
    scored["sentiment_source"] = cfg.sentiment.model_name
    cache_sentiment(scored, cache)
    return scored, {"source": cfg.sentiment.model_name, "path": cache, "n_news": int(len(scored))}


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------
def build_base_features(cfg, artifacts_dir: str, *, with_sentiment: bool = True, stub_sentiment: bool = False,
                        lob_path: str | None = None, news_path: str | None = None,
                        device: str = "auto") -> dict:
    fdir = features_dir(artifacts_dir)
    os.makedirs(fdir, exist_ok=True)
    f = cfg.features

    lob = load_lob(lob_path or cfg.data.lob_path)
    lob, lob_report = validate_lob(lob)
    if len(lob) == 0:
        raise ValueError(f"all LOB rows removed by validation: {lob_report}")
    feats = compute_lob_features(lob, depth=cfg.data.depth, return_lags=f.return_lags, vol_window=f.vol_window)
    assert len(feats) == len(lob) and (feats["timestamp"].to_numpy() == lob["timestamp"].to_numpy()).all(), \
        "feature rows must align with validated LOB rows"
    cols = [c for c in feats.columns if c != "timestamp"]
    X = feats[cols].to_numpy(dtype=np.float64)
    del feats

    lob = lob.assign(_row=np.arange(len(lob)))
    splits = chronological_split(lob, "timestamp", **cfg.data.split.model_dump())
    empty = [k for k, v in splits.items() if len(v) == 0]
    if empty:
        raise ValueError(f"chronological split produced empty partition(s) {empty}; provide more days of data")
    rows = {k: v["_row"].to_numpy() for k, v in splits.items()}
    for k, r in rows.items():
        assert (np.diff(r) == 1).all(), f"split {k} must be a contiguous block of rows"

    # Scaler: TRAIN rows only.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X[rows["train"]])
    joblib.dump(scaler, os.path.join(fdir, "scaler.joblib"))

    sent_cols: list[str] = []
    sent_raw = None
    sent_meta = {"source": None}
    if with_sentiment and cfg.sentiment.enabled:
        scored, sent_meta = load_scored_news(cfg, artifacts_dir, stub=stub_sentiment, news_path=news_path,
                                             device=device)
        scored = scored.assign(sentiment_score=scored["sentiment_score"].fillna(0.0))
        aligned = align_sentiment_fast(lob[["timestamp"]], scored, cfg.sentiment.lookback_minutes,
                                       cfg.sentiment.max_age_minutes)
        sent_cols = [c for c in aligned.columns if c != "timestamp"]
        sent_raw = aligned[sent_cols].to_numpy(dtype=np.float32)
        del aligned
    elif with_sentiment:
        sent_meta = {"source": "disabled_in_config"}

    clip = f.clip_value
    split_info = {}
    for name in SPLITS:
        r = rows[name]
        Z = np.clip(scaler.transform(X[r]), -clip, clip).astype(np.float32)
        np.save(base_features_path(artifacts_dir, name), Z)
        sdf = splits[name].drop(columns=["_row"]).reset_index(drop=True)
        sdf.to_parquet(lob_split_path(artifacts_dir, name))
        if sent_raw is not None:
            np.save(os.path.join(fdir, f"sentiment_raw_{name}.npy"), sent_raw[r])
        split_info[name] = {"rows": int(len(r)), "start": str(sdf["timestamp"].min()),
                            "end": str(sdf["timestamp"].max())}
    if sent_cols:
        _dump_json(os.path.join(fdir, "sentiment_columns.json"), sent_cols)

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lob_feature_columns": cols,
        "feature_mean": scaler.mean_.tolist(),
        "feature_scale": scaler.scale_.tolist(),
        "rolling_vol_index": cols.index("rolling_vol"),
        "midprice_index": cols.index("midprice"),
        "clip_value": clip, "vol_window": f.vol_window, "return_lags": f.return_lags,
        "depth": cfg.data.depth, "label_threshold_bps": f.label_threshold_bps,
        "scaler_fit_rows": "train only", "scaler_path": os.path.join(fdir, "scaler.joblib"),
        "splits": split_info, "lob_validation_report": lob_report,
        "sentiment": {**sent_meta, "columns": sent_cols, "lookback_minutes": cfg.sentiment.lookback_minutes,
                      "max_age_minutes": cfg.sentiment.max_age_minutes},
    }
    _dump_json(os.path.join(fdir, "feature_meta.json"), meta)
    logger.info("Base features built: %s", {k: v["rows"] for k, v in split_info.items()})
    return meta


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------
def _fit_std(train_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_block.astype(np.float64).mean(axis=0)
    std = train_block.astype(np.float64).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)  # constant columns (e.g. signal_valid all 1) are left centered only
    return mean, std


def _apply_std(block: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float) -> np.ndarray:
    return np.clip((block.astype(np.float64) - mean) / std, -clip, clip).astype(np.float32)


def build_variant_features(cfg, artifacts_dir: str, variants: list[str] | None = None, device: str = "auto") -> dict:
    """Embed (if a Transformer checkpoint exists) and fuse the requested variants. Returns {built, skipped}."""
    from market_maker.models.embedding import embed_all, load_frozen_transformer
    fdir = features_dir(artifacts_dir)
    meta = load_meta(artifacts_dir)
    clip = float(meta["clip_value"])
    variants = [v.upper() for v in (variants or SAC_VARIANTS)]
    for v in variants:
        if v not in VARIANT_USES:
            raise ValueError(f"unknown SAC variant {v}; expected one of {list(VARIANT_USES)}")
    lob_cols = meta["lob_feature_columns"]
    base = {s: np.load(base_features_path(artifacts_dir, s)) for s in SPLITS}
    lob_frames = {s: pd.read_parquet(lob_split_path(artifacts_dir, s), columns=["timestamp"]) for s in SPLITS}

    need_sent = any(VARIANT_USES[v][0] for v in variants)
    need_tr = any(VARIANT_USES[v][1] for v in variants)
    built, skipped = [], {}

    sent_blocks, sent_cols = {}, []
    if need_sent:
        scp = os.path.join(fdir, "sentiment_columns.json")
        if not os.path.exists(scp):
            for v in variants:
                if VARIANT_USES[v][0]:
                    skipped[v] = "no sentiment features (run build_features with sentiment enabled)"
        else:
            sent_cols = json.load(open(scp))
            raw = {s: np.load(os.path.join(fdir, f"sentiment_raw_{s}.npy")) for s in SPLITS}
            mean, std = _fit_std(raw["train"])
            sent_blocks = {s: _apply_std(raw[s], mean, std, clip) for s in SPLITS}
            sent_scaler = {"columns": sent_cols, "mean": mean.tolist(), "scale": std.tolist(), "fit": "train only"}

    tr_blocks, tr_probs, tr_valid, tr_emb_cols = {}, {}, {}, []
    if need_tr:
        ckpt = transformer_checkpoint_path(artifacts_dir)
        if not os.path.exists(ckpt):
            for v in variants:
                if VARIANT_USES[v][1]:
                    skipped[v] = f"no Transformer checkpoint at {ckpt} (run scripts/train_transformer.py)"
        else:
            dev = resolve_device(device if device != "auto" else cfg.project.device)
            model, extra = load_frozen_transformer(ckpt, dev)
            if extra["feature_cols"] != lob_cols:
                raise ValueError("Transformer checkpoint was trained on different feature columns; retrain it")
            seq_len = int(extra["sequence_length"])
            emb, probs, valid = {}, {}, {}
            for s in SPLITS:
                emb[s], probs[s], valid[s] = embed_all(model, base[s], seq_len, cfg.transformer.embed_batch_size, dev)
            d = emb["train"].shape[1]
            tr_emb_cols = [f"transformer_embed_{i}" for i in range(d)]
            # Standardize [embedding | probs | signal_valid] with TRAIN stats.
            horizons = sorted(probs["train"].keys())
            blocks = {s: np.concatenate([emb[s]] + [probs[s][h] for h in horizons] + [valid[s][:, None]], axis=1)
                      for s in SPLITS}
            mean, std = _fit_std(blocks["train"])
            std_blocks = {s: _apply_std(blocks[s], mean, std, clip) for s in SPLITS}
            for s in SPLITS:
                tr_blocks[s] = (std_blocks[s][:, :d],
                                {h: std_blocks[s][:, d + 3 * j: d + 3 * j + 3] for j, h in enumerate(horizons)},
                                std_blocks[s][:, -1])
            tr_scaler = {"mean": mean.tolist(), "scale": std.tolist(), "fit": "train only",
                         "layout": "embedding(d) | probs(3 per horizon) | signal_valid",
                         "horizons": horizons}

    scalers = {}
    if sent_blocks:
        scalers["sentiment"] = sent_scaler
    if tr_blocks:
        scalers["transformer"] = tr_scaler
    _dump_json(os.path.join(fdir, "extras_scaler.json"), scalers)

    for v in variants:
        if v in skipped:
            continue
        uses_s, uses_t = VARIANT_USES[v]
        columns = None
        for s in SPLITS:
            lf = pd.DataFrame(base[s], columns=lob_cols)
            lf.insert(0, "timestamp", lob_frames[s]["timestamp"].to_numpy())
            n = len(lf)
            sdf = pd.DataFrame(sent_blocks[s], columns=sent_cols) if uses_s else pd.DataFrame(index=range(n))
            if uses_t:
                e, p, val = tr_blocks[s]
                fused, columns = build_fused_dataset(lf, sdf, e, p, private_state_placeholder_dim=0,
                                                     signal_valid=val, clip_value=clip)
            else:
                fused, columns = build_fused_dataset(lf, sdf, np.zeros((n, 0), np.float32), {},
                                                     private_state_placeholder_dim=0, clip_value=clip)
            expected = len(lob_cols) + (len(sent_cols) if uses_s else 0) + \
                (len(tr_emb_cols) + 3 * len(tr_blocks[s][1]) + 1 if uses_t else 0)
            assert fused.shape == (n, expected) and fused.dtype == np.float32, (v, s, fused.shape, expected)
            assert np.isfinite(fused).all(), f"non-finite values in {v}/{s}"
            assert np.abs(fused).max() <= clip + 1e-6
            np.save(variant_path(artifacts_dir, v, s), fused)
        _dump_json(columns_path(artifacts_dir, v), columns)
        built.append(v)
    logger.info("Variant features built: %s skipped: %s", built, skipped)
    return {"built": built, "skipped": skipped}


def ensure_features(cfg, artifacts_dir: str, variant: str, **kw) -> None:
    """Build whatever is missing for `variant` (used by single-variant scripts)."""
    variant = variant.upper()
    if all(os.path.exists(variant_path(artifacts_dir, variant, s)) for s in SPLITS) and \
            os.path.exists(columns_path(artifacts_dir, variant)):
        return
    uses_s, _ = VARIANT_USES[variant]
    have_base = os.path.exists(os.path.join(features_dir(artifacts_dir), "feature_meta.json"))
    have_sent = os.path.exists(os.path.join(features_dir(artifacts_dir), "sentiment_columns.json"))
    if not have_base or (uses_s and not have_sent):
        build_base_features(cfg, artifacts_dir, with_sentiment=uses_s, **kw)
    res = build_variant_features(cfg, artifacts_dir, [variant])
    if variant in res["skipped"]:
        raise RuntimeError(f"cannot build {variant}: {res['skipped'][variant]}")
