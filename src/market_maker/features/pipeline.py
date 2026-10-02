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

Model inputs exclude absolute price LEVEL columns (``features.exclude_model_columns``, default midprice and
microprice; see ``model_feature_columns``): the LOB frame keeps them for labels/backtests, but models see the
scale-free ``mid_vs_ma`` / ``microprice_offset`` instead.

Staleness: feature_meta.json stores ``lob_hash`` / ``sentiment_hash`` and every ``<variant>_columns.json`` a
``feature_hash`` (config sections + data fingerprints + Transformer checkpoint hash). ``ensure_features`` rebuilds or
raises when they do not match the current config/data.

Column order in every variant array: [LOB | sentiment | embedding | probs | signal_valid]. The LOB block is
identical in every variant, so the Avellaneda-Stoikov indices/scale in feature_meta.json are valid for all of them.
"""
from __future__ import annotations
import hashlib
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
from market_maker.features.lob_features import compute_lob_features, model_feature_columns
from market_maker.features.news_alignment import align_sentiment_fast
from market_maker.features.sentiment import (FinbertScorer, cache_sentiment, load_cached_sentiment,
                                             score_news_dataframe)
from market_maker.utils.io import file_sha256
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


def load_columns(artifacts_dir: str, variant: str) -> list[str]:
    """Column names of a variant array (file is {"columns": [...], "feature_hash": ...}; legacy plain list accepted)."""
    with open(columns_path(artifacts_dir, variant)) as f:
        obj = json.load(f)
    return obj["columns"] if isinstance(obj, dict) else obj


def _hash_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def sentiment_cache_path(artifacts_dir: str, source: str | None) -> str:
    name = "news_sentiment_stub.parquet" if source == "synthetic_stub" else "news_sentiment.parquet"
    return os.path.join(artifacts_dir, "sentiment", name)


def lob_feature_hash(cfg, lob_path: str | None = None) -> str:
    """Hash of everything that determines the base LOB feature arrays and splits."""
    return _hash_obj({"features": cfg.features.model_dump(), "split": cfg.data.split.model_dump(),
                      "depth": cfg.data.depth, "lob_sha256": file_sha256(lob_path or cfg.data.lob_path)})


def sentiment_feature_hash(cfg, artifacts_dir: str, source: str | None, news_path: str | None = None) -> str:
    s = cfg.sentiment
    return _hash_obj({"lookback_minutes": s.lookback_minutes, "max_age_minutes": s.max_age_minutes,
                      "enabled": s.enabled, "source": source,
                      "news_sha256": file_sha256(news_path or cfg.data.news_path),
                      "sentiment_cache_sha256": file_sha256(sentiment_cache_path(artifacts_dir, source))})


_TRANSFORMER_TRAIN_ONLY = {"embed_batch_size", "num_workers", "epochs", "patience", "lr", "weight_decay",
                           "batch_size", "dropout", "class_weights"}  # captured by the checkpoint hash instead


def variant_feature_hash(cfg, artifacts_dir: str, variant: str, meta: dict) -> str:
    """Hash stored in <variant>_columns.json. Needs the base `meta` (lob_hash, sentiment hash)."""
    uses_s, uses_t = VARIANT_USES[variant.upper()]
    comp: dict = {"variant": variant.upper(), "lob_hash": meta.get("lob_hash")}
    if uses_s:
        comp["sentiment_hash"] = meta.get("sentiment", {}).get("hash")
    if uses_t:
        comp["transformer"] = {k: v for k, v in cfg.transformer.model_dump().items()
                               if k not in _TRANSFORMER_TRAIN_ONLY}
        comp["checkpoint_sha256"] = file_sha256(transformer_checkpoint_path(artifacts_dir))
    return _hash_obj(comp)


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
    feats = compute_lob_features(lob, depth=cfg.data.depth, return_lags=f.return_lags, vol_window=f.vol_window,
                                 price_ma_window=f.price_ma_window, volume_window=f.volume_window)
    assert len(feats) == len(lob) and (feats["timestamp"].to_numpy() == lob["timestamp"].to_numpy()).all(), \
        "feature rows must align with validated LOB rows"
    # absolute price levels (midprice, microprice) stay in the frame for labels/backtests but are NOT model inputs
    cols = model_feature_columns(feats.columns, f.exclude_model_columns)
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
    clip_fractions: dict = {}
    for name in SPLITS:
        r = rows[name]
        Zraw = scaler.transform(X[r])
        clip_fractions[name] = _clip_fractions(Zraw, cols, clip)
        Z = np.clip(Zraw, -clip, clip).astype(np.float32)
        del Zraw
        np.save(base_features_path(artifacts_dir, name), Z)
        sdf = splits[name].drop(columns=["_row"]).reset_index(drop=True)
        sdf.to_parquet(lob_split_path(artifacts_dir, name))
        if sent_raw is not None:
            np.save(os.path.join(fdir, f"sentiment_raw_{name}.npy"), sent_raw[r])
        split_info[name] = {"rows": int(len(r)), "start": str(sdf["timestamp"].min()),
                            "end": str(sdf["timestamp"].max())}
    if sent_cols:
        _dump_json(os.path.join(fdir, "sentiment_columns.json"), sent_cols)
    else:  # never leave sentiment arrays of an older build around
        for stale in ["sentiment_columns.json"] + [f"sentiment_raw_{n}.npy" for n in SPLITS]:
            if os.path.exists(os.path.join(fdir, stale)):
                os.remove(os.path.join(fdir, stale))
    _warn_clipping(clip_fractions, f.max_clipped_fraction, "LOB")
    if sent_cols:
        sent_meta = {**sent_meta, "hash": sentiment_feature_hash(cfg, artifacts_dir, sent_meta.get("source"),
                                                                  news_path)}

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lob_feature_columns": cols,
        "feature_mean": scaler.mean_.tolist(),
        "feature_scale": scaler.scale_.tolist(),
        "rolling_vol_index": cols.index("rolling_vol"),
        "excluded_model_columns": list(f.exclude_model_columns),
        "lob_hash": lob_feature_hash(cfg, lob_path),
        "clip_fractions": clip_fractions, "max_clipped_fraction": f.max_clipped_fraction,
        "clip_value": clip, "vol_window": f.vol_window, "return_lags": f.return_lags,
        "price_ma_window": f.price_ma_window, "volume_window": f.volume_window,
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
def _clip_fractions(Z: np.ndarray, cols: list[str], clip: float) -> dict[str, float]:
    """Fraction of values per column with |z| > clip (measured before clipping)."""
    frac = (np.abs(Z) > clip).mean(axis=0) if len(Z) else np.zeros(Z.shape[1])
    return {c: float(v) for c, v in zip(cols, frac)}


def _warn_clipping(fractions_by_split: dict, threshold: float, what: str) -> None:
    for split, fr in fractions_by_split.items():
        for col, v in fr.items():
            if v > threshold:
                logger.warning("%s feature '%s' is clipped for %.2f%% of %s rows (threshold %.2f%%): the "
                               "standardized range is saturating", what, col, 100 * v, split, 100 * threshold)


_CHUNK = 100_000


def _fit_std(train_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-column mean/std of a (float32) block, accumulated in float64 chunk by chunk (no full float64 copy)."""
    n, d = train_block.shape
    s1, s2 = np.zeros(d), np.zeros(d)
    for i in range(0, n, _CHUNK):
        c = train_block[i:i + _CHUNK].astype(np.float64)
        s1 += c.sum(axis=0)
    mean = s1 / max(n, 1)
    for i in range(0, n, _CHUNK):
        c = train_block[i:i + _CHUNK].astype(np.float64) - mean
        s2 += (c * c).sum(axis=0)
    std = np.sqrt(s2 / max(n, 1))
    std = np.where(std < 1e-8, 1.0, std)  # constant columns (e.g. signal_valid all 1) are left centered only
    return mean, std


def _standardize_inplace(block: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float, cols: list[str]
                         ) -> dict[str, float]:
    """(block - mean) / std, clipped to +-clip, written back into the float32 `block` chunk by chunk (float64
    only per chunk). Returns the per-column clipped fraction."""
    n = len(block)
    counts = np.zeros(block.shape[1])
    for i in range(0, n, _CHUNK):
        z = (block[i:i + _CHUNK].astype(np.float64) - mean) / std
        counts += (np.abs(z) > clip).sum(axis=0)
        block[i:i + _CHUNK] = np.clip(z, -clip, clip)
    return {c: float(v / max(n, 1)) for c, v in zip(cols, counts)}


def _write_variant_split(path: str, lob: np.ndarray, sent: np.ndarray | None, tr: np.ndarray | None,
                         clip: float) -> None:
    """Write [LOB | sentiment | transformer block] straight into an .npy memmap (float32), chunk by chunk."""
    n = len(lob)
    parts = [lob.shape[1], 0 if sent is None else sent.shape[1], 0 if tr is None else tr.shape[1]]
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, sum(parts)))
    for i in range(0, n, _CHUNK):
        j = min(n, i + _CHUNK)
        o = 0
        for blk in (lob, sent, tr):
            if blk is not None and blk.shape[1]:
                out[i:j, o:o + blk.shape[1]] = np.clip(blk[i:j], -clip, clip)
                o += blk.shape[1]
    out.flush()
    del out


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
    clipfr: dict = {}

    sent_blocks, sent_cols, tr_cols = {}, [], []
    if need_sent:
        scp = os.path.join(fdir, "sentiment_columns.json")
        if not os.path.exists(scp):
            for v in variants:
                if VARIANT_USES[v][0]:
                    skipped[v] = "no sentiment features (run build_features with sentiment enabled)"
        else:
            sent_cols = json.load(open(scp))
            sent_blocks = {s: np.load(os.path.join(fdir, f"sentiment_raw_{s}.npy")) for s in SPLITS}  # float32
            mean, std = _fit_std(sent_blocks["train"])
            for s in SPLITS:  # in place, one split at a time
                clipfr.setdefault("sentiment", {})[s] = _standardize_inplace(sent_blocks[s], mean, std, clip, sent_cols)
            sent_scaler = {"columns": sent_cols, "mean": mean.tolist(), "scale": std.tolist(), "fit": "train only"}

    tr_blocks, tr_emb_cols = {}, []
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
            seq_len = int(extra["sequence_length"])
            horizons = None
            for s in SPLITS:
                e, pr, va = embed_all(model, base[s], seq_len, cfg.transformer.embed_batch_size, dev)
                horizons = sorted(pr.keys())
                d = e.shape[1]
                # float32 block [embedding | probs (3 per horizon) | signal_valid]; arrays are freed right away
                blk = np.concatenate([e] + [pr[h] for h in horizons] + [va[:, None]], axis=1).astype(np.float32)
                del e, pr, va
                tr_blocks[s] = blk
            tr_emb_cols = [f"transformer_embed_{i}" for i in range(d)]
            tr_cols = tr_emb_cols + [f"transformer_prob_h{h}_{c}" for h in horizons for c in ("down", "flat", "up")] \
                + ["transformer_signal_valid"]
            mean, std = _fit_std(tr_blocks["train"])
            for s in SPLITS:  # in place, one split at a time
                clipfr.setdefault("transformer", {})[s] = _standardize_inplace(tr_blocks[s], mean, std, clip, tr_cols)
            tr_scaler = {"mean": mean.tolist(), "scale": std.tolist(), "fit": "train only",
                         "layout": "embedding(d) | probs(3 per horizon) | signal_valid",
                         "horizons": horizons}

    scalers = {}
    if sent_blocks:
        scalers["sentiment"] = sent_scaler
    if tr_blocks:
        scalers["transformer"] = tr_scaler
    sc_path = os.path.join(fdir, "extras_scaler.json")
    if os.path.exists(sc_path):  # keep scalers of previously built groups when only one variant is rebuilt
        with open(sc_path) as fh:
            scalers = {**json.load(fh), **scalers}
    _dump_json(sc_path, scalers)
    for grp, by_split in clipfr.items():
        _warn_clipping(by_split, float(meta.get("max_clipped_fraction", cfg.features.max_clipped_fraction)), grp)
    if clipfr:
        meta["extras_clip_fractions"] = {**meta.get("extras_clip_fractions", {}), **clipfr}
        _dump_json(os.path.join(fdir, "feature_meta.json"), meta)

    for v in variants:
        if v in skipped:
            continue
        uses_s, uses_t = VARIANT_USES[v]
        columns = lob_cols + (sent_cols if uses_s else []) + (tr_cols if uses_t else [])
        for s in SPLITS:
            _write_variant_split(variant_path(artifacts_dir, v, s), base[s], sent_blocks[s] if uses_s else None,
                                 tr_blocks[s] if uses_t else None, clip)
        _dump_json(columns_path(artifacts_dir, v),
                   {"columns": columns, "feature_hash": variant_feature_hash(cfg, artifacts_dir, v, meta)})
        built.append(v)
    logger.info("Variant features built: %s skipped: %s", built, skipped)
    return {"built": built, "skipped": skipped}


def feature_staleness(cfg, artifacts_dir: str, variant: str, news_path: str | None = None,
                      lob_path: str | None = None) -> tuple[str, str] | None:
    """None if the variant's artifacts are present and match the current config/data; else (level, reason) with
    level 'missing' (nothing to lose by building), 'base' (base LOB/sentiment arrays stale) or 'variant'."""
    variant = variant.upper()
    uses_s, uses_t = VARIANT_USES[variant]
    meta_path = os.path.join(features_dir(artifacts_dir), "feature_meta.json")
    if not os.path.exists(meta_path):
        return "missing", "no feature_meta.json"
    meta = load_meta(artifacts_dir)
    if meta.get("lob_hash") != lob_feature_hash(cfg, lob_path):
        return "base", "LOB data / features / split config changed since the base features were built"
    if uses_s:
        sm = meta.get("sentiment", {})
        if not sm.get("columns"):
            return "missing", "no sentiment features in the base build"
        if sm.get("hash") != sentiment_feature_hash(cfg, artifacts_dir, sm.get("source"), news_path):
            return "base", "news / sentiment cache / sentiment config changed since the base features were built"
    have = all(os.path.exists(variant_path(artifacts_dir, variant, s)) for s in SPLITS) and \
        os.path.exists(columns_path(artifacts_dir, variant))
    if not have:
        return "variant", "variant arrays missing"
    with open(columns_path(artifacts_dir, variant)) as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict) or obj.get("feature_hash") != variant_feature_hash(cfg, artifacts_dir, variant, meta):
        return "variant", "feature hash differs (config, Transformer checkpoint, or sentiment changed)"
    return None


def ensure_features(cfg, artifacts_dir: str, variant: str, force: bool = False, **kw) -> None:
    """Make sure up-to-date features exist for `variant`.

    Missing or variant-level-stale arrays (changed Transformer checkpoint / settings) are (re)built. If the BASE
    features are stale (LOB data, feature config, split or sentiment inputs changed) this raises, because a rebuild
    invalidates the Transformer checkpoint and every other variant; pass ``force=True`` (CLI ``--force-features``)
    to rebuild the base features and this variant anyway."""
    variant = variant.upper()
    uses_s, _ = VARIANT_USES[variant]
    st = feature_staleness(cfg, artifacts_dir, variant, news_path=kw.get("news_path"), lob_path=kw.get("lob_path"))
    if st is None:
        return
    level, reason = st
    logger.info("features for %s need (re)build [%s]: %s", variant, level, reason)
    if level == "base" and not force:
        raise RuntimeError(
            f"Stale base features for {variant}: {reason}. Rebuild with `python scripts/build_features.py "
            f"--config <cfg>` (then retrain the Transformer and re-run with --stage fusion), or pass "
            f"force=True (run_experiments.py --force-features) to rebuild automatically.")
    if level in ("missing", "base"):
        build_base_features(cfg, artifacts_dir, with_sentiment=uses_s, **kw)
    res = build_variant_features(cfg, artifacts_dir, [variant])
    if variant in res["skipped"]:
        raise RuntimeError(f"cannot build {variant}: {res['skipped'][variant]}")
