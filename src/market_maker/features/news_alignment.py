from __future__ import annotations
import numpy as np
import pandas as pd


def align_sentiment_to_market(
    market_df: pd.DataFrame,
    news_df: pd.DataFrame,
    lookback_minutes: list[int],
    max_age_minutes: int,
) -> pd.DataFrame:
    result = market_df.copy().reset_index(drop=True)
    news_sorted = news_df.sort_values("published_at", kind="stable").reset_index(drop=True)
    news_times = news_sorted["published_at"].to_numpy()
    news_scores = news_sorted["sentiment_score"].to_numpy() if len(news_sorted) else np.array([])

    for w in lookback_minutes:
        result[f"sentiment_ewm_{w}"] = 0.0
        result[f"sentiment_max_abs_{w}"] = 0.0
        result[f"article_count_{w}"] = 0
    result["minutes_since_last_article"] = float(max_age_minutes)
    result["has_recent_news"] = False

    for idx, t in enumerate(result["timestamp"]):
        # as-of backward join: only news with published_at <= t is visible
        visible_mask = news_times <= t
        n_visible = int(visible_mask.sum())
        if n_visible == 0:
            continue

        last_pub = news_times[visible_mask][-1]
        age_minutes = (t - last_pub) / np.timedelta64(1, "m")
        result.loc[idx, "minutes_since_last_article"] = min(float(age_minutes), float(max_age_minutes))
        result.loc[idx, "has_recent_news"] = bool(age_minutes <= max_age_minutes)

        for w in lookback_minutes:
            window_mask = visible_mask & (news_times > t - np.timedelta64(w, "m"))
            count = int(window_mask.sum())
            result.loc[idx, f"article_count_{w}"] = count
            if count > 0:
                window_scores = news_scores[window_mask]
                weights = np.exp(np.linspace(-1.0, 0.0, count))
                weights /= weights.sum()
                result.loc[idx, f"sentiment_ewm_{w}"] = float(np.dot(weights, window_scores))
                result.loc[idx, f"sentiment_max_abs_{w}"] = float(np.max(np.abs(window_scores)))

    return result


# ---------------------------------------------------------------------------
# Vectorized implementation (production path). `align_sentiment_to_market` above is the slow reference.
# ---------------------------------------------------------------------------
_NS_PER_MIN = 60_000_000_000


def _to_ns(series: pd.Series) -> np.ndarray:
    """datetime Series (tz-aware or naive, any unit) -> int64 nanoseconds since epoch (UTC)."""
    ts = pd.to_datetime(series, utc=True)
    return ts.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype("int64")


def _window_stats(scores: np.ndarray, lo: np.ndarray, hi: np.ndarray, max_elems: int = 4_000_000):
    """Exp-weighted mean and max|score| over news slices [lo, hi) for every (lo, hi) pair.

    Weights are exactly the reference's: ``exp(linspace(-1, 0, count))`` normalised to sum 1 (oldest article
    in the window gets exp(-1), newest gets 1). Variable-length slices are flattened chunk by chunk
    (<= max_elems elements per chunk) and reduced with np.add.reduceat / np.maximum.reduceat.
    Returns (ewm, max_abs) arrays aligned with the inputs; empty slices give 0.
    """
    m = len(lo)
    ewm = np.zeros(m)
    mabs = np.zeros(m)
    counts = (hi - lo).astype(np.int64)
    nz = np.nonzero(counts > 0)[0]
    if len(nz) == 0:
        return ewm, mabs
    # Chunk the non-empty slices so the flattened arrays stay bounded in memory.
    csum = np.cumsum(counts[nz])
    start = 0
    while start < len(nz):
        base = csum[start - 1] if start else 0
        end = int(np.searchsorted(csum, base + max_elems, side="right"))
        end = max(end, start + 1)
        sel = nz[start:end]
        c = counts[sel]
        offs = np.concatenate([[0], np.cumsum(c)[:-1]])
        total = int(c.sum())
        pos = np.arange(total) - np.repeat(offs, c)  # position inside each slice (0 = oldest)
        denom = np.repeat(np.maximum(c - 1, 1), c)
        w = np.exp(-1.0 + pos / denom * 1.0)  # == exp(linspace(-1, 0, c)); for c == 1 -> exp(-1) (normalised to 1)
        idx = np.repeat(lo[sel], c) + pos
        sc = scores[idx]
        wsum = np.add.reduceat(w, offs)
        wn = w / np.repeat(wsum, c)
        ewm[sel] = np.add.reduceat(wn * sc, offs)
        mabs[sel] = np.maximum.reduceat(np.abs(sc), offs)
        start = end
    return ewm, mabs


def align_sentiment_fast(
    market_df: pd.DataFrame,
    news_df: pd.DataFrame,
    lookback_minutes: list[int],
    max_age_minutes: int,
    max_chunk_elements: int = 4_000_000,
) -> pd.DataFrame:
    """Vectorized, leak-free as-of alignment with semantics identical to ``align_sentiment_to_market``.

    For each market timestamp t: visible news = published_at <= t; window w = (t - w minutes, t];
    ``sentiment_ewm_w`` (exp-weighted mean, oldest->newest weights exp(linspace(-1, 0, n))), ``sentiment_max_abs_w``,
    ``article_count_w``, ``minutes_since_last_article`` (capped at max_age_minutes; sentinel = max_age when no
    news is visible), ``has_recent_news`` (age <= max_age_minutes).

    Complexity: two searchsorted per window over all rows, then the exp-weighted statistics are computed only for
    DISTINCT (lo, hi) slice pairs (sparse news => few distinct windows even with millions of snapshots).
    """
    result = market_df.copy().reset_index(drop=True)
    n = len(result)
    t_ns = _to_ns(result["timestamp"])
    news_sorted = news_df.sort_values("published_at", kind="stable").reset_index(drop=True)
    news_ns = _to_ns(news_sorted["published_at"]) if len(news_sorted) else np.zeros(0, dtype="int64")
    scores = news_sorted["sentiment_score"].to_numpy(dtype=float) if len(news_sorted) else np.zeros(0)

    hi = np.searchsorted(news_ns, t_ns, side="right")  # number of visible articles (published_at <= t)
    has_any = hi > 0
    if len(news_ns):
        last_pub = news_ns[np.maximum(hi - 1, 0)]
        age_min = np.where(has_any, (t_ns - last_pub) / _NS_PER_MIN, float(max_age_minutes))
    else:
        age_min = np.full(n, float(max_age_minutes))

    cols: dict[str, np.ndarray] = {}
    for w in lookback_minutes:
        lo = np.searchsorted(news_ns, t_ns - int(w) * _NS_PER_MIN, side="right")  # published_at > t - w
        lo = np.minimum(lo, hi)
        ewm = np.zeros(n)
        mabs = np.zeros(n)
        if len(news_ns):
            key = lo.astype(np.int64) * (len(news_ns) + 1) + hi.astype(np.int64)
            uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
            u_ewm, u_mabs = _window_stats(scores, lo[first], hi[first], max_chunk_elements)
            ewm, mabs = u_ewm[inv], u_mabs[inv]
        cols[f"sentiment_ewm_{w}"] = ewm
        cols[f"sentiment_max_abs_{w}"] = mabs
        cols[f"article_count_{w}"] = (hi - lo).astype(np.int64)
    cols["minutes_since_last_article"] = np.where(has_any, np.minimum(age_min, float(max_age_minutes)),
                                                  float(max_age_minutes))
    cols["has_recent_news"] = has_any & (age_min <= max_age_minutes)
    return pd.concat([result, pd.DataFrame(cols, index=result.index)], axis=1)
