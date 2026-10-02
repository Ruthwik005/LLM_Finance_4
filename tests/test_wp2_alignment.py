import numpy as np
import pandas as pd
import pytest

from market_maker.features.news_alignment import align_sentiment_fast, align_sentiment_to_market


def _market(times):
    return pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True)})


@pytest.mark.parametrize("seed", range(6))
def test_fast_equals_reference_randomized(seed):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    m = t0 + pd.to_timedelta(np.sort(rng.integers(0, 4 * 3600 * 1000, 500)), unit="ms")
    # news before/after the market range, with exact ties and timestamps equal to market timestamps
    nt = t0 + pd.to_timedelta(rng.integers(-1800, 4 * 3600 + 1800, 50) * 1000, unit="ms")
    nt = nt.append(pd.DatetimeIndex([m[5], m[5], m[200], nt[0]]))
    news = pd.DataFrame({"published_at": nt, "sentiment_score": rng.uniform(-1, 1, len(nt))})
    mk = pd.DataFrame({"timestamp": m})
    ref = align_sentiment_to_market(mk, news, [5, 30, 120], 240)
    fast = align_sentiment_fast(mk, news, [5, 30, 120], 240)
    assert list(ref.columns) == list(fast.columns)
    pd.testing.assert_frame_equal(ref, fast, check_dtype=False, atol=1e-12, rtol=0)


def test_fast_empty_news_and_all_future_or_all_past():
    mk = _market(["2024-01-01T10:00:00Z", "2024-01-01T11:00:00Z"])
    empty = pd.DataFrame({"published_at": pd.to_datetime([], utc=True), "sentiment_score": []})
    fast = align_sentiment_fast(mk, empty, [5, 30], 240)
    assert (fast["minutes_since_last_article"] == 240).all() and not fast["has_recent_news"].any()
    assert (fast["article_count_30"] == 0).all()
    future = pd.DataFrame({"published_at": pd.to_datetime(["2024-02-01T00:00:00Z"], utc=True), "sentiment_score": [1.0]})
    pd.testing.assert_frame_equal(align_sentiment_fast(mk, future, [5], 240),
                                  align_sentiment_to_market(mk, future, [5], 240), check_dtype=False)
    old = pd.DataFrame({"published_at": pd.to_datetime(["2023-01-01T00:00:00Z"], utc=True), "sentiment_score": [0.5]})
    f = align_sentiment_fast(mk, old, [5], 240)
    assert (f["minutes_since_last_article"] == 240).all() and not f["has_recent_news"].any()
    assert (f["sentiment_ewm_5"] == 0).all()


# ---- mandatory leakage tests, re-run against the fast implementation ----
def test_fast_future_news_cannot_leak_backward():
    news = pd.DataFrame({"published_at": pd.to_datetime(["2024-01-01T10:00:01Z"], utc=True), "sentiment_score": [0.99]})
    a = align_sentiment_fast(_market(["2024-01-01T10:00:00Z"]), news, [5], 240)
    assert a.loc[0, "article_count_5"] == 0 and a.loc[0, "sentiment_ewm_5"] == 0.0 and not a.loc[0, "has_recent_news"]


def test_fast_news_at_or_before_timestamp_is_visible():
    for pub in ["2024-01-01T10:00:00Z", "2024-01-01T09:59:59Z"]:
        news = pd.DataFrame({"published_at": pd.to_datetime([pub], utc=True), "sentiment_score": [0.8]})
        a = align_sentiment_fast(_market(["2024-01-01T10:00:00Z"]), news, [5], 240)
        assert a.loc[0, "article_count_5"] == 1 and a.loc[0, "has_recent_news"]


def test_fast_window_boundary_is_exclusive_on_the_left():
    news = pd.DataFrame({"published_at": pd.to_datetime(["2024-01-01T09:55:00Z", "2024-01-01T09:55:01Z"], utc=True),
                         "sentiment_score": [0.5, -0.3]})
    a = align_sentiment_fast(_market(["2024-01-01T10:00:00Z"]), news, [5, 30], 240)
    assert a.loc[0, "article_count_5"] == 1 and a.loc[0, "article_count_30"] == 2


def test_fast_does_not_use_future_when_future_news_added():
    """Adding news strictly after t must not change features at t."""
    t = _market(["2024-01-01T10:00:00Z"])
    base = pd.DataFrame({"published_at": pd.to_datetime(["2024-01-01T09:58:00Z"], utc=True), "sentiment_score": [0.4]})
    plus = pd.concat([base, pd.DataFrame({"published_at": pd.to_datetime(["2024-01-01T10:00:01Z"], utc=True),
                                           "sentiment_score": [-1.0]})])
    pd.testing.assert_frame_equal(align_sentiment_fast(t, base, [5], 240), align_sentiment_fast(t, plus, [5], 240))
