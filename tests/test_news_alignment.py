import pandas as pd
from market_maker.features.news_alignment import align_sentiment_to_market


def _market_df(times):
    return pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True)})


def test_future_news_cannot_leak_backward():
    market_df = _market_df(["2024-01-01T10:00:00Z"])
    news_df = pd.DataFrame({
        "published_at": pd.to_datetime(["2024-01-01T10:00:01Z"], utc=True),  # 1 second AFTER market state
        "sentiment_score": [0.99],  # strongly positive
    })
    aligned = align_sentiment_to_market(market_df, news_df, lookback_minutes=[5], max_age_minutes=240)
    assert aligned.loc[0, "article_count_5"] == 0
    assert aligned.loc[0, "sentiment_ewm_5"] == 0.0
    assert aligned.loc[0, "has_recent_news"] == False


def test_news_at_or_before_timestamp_is_visible():
    market_df = _market_df(["2024-01-01T10:00:01Z"])
    news_df = pd.DataFrame({
        "published_at": pd.to_datetime(["2024-01-01T10:00:00Z"], utc=True),  # 1 second BEFORE
        "sentiment_score": [0.8],
    })
    aligned = align_sentiment_to_market(market_df, news_df, lookback_minutes=[5], max_age_minutes=240)
    assert aligned.loc[0, "article_count_5"] == 1
    assert aligned.loc[0, "has_recent_news"] == True


def test_no_news_returns_default_neutral_features():
    market_df = _market_df(["2024-01-01T10:00:00Z"])
    news_df = pd.DataFrame({"published_at": pd.to_datetime([], utc=True), "sentiment_score": []})
    aligned = align_sentiment_to_market(market_df, news_df, lookback_minutes=[5, 30], max_age_minutes=240)
    assert aligned.loc[0, "sentiment_ewm_5"] == 0.0
    assert aligned.loc[0, "article_count_30"] == 0
    assert aligned.loc[0, "minutes_since_last_article"] == 240
    assert aligned.loc[0, "has_recent_news"] == False


def test_multiple_windows_have_different_counts():
    market_df = _market_df(["2024-01-01T10:30:00Z"])
    news_df = pd.DataFrame({
        "published_at": pd.to_datetime([
            "2024-01-01T10:29:00Z",  # 1 min before -> inside 5-min window
            "2024-01-01T10:05:00Z",  # 25 min before -> outside 5-min, inside 30-min
        ], utc=True),
        "sentiment_score": [0.5, -0.3],
    })
    aligned = align_sentiment_to_market(market_df, news_df, lookback_minutes=[5, 30], max_age_minutes=240)
    assert aligned.loc[0, "article_count_5"] == 1
    assert aligned.loc[0, "article_count_30"] == 2


def test_news_published_exactly_at_timestamp_is_visible():
    market_df = _market_df(["2024-01-01T10:00:00Z"])
    news_df = pd.DataFrame({
        "published_at": pd.to_datetime(["2024-01-01T10:00:00Z"], utc=True),  # EXACTLY at market timestamp
        "sentiment_score": [0.75],
    })
    aligned = align_sentiment_to_market(market_df, news_df, lookback_minutes=[5], max_age_minutes=240)
    assert aligned.loc[0, "article_count_5"] == 1
    assert aligned.loc[0, "has_recent_news"] == True
