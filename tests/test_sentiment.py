import numpy as np
import pandas as pd
from market_maker.features.sentiment import (
    score_news_dataframe, cache_sentiment, load_cached_sentiment,
)


class StubScorer:
    """Deterministic stand-in for FinbertScorer — avoids downloading weights in unit tests."""
    def score(self, texts: list[str]) -> pd.DataFrame:
        n = len(texts)
        p_pos = np.full(n, 0.6)
        p_neg = np.full(n, 0.1)
        p_neu = np.full(n, 0.3)
        return pd.DataFrame({
            "p_positive": p_pos, "p_negative": p_neg, "p_neutral": p_neu,
            "sentiment_score": p_pos - p_neg,
        })


def test_score_news_dataframe_adds_sentiment_columns():
    news_df = pd.DataFrame({
        "published_at": pd.to_datetime(["2024-01-01T00:00:00Z"]),
        "symbol": ["BTCUSDT"], "headline": ["Great news"], "source": ["synthetic_stub"],
    })
    scored = score_news_dataframe(news_df, StubScorer())
    assert scored.loc[0, "sentiment_score"] == 0.5
    assert set(["p_positive", "p_negative", "p_neutral"]).issubset(scored.columns)


def test_cache_sentiment_round_trip(tmp_path):
    df = pd.DataFrame({
        "published_at": pd.to_datetime(["2024-01-01T00:00:00Z"]),
        "symbol": ["BTCUSDT"], "headline": ["x"], "source": ["s"],
        "p_positive": [0.6], "p_negative": [0.1], "p_neutral": [0.3], "sentiment_score": [0.5],
    })
    path = str(tmp_path / "sentiment.parquet")
    cache_sentiment(df, path)
    loaded = load_cached_sentiment(path)
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), df.reset_index(drop=True))


def test_load_cached_sentiment_returns_none_when_missing(tmp_path):
    assert load_cached_sentiment(str(tmp_path / "missing.parquet")) is None
