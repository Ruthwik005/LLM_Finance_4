import pandas as pd
from market_maker.schemas import LobRow, NewsRow
from market_maker.data.sample_data import generate_synthetic_lob, generate_synthetic_news


def test_lob_row_validates_required_fields():
    row = LobRow(
        timestamp="2024-01-01T00:00:00Z", symbol="BTCUSDT",
        bid_price_1=100.0, bid_size_1=1.0, ask_price_1=100.1, ask_size_1=1.0,
        trade_price=None, trade_size=None, trade_side=None,
    )
    assert row.symbol == "BTCUSDT"


def test_news_row_validates_required_fields():
    row = NewsRow(published_at="2024-01-01T00:00:00Z", symbol="BTCUSDT",
                   headline="Test", body=None, source="synthetic_stub", url_or_id=None)
    assert row.headline == "Test"


def test_generate_synthetic_lob_has_expected_columns():
    df = generate_synthetic_lob(n_rows=100, depth=5, symbol="BTCUSDT", seed=42)
    assert len(df) == 100
    for i in range(1, 6):
        assert f"bid_price_{i}" in df.columns
        assert f"ask_price_{i}" in df.columns
    assert (df["ask_price_1"] > df["bid_price_1"]).all()


def test_generate_synthetic_news_is_within_lob_time_range():
    lob_df = generate_synthetic_lob(n_rows=1000, depth=5, symbol="BTCUSDT", seed=42)
    news_df = generate_synthetic_news(n_items=20, symbol="BTCUSDT", lob_df=lob_df, seed=42)
    assert (news_df["published_at"] >= lob_df["timestamp"].min()).all()
    assert (news_df["published_at"] <= lob_df["timestamp"].max()).all()
    assert (news_df["source"] == "synthetic_stub").all()
