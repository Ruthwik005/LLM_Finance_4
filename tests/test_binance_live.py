import pandas as pd
from market_maker.data.binance_live import TradeBucket, depth_to_row, build_snapshot_row, merge_parts

P = {"bids": [["100.0", "1"], ["99.9", "2"]], "asks": [["100.1", "3"], ["100.2", "4"]]}
TS = pd.Timestamp("2024-01-01", tz="UTC")


def test_depth_row():
    r = depth_to_row(P, "BTCUSDT", 2, TS)
    assert r["bid_price_2"] == 99.9 and r["ask_size_1"] == 3.0
    assert depth_to_row(P, "BTCUSDT", 3, TS) is None


def test_bucket():
    b = TradeBucket()
    b.add({"p": "100", "q": "1", "m": False})
    b.add({"p": "102", "q": "3", "m": True})
    o = b.flush()
    assert o["trade_size"] == 4 and abs(o["trade_price"] - 101.5) < 1e-9 and o["trade_side"] == "sell"
    assert b.flush()["trade_size"] is None


def test_merge_dedup(tmp_path):
    row = build_snapshot_row(P, TradeBucket(), "BTCUSDT", 2, TS)
    row2 = dict(row, timestamp=TS + pd.Timedelta(seconds=1))
    pd.DataFrame([row2, row]).to_parquet(tmp_path / "lob_part_a.parquet")
    pd.DataFrame([row]).to_parquet(tmp_path / "lob_part_b.parquet")
    df, rep = merge_parts(tmp_path, tmp_path / "out.parquet")
    assert len(df) == 2 and df["timestamp"].is_monotonic_increasing
