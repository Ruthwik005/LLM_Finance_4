import pandas as pd
import pytest
from market_maker.data.adapters import normalize_lob, normalize_news, load_lob_with_mapping

MAP = {"timestamp": {"column": "ts", "unit": "ms"}, "bid_price": "bp{i}", "bid_size": "bq{i}",
       "ask_price": "ap{i}", "ask_size": "aq{i}", "trade_price": "tp", "trade_size": "tq", "trade_side": "ts_side"}


def _df():
    return pd.DataFrame({"ts": [1700000001000, 1700000000000], "bp1": [99.0, 98.0], "bq1": [1, 2], "ap1": [100.0, 99.0],
                         "aq1": [3, 4], "bp2": [98.0, 97.0], "bq2": [1, 1], "ap2": [101.0, 100.0], "aq2": [1, 1],
                         "tp": [99.5, None], "tq": [0.1, None], "ts_side": ["B", "x"]})


def test_normalize_sorted_utc_and_columns():
    out = normalize_lob(_df(), MAP, "BTCUSDT", 2)
    assert str(out["timestamp"].dt.tz) == "UTC"
    assert out["timestamp"].is_monotonic_increasing
    assert {"bid_price_2", "ask_size_2", "trade_side", "symbol"} <= set(out.columns)
    assert out["trade_side"].tolist() == ["unknown", "buy"]


def test_missing_column_error():
    with pytest.raises(ValueError, match="missing source columns"):
        normalize_lob(_df().drop(columns=["bp2"]), MAP, "X", 2)


def test_bad_unit():
    with pytest.raises(ValueError, match="Unknown timestamp unit"):
        normalize_lob(_df(), {**MAP, "timestamp": {"column": "ts", "unit": "weeks"}}, "X", 1)


def test_load_with_yaml(tmp_path):
    import yaml
    _df().to_csv(tmp_path / "a.csv", index=False)
    (tmp_path / "m.yaml").write_text(yaml.safe_dump({"symbol": "BTCUSDT", "depth": 2, "lob": MAP}))
    out = load_lob_with_mapping(tmp_path / "a.csv", tmp_path / "m.yaml")
    assert len(out) == 2 and out["symbol"].iloc[0] == "BTCUSDT"


def test_normalize_news():
    df = pd.DataFrame({"d": ["2020-01-01 10:00:00 UTC", None, "2020-01-01 09:00:00 UTC"], "t": ["a", "b", "c"], "s": ["AAPL"] * 3})
    out = normalize_news(df, {"published_at": "d", "headline": "t", "symbol": "s"})
    assert out["headline"].tolist() == ["c", "a"]
    assert str(out["published_at"].dt.tz) == "UTC"
