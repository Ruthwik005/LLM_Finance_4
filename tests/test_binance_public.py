import pandas as pd
from market_maker.data.binance_public import book_ticker_to_lob, agg_trades_to_trades, join_trades_to_lob
from market_maker.data.validators import validate_lob
from market_maker.features.lob_features import compute_lob_features

T0 = 1_700_000_000_000


def _book():
    return pd.DataFrame({"update_id": [1, 2, 3], "best_bid_price": [100.0, 100.1, 100.2], "best_bid_qty": [1, 1, 1],
                         "best_ask_price": [100.5, 100.6, 100.7], "best_ask_qty": [1, 2, 3],
                         "transaction_time": [T0 + 10, T0 + 400, T0 + 600], "event_time": [T0 + 10, T0 + 400, T0 + 600]})


def test_book_depth1_and_features():
    lob = book_ticker_to_lob(_book(), "BTCUSDT", 500)
    assert len(lob) == 2 and lob["bid_price_1"].iloc[0] == 100.1
    v, _ = validate_lob(lob)
    f = compute_lob_features(v, 1, [1])
    assert len(f) == 2 and f["midprice"].notna().all()


def test_trades_join_asof():
    lob = book_ticker_to_lob(_book(), "BTCUSDT", 500)
    tr = agg_trades_to_trades(pd.DataFrame({"a": [1, 2], "p": [100.0, 101.0], "q": [1, 1], "f": [1, 2], "l": [1, 2],
                                            "t": [T0 + 100, T0 + 200], "m": ["True", "False"], "b": [True, True]}), "BTCUSDT")
    assert tr["trade_side"].tolist() == ["sell", "buy"]
    j = join_trades_to_lob(lob, tr, 500)
    assert j["trade_size"].iloc[0] == 2 and j["trade_size"].iloc[1] != j["trade_size"].iloc[1]
