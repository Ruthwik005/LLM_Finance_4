"""Conversion helpers for data.binance.vision public files (pure, no network)."""
from __future__ import annotations
import pandas as pd

BOOK_COLS = ["update_id", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty", "transaction_time", "event_time"]
AGG_COLS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]


def to_utc(ms: pd.Series) -> pd.Series:
    """Epoch to UTC; auto-detect ms vs us (Binance spot switched to us in 2025)."""
    v = pd.to_numeric(ms, errors="coerce")
    unit = "us" if v.dropna().abs().median() > 1e14 else "ms"
    return pd.to_datetime(v, unit=unit, utc=True)


def book_ticker_to_lob(raw: pd.DataFrame, symbol: str, interval_ms: int = 500) -> pd.DataFrame:
    """bookTicker -> depth-1 LOB; keeps the last update in each interval_ms bucket (0 = keep all)."""
    df = raw.rename(columns=dict(zip(raw.columns, BOOK_COLS[: len(raw.columns)])))
    ts_col = "transaction_time" if "transaction_time" in df.columns else "event_time"
    out = pd.DataFrame({
        "timestamp": to_utc(df[ts_col]), "symbol": symbol,
        "bid_price_1": pd.to_numeric(df["best_bid_price"]), "bid_size_1": pd.to_numeric(df["best_bid_qty"]),
        "ask_price_1": pd.to_numeric(df["best_ask_price"]), "ask_size_1": pd.to_numeric(df["best_ask_qty"]),
    }).dropna()
    out = out.sort_values("timestamp", kind="stable")
    if interval_ms > 0:
        b = out["timestamp"].dt.floor(f"{interval_ms}ms")
        out = out.groupby(b, sort=True).last()
        out.index.name = None
        out["timestamp"] = out.index
    return out.reset_index(drop=True)


def agg_trades_to_trades(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = raw.copy()
    if "price" not in df.columns or str(df.columns[0]).isdigit() or not isinstance(df.columns[0], str):
        df.columns = (AGG_COLS + ["is_best_match"])[: len(df.columns)]
    side = df["is_buyer_maker"].astype(str).str.lower().map({"true": "sell", "false": "buy"}).fillna("unknown")
    return pd.DataFrame({
        "timestamp": to_utc(df["transact_time"]), "symbol": symbol,
        "trade_price": pd.to_numeric(df["price"]), "trade_size": pd.to_numeric(df["quantity"]), "trade_side": side,
    }).sort_values("timestamp", kind="stable").reset_index(drop=True)


def bucket_trades(trades: pd.DataFrame, interval_ms: int) -> pd.DataFrame:
    """Aggregate trades per bucket: VWAP price, total size, net-aggressor side. Index = bucket end timestamp."""
    t = trades.copy()
    t["bucket"] = t["timestamp"].dt.floor(f"{interval_ms}ms")
    t["signed"] = t["trade_size"].where(t["trade_side"] == "buy", -t["trade_size"])
    t["notional"] = t["trade_price"] * t["trade_size"]
    g = t.groupby("bucket").agg(notional=("notional", "sum"), trade_size=("trade_size", "sum"), signed=("signed", "sum"))
    g["trade_price"] = g["notional"] / g["trade_size"]
    g["trade_side"] = g["signed"].apply(lambda s: "buy" if s > 0 else "sell" if s < 0 else "unknown")
    return g[["trade_price", "trade_size", "trade_side"]].reset_index().rename(columns={"bucket": "timestamp"})


def join_trades_to_lob(lob: pd.DataFrame, trades: pd.DataFrame, interval_ms: int) -> pd.DataFrame:
    """Attach each trade bucket to the nearest PRIOR (or equal) snapshot (as-of backward, tolerance = interval)."""
    tb = bucket_trades(trades, interval_ms).sort_values("timestamp")
    base = lob.drop(columns=[c for c in ("trade_price", "trade_size", "trade_side") if c in lob.columns]).sort_values("timestamp")
    tb = tb.rename(columns={"timestamp": "trade_ts"})
    # Match each snapshot to the bucket starting at/just before it; a bucket is used by at most one snapshot.
    m = pd.merge_asof(base, tb, left_on="timestamp", right_on="trade_ts", direction="backward",
                      tolerance=pd.Timedelta(milliseconds=interval_ms))
    dup = m["trade_ts"].duplicated(keep="first") & m["trade_ts"].notna()
    m.loc[dup, ["trade_price", "trade_size", "trade_side"]] = None
    return m.drop(columns=["trade_ts"]).reset_index(drop=True)
