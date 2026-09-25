from __future__ import annotations
import numpy as np
import pandas as pd


def generate_synthetic_lob(n_rows: int, depth: int, symbol: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_rows, freq="500ms", tz="UTC")
    mid = 100.0 + np.cumsum(rng.normal(0, 0.01, n_rows))
    data = {"timestamp": timestamps, "symbol": symbol}
    spread = 0.02
    for level in range(1, depth + 1):
        offset = spread / 2 + (level - 1) * 0.01
        data[f"bid_price_{level}"] = mid - offset
        data[f"bid_size_{level}"] = rng.uniform(0.1, 2.0, n_rows)
        data[f"ask_price_{level}"] = mid + offset
        data[f"ask_size_{level}"] = rng.uniform(0.1, 2.0, n_rows)
    trade_mask = rng.random(n_rows) < 0.3
    data["trade_price"] = np.where(trade_mask, mid, np.nan)
    data["trade_size"] = np.where(trade_mask, rng.uniform(0.01, 0.5, n_rows), np.nan)
    data["trade_side"] = np.where(trade_mask, rng.choice(["buy", "sell"], n_rows), None)
    return pd.DataFrame(data)


def generate_synthetic_news(n_items: int, symbol: str, lob_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t_min, t_max = lob_df["timestamp"].min(), lob_df["timestamp"].max()
    span_ns = (t_max - t_min).value
    offsets_ns = np.sort(rng.integers(0, span_ns, n_items))
    published_at = [t_min + pd.Timedelta(nanoseconds=int(o)) for o in offsets_ns]
    headlines = [f"Synthetic headline {i} about {symbol}" for i in range(n_items)]
    return pd.DataFrame({
        "published_at": published_at,
        "symbol": symbol,
        "headline": headlines,
        "body": None,
        "source": "synthetic_stub",
        "url_or_id": [f"stub-{i}" for i in range(n_items)],
    })
