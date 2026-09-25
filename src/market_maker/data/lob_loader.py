from __future__ import annotations
import pandas as pd

REQUIRED_LOB_COLUMNS = [
    "timestamp", "symbol", "bid_price_1", "bid_size_1", "ask_price_1", "ask_size_1",
]


def load_lob(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    missing = [c for c in REQUIRED_LOB_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"LOB file {path} missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
