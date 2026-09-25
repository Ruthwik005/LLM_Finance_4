from __future__ import annotations
import pandas as pd

REQUIRED_NEWS_COLUMNS = ["published_at", "headline"]


def load_news(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    missing = [c for c in REQUIRED_NEWS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"News file {path} missing required columns: {missing}")
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df
