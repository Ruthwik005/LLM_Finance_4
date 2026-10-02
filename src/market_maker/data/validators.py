from __future__ import annotations
import pandas as pd
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)


def validate_lob(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"crossed_removed": 0, "negative_size_removed": 0, "duplicate_removed": 0}
    original_len = len(df)

    crossed_mask = df["bid_price_1"] >= df["ask_price_1"]
    report["crossed_removed"] = int(crossed_mask.sum())
    df = df.loc[~crossed_mask].copy()

    size_cols = [c for c in df.columns if "size" in c]
    negative_mask = (df[size_cols] < 0).any(axis=1)
    report["negative_size_removed"] = int(negative_mask.sum())
    df = df.loc[~negative_mask].copy()

    before_dedup = len(df)
    df = df.drop_duplicates(keep="first")
    report["duplicate_removed"] = before_dedup - len(df)

    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    logger.info(
        "validate_lob: input=%d output=%d removed=%s",
        original_len, len(df), report,
    )
    return df, report


def validate_news(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"missing_timestamp_removed": 0, "duplicate_removed": 0}
    original_len = len(df)

    missing_mask = df["published_at"].isna() | df["headline"].isna()
    report["missing_timestamp_removed"] = int(missing_mask.sum())
    df = df.loc[~missing_mask].copy()

    before_dedup = len(df)
    df = df.drop_duplicates(keep="first")
    report["duplicate_removed"] = before_dedup - len(df)

    df = df.sort_values("published_at").reset_index(drop=True)
    logger.info(
        "validate_news: input=%d output=%d removed=%s",
        original_len, len(df), report,
    )
    return df, report
