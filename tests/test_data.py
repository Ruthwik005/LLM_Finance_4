import numpy as np
import pandas as pd
from market_maker.data.sample_data import generate_synthetic_lob
from market_maker.data.validators import validate_lob
from market_maker.data.splits import chronological_split


def test_validate_lob_removes_crossed_book_rows():
    df = generate_synthetic_lob(n_rows=50, depth=1, symbol="BTCUSDT", seed=1)
    df.loc[5, "bid_price_1"] = df.loc[5, "ask_price_1"] + 1.0  # force crossed
    cleaned, report = validate_lob(df)
    assert report["crossed_removed"] == 1
    assert (cleaned["bid_price_1"] < cleaned["ask_price_1"]).all()


def test_validate_lob_removes_negative_sizes():
    df = generate_synthetic_lob(n_rows=50, depth=1, symbol="BTCUSDT", seed=1)
    df.loc[3, "bid_size_1"] = -1.0
    cleaned, report = validate_lob(df)
    assert report["negative_size_removed"] == 1
    assert (cleaned["bid_size_1"] >= 0).all()


def test_validate_lob_removes_exact_duplicates():
    df = generate_synthetic_lob(n_rows=50, depth=1, symbol="BTCUSDT", seed=1)
    dup_row = df.iloc[[0]]
    df = pd.concat([df, dup_row], ignore_index=True)
    cleaned, report = validate_lob(df)
    assert report["duplicate_removed"] == 1
    assert len(cleaned) == 50


def test_chronological_split_has_no_overlap_and_is_ordered():
    df = generate_synthetic_lob(n_rows=100_000, depth=1, symbol="BTCUSDT", seed=1)
    splits = chronological_split(df, "timestamp", 0.70, 0.15, 0.15)
    assert splits["train"]["timestamp"].max() <= splits["validation"]["timestamp"].min()
    assert splits["validation"]["timestamp"].max() <= splits["test"]["timestamp"].min()
    total = len(splits["train"]) + len(splits["validation"]) + len(splits["test"])
    assert total == len(df)
