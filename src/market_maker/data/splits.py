from __future__ import annotations
import pandas as pd


def chronological_split(
    df: pd.DataFrame, timestamp_col: str, train: float, validation: float, test: float
) -> dict[str, pd.DataFrame]:
    assert abs(train + validation + test - 1.0) < 1e-6, "split fractions must sum to 1"
    df = df.sort_values(timestamp_col, kind="stable").reset_index(drop=True)
    days = pd.to_datetime(df[timestamp_col]).dt.date
    unique_days = sorted(days.unique())
    n_days = len(unique_days)
    n_train = max(1, int(n_days * train))
    n_val = max(1, int(n_days * validation))
    train_days = set(unique_days[:n_train])
    val_days = set(unique_days[n_train:n_train + n_val])
    test_days = set(unique_days[n_train + n_val:])
    return {
        "train": df.loc[days.isin(train_days)].reset_index(drop=True),
        "validation": df.loc[days.isin(val_days)].reset_index(drop=True),
        "test": df.loc[days.isin(test_days)].reset_index(drop=True),
    }
