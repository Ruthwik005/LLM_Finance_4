from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_lob_features(df: pd.DataFrame, depth: int, return_lags: list[int], vol_window: int = 20) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    feats = pd.DataFrame(index=df.index)

    feats["midprice"] = (df["bid_price_1"] + df["ask_price_1"]) / 2
    feats["spread"] = df["ask_price_1"] - df["bid_price_1"]
    feats["relative_spread"] = feats["spread"] / feats["midprice"]

    bid_sz1, ask_sz1 = df["bid_size_1"], df["ask_size_1"]
    feats["imbalance_l1"] = (bid_sz1 - ask_sz1) / (bid_sz1 + ask_sz1 + 1e-9)
    denom = bid_sz1 + ask_sz1 + 1e-9
    feats["microprice"] = (df["bid_price_1"] * ask_sz1 + df["ask_price_1"] * bid_sz1) / denom

    cum_bid, cum_ask = 0.0, 0.0
    for level in range(1, depth + 1):
        bp, bsz = df[f"bid_price_{level}"], df[f"bid_size_{level}"]
        ap, asz = df[f"ask_price_{level}"], df[f"ask_size_{level}"]
        feats[f"bid_price_rel_{level}"] = bp - feats["midprice"]
        feats[f"ask_price_rel_{level}"] = ap - feats["midprice"]
        feats[f"bid_size_log_{level}"] = np.log1p(bsz)
        feats[f"ask_size_log_{level}"] = np.log1p(asz)
        cum_bid = cum_bid + bsz
        cum_ask = cum_ask + asz
    feats["imbalance_cum"] = (cum_bid - cum_ask) / (cum_bid + cum_ask + 1e-9)

    for lag in return_lags:
        feats[f"return_lag_{lag}"] = feats["midprice"].pct_change(lag).fillna(0.0)

    feats["rolling_vol"] = feats["midprice"].pct_change().rolling(vol_window, min_periods=1).std().fillna(0.0)

    if "trade_size" in df.columns:
        feats["rolling_traded_volume"] = df["trade_size"].fillna(0.0).rolling(20, min_periods=1).sum()
    else:
        feats["rolling_traded_volume"] = 0.0

    ts = pd.to_datetime(df["timestamp"], utc=True)
    seconds_in_day = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    feats["time_of_session"] = seconds_in_day / 86400.0

    feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    feats.insert(0, "timestamp", df["timestamp"])
    return feats


def fit_scaler(df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].to_numpy())
    return scaler


def apply_scaler(df: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols].to_numpy())
    return df
