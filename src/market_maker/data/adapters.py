"""Generic adapters converting arbitrary CSV/Parquet snapshot data into the normalized schemas.

LOB mapping (dict or YAML):
    timestamp: {column: ts, unit: ms|us|ns|s|iso}
    bid_price: "bp{i}"   # per-level templates, {i} in 1..depth
    bid_size:  "bq{i}"
    ask_price: "ap{i}"
    ask_size:  "aq{i}"
    trade_price / trade_size / trade_side: optional column names
    symbol_column: optional (otherwise the `symbol` argument is used)
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
import yaml

_UNITS = {"ms", "us", "ns", "s"}


def parse_timestamps(series: pd.Series, unit: str) -> pd.Series:
    if unit == "iso":
        return pd.to_datetime(series, utc=True, errors="coerce")
    if unit not in _UNITS:
        raise ValueError(f"Unknown timestamp unit '{unit}'; use one of {sorted(_UNITS | {'iso'})}")
    return pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit=unit, utc=True)


def _ts_spec(spec: Any) -> tuple[str, str]:
    if isinstance(spec, str):
        return spec, "iso"
    if not isinstance(spec, dict) or "column" not in spec:
        raise ValueError("mapping['timestamp'] must be a column name or {column, unit}")
    return spec["column"], spec.get("unit", "iso")


def _need(df: pd.DataFrame, cols: list[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what}: missing source columns {missing}; available: {list(df.columns)[:30]}")


def normalize_lob(df: pd.DataFrame, mapping: dict, symbol: str, depth: int) -> pd.DataFrame:
    if "timestamp" not in mapping:
        raise ValueError("mapping must contain 'timestamp'")
    ts_col, unit = _ts_spec(mapping["timestamp"])
    out = pd.DataFrame()
    _need(df, [ts_col], "timestamp")
    out["timestamp"] = parse_timestamps(df[ts_col], unit)
    sym_col = mapping.get("symbol_column")
    if sym_col:
        _need(df, [sym_col], "symbol")
        out["symbol"] = df[sym_col].astype(str)
    else:
        out["symbol"] = symbol
    for key in ("bid_price", "bid_size", "ask_price", "ask_size"):
        tmpl = mapping.get(key)
        if not tmpl:
            raise ValueError(f"mapping must contain '{key}' template, e.g. '{key}_{{i}}'")
        for i in range(1, depth + 1):
            src = tmpl.format(i=i)
            _need(df, [src], f"level {i} ({key})")
            out[f"{key}_{i}"] = pd.to_numeric(df[src], errors="coerce").astype(float)
    for key in ("trade_price", "trade_size", "trade_side"):
        src = mapping.get(key)
        if src:
            _need(df, [src], key)
            out[key] = df[src].values
    for key in ("trade_price", "trade_size"):
        if key in out:
            out[key] = pd.to_numeric(out[key], errors="coerce")
    if "trade_side" in out:
        s = out["trade_side"].astype(str).str.lower().replace(
            {"b": "buy", "s": "sell", "bid": "buy", "ask": "sell", "1": "buy", "-1": "sell"})
        out["trade_side"] = s.where(s.isin(["buy", "sell"]), "unknown")
    out = out.dropna(subset=["timestamp"]).dropna(
        subset=[f"{k}_{i}" for k in ("bid_price", "bid_size", "ask_price", "ask_size") for i in range(1, depth + 1)])
    return out.sort_values("timestamp").reset_index(drop=True)


def _read_any(path: str | Path) -> pd.DataFrame:
    p = str(path)
    return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)


def load_mapping(mapping_yaml: str | Path | dict) -> dict:
    if isinstance(mapping_yaml, dict):
        return mapping_yaml
    with open(mapping_yaml) as f:
        return yaml.safe_load(f)


def load_lob_with_mapping(path: str | Path, mapping_yaml: str | Path | dict) -> pd.DataFrame:
    m = load_mapping(mapping_yaml)
    return normalize_lob(_read_any(path), m["lob"] if "lob" in m else m,
                         m.get("symbol", "UNKNOWN"), int(m.get("depth", 5)))


def normalize_news(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """mapping: published_at: {column, unit}|column, headline: col, optional symbol/body/source/url_or_id columns."""
    ts_col, unit = _ts_spec(mapping.get("published_at", mapping.get("timestamp")))
    if "headline" not in mapping:
        raise ValueError("news mapping must contain 'headline'")
    _need(df, [ts_col, mapping["headline"]], "news")
    out = pd.DataFrame({
        "published_at": parse_timestamps(df[ts_col], unit),
        "headline": df[mapping["headline"]],
    })
    for key in ("symbol", "body", "source", "url_or_id"):
        src = mapping.get(key)
        if src and src in df.columns:
            out[key] = df[src].values
        else:
            out[key] = mapping.get(f"default_{key}")
    out = out.dropna(subset=["published_at", "headline"])
    out = out[["published_at", "symbol", "headline", "body", "source", "url_or_id"]]
    return out.sort_values("published_at").reset_index(drop=True)
