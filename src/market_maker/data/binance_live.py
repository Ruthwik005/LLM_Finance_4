"""Pure helpers for the Binance live collector (no network)."""
from __future__ import annotations
import glob
from pathlib import Path
from typing import Optional
import pandas as pd
from market_maker.data.validators import validate_lob


def depth_to_row(payload: dict, symbol: str, depth: int, ts: pd.Timestamp) -> Optional[dict]:
    """Convert a Binance partial-book payload ({bids:[[p,q]..], asks:[..]}) into a normalized row dict."""
    bids, asks = payload.get("bids") or [], payload.get("asks") or []
    if len(bids) < depth or len(asks) < depth:
        return None
    row = {"timestamp": ts, "symbol": symbol}
    for i in range(depth):
        row[f"bid_price_{i+1}"] = float(bids[i][0])
        row[f"bid_size_{i+1}"] = float(bids[i][1])
        row[f"ask_price_{i+1}"] = float(asks[i][0])
        row[f"ask_size_{i+1}"] = float(asks[i][1])
    return row


class TradeBucket:
    """Aggregates aggTrade messages between snapshots."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._notional = 0.0
        self._size = 0.0
        self._signed = 0.0

    def add(self, msg: dict) -> None:
        p, q = float(msg["p"]), float(msg["q"])
        self._notional += p * q
        self._size += q
        # m == True: buyer is the maker, so the aggressor sold.
        self._signed += -q if msg.get("m") else q

    def flush(self) -> dict:
        if self._size <= 0:
            out = {"trade_price": None, "trade_size": None, "trade_side": None}
        else:
            side = "buy" if self._signed > 0 else "sell" if self._signed < 0 else "unknown"
            out = {"trade_price": self._notional / self._size, "trade_size": self._size, "trade_side": side}
        self.reset()
        return out


def build_snapshot_row(depth_payload: dict, bucket: TradeBucket, symbol: str, depth: int, ts: pd.Timestamp) -> Optional[dict]:
    row = depth_to_row(depth_payload, symbol, depth, ts)
    if row is None:
        return None
    row.update(bucket.flush())
    return row


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def merge_parts(parts_dir: str | Path, out_path: str | Path) -> tuple[pd.DataFrame, dict]:
    files = sorted(glob.glob(str(Path(parts_dir) / "lob_part_*.parquet")))
    if not files:
        raise FileNotFoundError(f"No lob_part_*.parquet in {parts_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df, report = validate_lob(df)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    report["rows"] = len(df)
    report["parts"] = len(files)
    return df, report
