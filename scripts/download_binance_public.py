"""Download Binance public daily data (data.binance.vision) and convert to the normalized LOB schema.

*** IMPORTANT: public data gives DEPTH = 1 ONLY (best bid/ask from futures-um bookTicker). ***
Binance publishes no free multi-level historical order book; use scripts/collect_binance_lob.py for depth 5+.
Use configs with `data.depth: 1` for this output.

  bookTicker  (futures/um) -> depth-1 LOB, downsampled to the last update per --interval-ms bucket
  aggTrades   (spot, or futures-um with --trades-market) -> trade columns (VWAP, size, net side) joined by as-of bucket
                 to the nearest prior snapshot when bookTicker is also requested; otherwise a trades-only parquet.
Files: data/raw/lob/binance_public/{raw zips, csv} and binance_<SYMBOL>_<kind>_<start>_<end>.parquet.
Mode B note: this is crypto data, NOT causally matched with FNSPID equity news. bookTicker daily zips are ~100-300MB;
use --max-rows to stop early.
"""
from __future__ import annotations
import argparse, zipfile
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import requests
from market_maker.data.binance_public import book_ticker_to_lob, agg_trades_to_trades, join_trades_to_lob
from market_maker.data.validators import validate_lob

BASE = "https://data.binance.vision/data"


def url_for(kind: str, market: str, symbol: str, day: str) -> str:
    return f"{BASE}/{market}/daily/{kind}/{symbol}/{symbol}-{kind}-{day}.zip"


def fetch(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


def read_zip(zpath: Path, max_rows: int | None) -> pd.DataFrame:
    with zipfile.ZipFile(zpath) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            head = f.readline().decode()
        has_header = head[:1].isalpha()
        with z.open(name) as f:
            return pd.read_csv(f, header=0 if has_header else None, nrows=max_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--kinds", default="bookTicker", help="comma list: bookTicker,aggTrades")
    ap.add_argument("--trades-market", default="spot", choices=["spot", "futures/um"])
    ap.add_argument("--interval-ms", type=int, default=500)
    ap.add_argument("--max-rows", type=int, default=None, help="max raw rows read per day-file")
    ap.add_argument("--out-dir", default="data/raw/lob/binance_public")
    a = ap.parse_args()
    out = Path(a.out_dir)
    kinds = [k.strip() for k in a.kinds.split(",")]
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    days = [(d0 + timedelta(i)).isoformat() for i in range((d1 - d0).days + 1)]
    lobs, trs = [], []
    for day in days:
        if "bookTicker" in kinds:
            z = fetch(url_for("bookTicker", "futures/um", a.symbol, day), out / f"{a.symbol}-bookTicker-{day}.zip")
            lobs.append(book_ticker_to_lob(read_zip(z, a.max_rows), a.symbol, a.interval_ms))
        if "aggTrades" in kinds:
            z = fetch(url_for("aggTrades", a.trades_market, a.symbol, day), out / f"{a.symbol}-aggTrades-{day}.zip")
            trs.append(agg_trades_to_trades(read_zip(z, a.max_rows), a.symbol))
    tag = f"{a.symbol}_{'_'.join(kinds)}_{a.start}_{a.end}"
    if lobs:
        lob = pd.concat(lobs, ignore_index=True)
        if trs:
            lob = join_trades_to_lob(lob, pd.concat(trs, ignore_index=True), a.interval_ms)
        lob, rep = validate_lob(lob)
        p = out / f"binance_{tag}.parquet"
        lob.to_parquet(p, index=False)
        print(f"wrote {p}: rows={len(lob)} depth=1 {rep}")
    elif trs:
        t = pd.concat(trs, ignore_index=True)
        p = out / f"binance_{tag}_trades.parquet"
        t.to_parquet(p, index=False)
        print(f"wrote {p}: trades-only rows={len(t)}")


if __name__ == "__main__":
    main()
