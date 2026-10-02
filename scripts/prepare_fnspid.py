"""Stream FNSPID news (HF dataset Zihan1004/FNSPID) filtered by ticker/date into a normalized news parquet.

Inspected repo layout (HF API): Stock_news/All_external.csv (5.7GB), Stock_news/nasdaq_exteral_data.csv (23GB, adds a
leading 'Unnamed: 0' column). Columns: Date ('YYYY-MM-DD HH:MM:SS UTC'), Article_title, Stock_symbol, Url, Publisher,
Author, Article, *_summary. Ordering is not guaranteed, so the stream is read over HTTP in chunks and stopped by
--max-rows or --max-scan-mb (opt-in --sorted-stop for ticker-grouped files).
Output: data/raw/news/fnspid_<tickers>.parquet with symbol = the news ticker (equity!).
Mode B note: equity news vs crypto order book are NOT causally matched; sentiment is validated separately.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import requests
from market_maker.data.adapters import normalize_news
from market_maker.data.validators import validate_news

HF = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/{f}"
MAPPING = {"published_at": {"column": "Date", "unit": "iso"}, "headline": "Article_title", "symbol": "Stock_symbol",
           "body": "Article", "source": "Publisher", "url_or_id": "Url"}


def filter_chunk(chunk: pd.DataFrame, tickers: list[str], start, end) -> pd.DataFrame:
    c = chunk[chunk["Stock_symbol"].isin(tickers)]
    if c.empty:
        return c
    ts = pd.to_datetime(c["Date"], utc=True, errors="coerce")
    m = ts.notna()
    if start is not None:
        m &= ts >= start
    if end is not None:
        m &= ts <= end
    return c[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True, help="comma list, e.g. AAPL,MSFT")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--file", default="All_external.csv")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--max-scan-mb", type=int, default=3000)
    ap.add_argument("--sorted-stop", action="store_true", help="stop once past tickers alphabetically (only if file is grouped by ticker; NOT verified)")
    ap.add_argument("--chunksize", type=int, default=20000)
    ap.add_argument("--body-chars", type=int, default=2000)
    a = ap.parse_args()
    tickers = [t.strip().upper() for t in a.tickers.split(",")]
    start = pd.Timestamp(a.start, tz="UTC") if a.start else None
    end = pd.Timestamp(a.end, tz="UTC") if a.end else None
    cols = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher", "Article"]
    parts, n, seen = [], 0, False
    with requests.get(HF.format(f=a.file), stream=True, timeout=60) as r:
        r.raise_for_status()
        counter = {"b": 0}

        class Counting:
            def read(self, k=-1):
                b = r.raw.read(k)
                counter["b"] += len(b)
                return b
        r.raw.decode_content = True
        for chunk in pd.read_csv(Counting(), usecols=cols, chunksize=a.chunksize, dtype=str):
            hit = filter_chunk(chunk, tickers, start, end)
            if len(hit):
                parts.append(hit); n += len(hit)
            in_range = chunk["Stock_symbol"].isin(tickers).any()
            seen = seen or in_range
            past = seen and not in_range and (chunk["Stock_symbol"].dropna() > max(tickers)).all()
            print(f"scanned {counter['b']/1e6:.0f}MB matched={n}", flush=True)
            if (a.max_rows and n >= a.max_rows) or counter["b"] > a.max_scan_mb * 1e6 or (a.sorted_stop and past):
                break
    if not parts:
        raise SystemExit("No matching rows found (try larger --max-scan-mb or different tickers/dates)")
    df = pd.concat(parts, ignore_index=True)
    if a.max_rows:
        df = df.head(a.max_rows)
    df["Article"] = df["Article"].str.slice(0, a.body_chars)
    news = normalize_news(df, MAPPING)
    news, rep = validate_news(news)
    out = Path(f"data/raw/news/fnspid_{'_'.join(tickers)}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    news.to_parquet(out, index=False)
    print(f"source columns: {cols}; wrote {out}: rows={len(news)} range={news.published_at.min()}..{news.published_at.max()} {rep}")


if __name__ == "__main__":
    main()
