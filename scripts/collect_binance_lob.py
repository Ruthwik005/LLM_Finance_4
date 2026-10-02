"""Live Binance spot top-N order-book collector (self-collected deep books; Binance has no free historical archive).

Samples the latest depth snapshot every --interval-ms and attaches aggregated aggTrade info (VWAP price, total size,
net-aggressor side) for the bucket since the previous snapshot. Writes lob_part_*.parquet to data/raw/lob/.
Modes: websocket (default, needs `websocket-client`) or --mode rest (polling fallback).
Note: Mode B -- this crypto book is NOT causally matched with FNSPID equity news.
"""
from __future__ import annotations
import argparse, json, threading, time, sys
from pathlib import Path
import pandas as pd
import requests
from market_maker.data.binance_live import TradeBucket, build_snapshot_row, rows_to_frame, merge_parts

REST = "https://api.binance.com/api/v3"


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.depth = None
        self.bucket = TradeBucket()
        self.stop = threading.Event()


def ws_thread(state: State, symbol: str, depth: int):
    import websocket
    s = symbol.lower()
    url = f"wss://stream.binance.com:9443/stream?streams={s}@depth{depth}@100ms/{s}@aggTrade"
    backoff = 1.0

    def on_message(ws, raw):
        m = json.loads(raw)
        stream, data = m.get("stream", ""), m.get("data", {})
        with state.lock:
            if "depth" in stream:
                state.depth = data
            elif "aggTrade" in stream:
                state.bucket.add(data)

    while not state.stop.is_set():
        app = websocket.WebSocketApp(url, on_message=on_message, on_open=lambda w: None)
        t0 = time.time()
        try:
            app.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:  # noqa: BLE001
            print(f"[ws] error: {e}", file=sys.stderr)
        if state.stop.is_set():
            break
        backoff = 1.0 if time.time() - t0 > 30 else min(backoff * 2, 30)
        print(f"[ws] disconnected; reconnecting in {backoff:.0f}s", file=sys.stderr)
        state.stop.wait(backoff)


def rest_poll(state: State, symbol: str, depth: int, last: dict):
    r = requests.get(f"{REST}/depth", params={"symbol": symbol, "limit": depth}, timeout=5)
    r.raise_for_status()
    d = r.json()
    params = {"symbol": symbol, "limit": 500}
    if last.get("t"):
        params["startTime"] = last["t"] + 1
    else:
        params["startTime"] = int(time.time() * 1000) - 1000
    tr = requests.get(f"{REST}/aggTrades", params=params, timeout=5)
    tr.raise_for_status()
    with state.lock:
        state.depth = d
        for t in tr.json():
            state.bucket.add(t)
            last["t"] = max(last.get("t", 0), t["T"])


def flush(rows, out_dir: Path, symbol: str, n: int) -> int:
    if not rows:
        return n
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"lob_part_{symbol}_{int(time.time()*1000)}_{n:04d}.parquet"
    rows_to_frame(rows).to_parquet(p, index=False)
    print(f"flushed {len(rows)} rows -> {p}")
    return n + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--depth", type=int, default=5, choices=[5, 10, 20])
    ap.add_argument("--interval-ms", type=int, default=500)
    ap.add_argument("--flush-seconds", type=float, default=60)
    ap.add_argument("--duration-minutes", type=float, default=0, help="0 = run until Ctrl-C")
    ap.add_argument("--mode", choices=["ws", "rest"], default="ws")
    ap.add_argument("--out-dir", default="data/raw/lob")
    ap.add_argument("--merge", action="store_true", help="merge parts into --merge-out after collecting")
    ap.add_argument("--merge-out", default="data/processed/lob.parquet")
    a = ap.parse_args()

    state, out_dir = State(), Path(a.out_dir)
    mode = a.mode
    if mode == "ws":
        try:
            import websocket  # noqa: F401
            threading.Thread(target=ws_thread, args=(state, a.symbol, a.depth), daemon=True).start()
        except ImportError:
            print("websocket-client missing; falling back to REST", file=sys.stderr)
            mode = "rest"
    print(f"collecting {a.symbol} depth={a.depth} every {a.interval_ms}ms mode={mode}")
    rows, n, last, errs = [], 0, {}, 0
    t_end = time.time() + a.duration_minutes * 60 if a.duration_minutes else None
    t_flush = time.time() + a.flush_seconds
    interval = a.interval_ms / 1000.0
    nxt = time.time()
    try:
        while t_end is None or time.time() < t_end:
            nxt += interval
            if mode == "rest":
                try:
                    rest_poll(state, a.symbol, a.depth, last)
                except Exception as e:  # noqa: BLE001
                    errs += 1
                    print(f"[rest] error: {e}", file=sys.stderr)
                    time.sleep(min(2 ** errs, 30))
                    nxt = time.time()
                    continue
                errs = 0
            with state.lock:
                d = state.depth
                row = build_snapshot_row(d, state.bucket, a.symbol, a.depth, pd.Timestamp.now(tz="UTC")) if d else None
            if row:
                rows.append(row)
            if time.time() >= t_flush:
                n = flush(rows, out_dir, a.symbol, n); rows = []
                t_flush = time.time() + a.flush_seconds
            time.sleep(max(0.0, nxt - time.time()))
    except KeyboardInterrupt:
        print("Ctrl-C: flushing")
    finally:
        state.stop.set()
        n = flush(rows, out_dir, a.symbol, n)
    if a.merge:
        _, rep = merge_parts(out_dir, a.merge_out)
        print(f"merged -> {a.merge_out}: {rep}")


if __name__ == "__main__":
    main()
