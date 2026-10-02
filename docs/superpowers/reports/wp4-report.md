# WP4 Report: Real-data tooling

Data mode B: real Binance crypto order book + real FNSPID equity news, validated separately. They are NOT causally matched.

## Commands
- Collect live (ws; `--mode rest` fallback): `python scripts/collect_binance_lob.py --symbol BTCUSDT --depth 5 --interval-ms 500 --duration-minutes 30 --merge --merge-out data/processed/lob_binance.parquet` (or `make collect-lob`; merge alone: `scripts/merge_lob_parts.py`).
- Public daily data (depth 1 only): `python scripts/download_binance_public.py --symbol BTCUSDT --start 2023-06-01 --end 2023-06-01 --kinds bookTicker,aggTrades --trades-market futures/um --max-rows 1000000`.
- News: `python scripts/prepare_fnspid.py --tickers AAPL --max-rows 5000` -> `data/raw/news/fnspid_AAPL.parquet`.
- Arbitrary CSV/Parquet: `market_maker.data.adapters.load_lob_with_mapping(path, mapping.yaml)` / `normalize_news`.
- Pipeline: `python scripts/prepare_data.py --config configs/real_binance.yaml`.

## Smoke tests (real)
- Live collector, websocket mode, ~20s: 40 rows (2 parts), merged and validated (0 removed), depth 5, trades attached.
- Public: BTCUSDT 2023-06-01 bookTicker + futures aggTrades, first 1M raw rows -> 10,959 depth-1 rows at 500ms (00:00-01:31 UTC); `compute_lob_features` with depth=1 works (no NaN); `validate_lob` clean.
- FNSPID AAPL: 441 rows (2020-03-09..2020-06-10) from the first ~5MB of the stream. Columns: Date, Article_title, Stock_symbol, Url, Publisher, Author, Article, Lsa/Luhn/Textrank/Lexrank_summary (we use the first six).

## Known limits
- Public Binance data is depth 1 only (bookTicker); deep books must be self-collected. Use `data.depth: 1` for it.
- Tick size is per pair (0.01 for BTCUSDT spot; check exchangeInfo for others). Futures bookTicker vs spot aggTrades are different venues, so use futures-um trades when joining.
- Collector timestamps are local receive time (depth5 stream has no event time); sampled at the interval.
- FNSPID: Stock_news/All_external.csv (5.7GB) and nasdaq_exteral_data.csv (23GB); 2020 coverage seen at the start of AAPL; the file appears ticker-grouped, not date-sorted. The script streams over HTTP and stops at `--max-rows`/`--max-scan-mb` (`--sorted-stop` is opt-in). Dates outside the crypto collection window never overlap (mismatch is intended in Mode B). License CC BY-NC 4.0.
- Smoke-testing overwrote `data/processed/lob.parquet` with 40 live rows; it was regenerated via `make sample-data` (synthetic).

## Manual steps
Run the collector for long enough (hours) yourself; decide ticker list/date range for news; set `configs/real_binance.yaml` SAC budget.
