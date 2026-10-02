# WP3 report: read-only API + UI

## Backend (`src/market_maker/api/`)
All endpoints are GET, no auth, Pydantic response models in `api/schemas.py` (existing JSON shapes unchanged).
Env vars: `MARKET_MAKER_DB_PATH` (default `artifacts/market_maker.db`), `MARKET_MAKER_ARTIFACTS_ROOT` (default `artifacts`, read per request).

| Endpoint | Notes |
|---|---|
| `/health`, `/runs`, `/runs/{id}`, `/runs/{id}/metrics`, `/episodes`, `/fills`, `/equity-curve` | unchanged shapes |
| `/runs/summary` | one row per run: variant, seed, data_mode, created_at, mean_pnl, map, max_drawdown, n_fills, sharpe_like, pnl_over_map, full `metrics` dict |
| `/runs/{id}/episodes/{ep}/replay?max_points=2000` | steps (time-ordered) + fills (side, price, volume, step, timestamp) + `summary` computed over ALL steps. Stride downsampling always keeps fill steps and the last step. 404 unknown run/episode; empty valid payload for runs without steps |
| `/runs/{id}/training-curve` | parses `evaluation/<id>/training_curve.csv` (new 9-column and old `step,reward`), NaN -> null, missing file -> `points: []` |
| `/experiments`, `/experiments/{id}` | from `experiment_aggregates` + `<id>_summary.json`; per-variant mean/std/n_seeds, pairwise comparisons vs the reference variant with the "exploratory, small n" label and note carried through (NaN -> null) |
| `/runs/{id}/pnl-attribution` | spread_capture_pnl, inventory_pnl, fees_paid, rebates, total (null when undefined) |
| `/signals?split=test&max_points=2000` | strided series from `features/lob_<split>.parquet`, `sac_full_<split>.npy`, `sac_full_columns.json`; sentiment returned raw (from `sentiment_raw_<split>.npy`), transformer probs recovered by inverting the train-split standardisation (`extras_scaler.json`); else `{"available": false, "reason": ...}`. Nothing is invented |

Schema change needed for replay fills: `fills.step` (nullable int). `run_backtest` now tags each fill with its step, `write_run_to_db` stores it, `init_db` adds the column to existing DB files (`ALTER TABLE`). Runs stored before this have `step = null` (fills returned without markers position).

## Frontend (`frontend/src`)
New nav items: **Episode Replay** (flagship) and **Training**; Behavior, Model Comparison and Sentiment & Signals extended.
- Replay: run (top bar) + episode + compare-with (defaults to `heuristic`), ComposedChart with mid, L1/L2 bid/ask step lines (L2 lighter), buy/sell triangle markers sized by volume, synchronized inventory (split area) + cumulative MtM PnL panel (dual axis, dashed comparison overlay), play/pause/speed/scrubber with progressive reveal (autoplay disabled under prefers-reduced-motion), zoom window sliders, layer toggles, hover readout, KPI strip with delta-vs-comparison. Large episodes rely on API downsampling.
- Behavior: cancel ratio, fill rate, avg quoted spread, avg quote distance, adverse-selection KPIs ("—" when undefined) and a stacked PnL attribution bar.
- Training: train vs validation reward, val PnL, val MAP, best-checkpoint marker, entropy/critic/actor small multiples; empty state when no curve.
- Model Comparison: experiment table (mean ± std, n_seeds, best-per-metric highlight), error-bar chart, exploratory pairwise table with the note "Exploratory — small number of seeds; not a claim of statistical significance."; per-run comparison kept below / as fallback.
- Signals: real charts (midprice + sentiment, article counts, stacked transformer probabilities, signal-invalid shading) when `/signals` is available, otherwise the previous honest placeholders plus the API's reason. Status copy updated: features feed only the SENTIMENT/TRANSFORMER/FULL variants.
- Honesty rules unchanged (synthetic runs show only the smoke-test chip). Charts use `components/AutoSize.jsx`.
- Lib helpers (unit tested): `lib/replay.js`, `experiments.js`, `signals.js`, `training.js`, `attribution.js`.

## Demo data (synthetic, `data_mode=synthetic_smoke_test`)
Old demo DB/artifacts deleted and regenerated: `sample-data` equivalent, `build_features.py --stub-sentiment`, `train_transformer.py`, `build_features.py --stage fusion --stub-sentiment`, then
`run_experiments.py --config configs/fast_demo.yaml --fast --variants SAC_LOB_ONLY SAC_LOB_TRANSFORMER SAC_LOB_SENTIMENT SAC_FULL --seeds 0 1 2 --data-mode synthetic_smoke_test` (about 25 s). Run ids: `heuristic`, `avellaneda_stoikov`, `sac_<variant>_seed<N>`; experiment `fast_demo`. Stub sentiment carries no information; all numbers are pipeline checks.

## Launch
```bash
source .venv/bin/activate
uvicorn market_maker.api.main:app --port 8765          # ports 8000/8001 may be in use
cd frontend && VITE_API_URL=http://localhost:8765 npm run dev   # or put it in frontend/.env.local
```

## Tests
`tests/test_api_wp3.py` (shapes, replay downsampling/404/empty, training curve new+old, experiments, signals unavailable/available, summary, attribution); frontend Vitest: new lib tests + smoke renders of every view with data, with `/signals` unavailable, and with all requests failing.
