# Build Status

Last updated at the end of work package 5 (report, orchestration, documentation).

## Completed

- **Foundation (original build):** repo skeleton, YAML config (Pydantic), schemas, LOB/news loaders and validators, chronological split, synthetic data generator, LOB features, FinBERT scoring + cache, leak-free as-of news alignment, Transformer encoder, fusion, Gymnasium env, accounting, reward, SAC, heuristic baseline, metrics, SQLite writer, FastAPI, React dashboard.
- **WP1 simulator realism + metrics:** private state in the observation, quote persistence, latency, fees/rebates, per-step logs, full trader metrics, PnL attribution, Avellaneda-Stoikov baseline, rich run metadata, plots.
- **WP2 training pipeline:** scalable feature builder (scaler fit on train only, hashes for cache validity), Transformer training/embeddings, SAC with validation-based best checkpoint, multi-seed 4-variant experiment runner with resume, mean +- std aggregation and exploratory Welch tests.
- **WP3 API + UI:** replay, training-curve, experiments, signals and attribution endpoints; Episode Replay, Training, Behavior, Model Comparison, Signals screens.
- **WP4 real-data tooling:** Binance live order-book collector, public Binance downloader (depth 1), generic CSV/Parquet adapter, FNSPID streamer.
- **WP5:**
  - `FINAL_PROJECT_REPORT.md` generator with all 24 spec sections, written from the DB and run artifacts (`src/market_maker/evaluation/report.py`, `report_text.py`, `scripts/generate_report.py`), figures in `artifacts/reports/figures/`.
  - `scripts/run_all.py` rewritten for the new pipeline (`src/market_maker/orchestration.py`): `--fast --stub-sentiment --from-stage --only-stage --dry-run --resume`, news-optional mode, per-stage timing, fail-fast.
  - `scripts/launch_demo.py` (`make demo`), updated `Makefile`.
  - README, `docs/SPEC_COMPLIANCE.md`, `docs/RUNBOOK.md`.

## Current limitations (honest list)

- **No real-data result exists yet.** The shipped demo artifacts are synthetic (`synthetic_smoke_test`); they prove the software runs, not that anything works financially. The real collection, long training and final report are the user's steps (`docs/RUNBOOK.md`).
- Fills/queue priority are snapshot approximations. News and the crypto book are not matched in Mode B.
- The dashboard is React, not Streamlit. TD3, IMM imitation learning, stable reference price, Docker, MLflow are not built.
- `metrics.csv` is not written per run (`metrics.json` is). Episodes are not guarded against gaps in the collected data.
- `configs/real_binance.yaml` has placeholder budgets and unit sizes (`total_volume_per_side: 20`, `max_inventory: 100`, `total_timesteps: 200000`) that must be set for the instrument.
- An experiment's data mode is inferred from the config file name (`fast_demo.yaml` = synthetic) unless `--data-mode` is passed; `run_all.py --stub-sentiment` forces `synthetic_smoke_test`.

## Run commands

```bash
# setup
make install && source .venv/bin/activate && make install-frontend
pytest -q

# synthetic demo (about a minute) and dashboard
make demo-data
make demo                                   # API_PORT=8765 make demo if 8000 is busy

# whole pipeline
python scripts/run_all.py --config configs/fast_demo.yaml --fast --stub-sentiment
python scripts/run_all.py --config configs/real_binance.yaml --dry-run
python scripts/run_all.py --config configs/real_binance.yaml --resume

# single steps
make features CONFIG=configs/real_binance.yaml
make train-transformer CONFIG=configs/real_binance.yaml
make experiments CONFIG=configs/real_binance.yaml EXP=real_v1
make report

# data
python scripts/collect_binance_lob.py --duration-minutes 240 --out-dir data/raw/lob/btcusdt_run1
python scripts/merge_lob_parts.py --parts-dir data/raw/lob/btcusdt_run1 --out data/processed/lob_binance.parquet
python scripts/prepare_fnspid.py --tickers AAPL --max-rows 5000
```

## Endpoints and files

- API docs: http://localhost:8000/docs (or your `API_PORT`); dashboard: http://localhost:5173
- SQLite: `artifacts/market_maker.db`; report: `artifacts/reports/FINAL_PROJECT_REPORT.md`
- Spec compliance table: `docs/SPEC_COMPLIANCE.md`; beginner guide: `docs/RUNBOOK.md`
