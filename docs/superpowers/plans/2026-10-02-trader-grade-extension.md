# Trader-Grade Extension Plan

Goal: close the gaps found in the post-build audit. Everything is implemented and verified with tiny smoke runs;
long real training/data download is left to the user via `docs/RUNBOOK.md`.

## Work packages (sequential)
1. **WP1 Simulator realism + metrics**: private state in observation, quote persistence (cancel only changes), latency, fees/rebates,
   per-step quote log, full trader metrics, PnL attribution, Avellaneda-Stoikov baseline, rich run metadata, plots, training_curve/orders files.
2. **WP2 Training pipeline (code only, smoke-tested)**: feature builder (scaler fit on train only, FinBERT sentiment, as-of alignment,
   frozen Transformer embeddings/probs, fusion per variant), config-driven hyperparameters, SAC with validation-based best checkpoint,
   multi-seed 4-variant experiment runner, aggregation (mean +- std).
3. **WP3 API + UI**: steps/quotes tables, replay endpoint, Pydantic schemas, Episode Replay screen, richer Behavior/Signals/Compare screens.
4. **WP4 Real-data tooling**: Binance order-book collector, Tardis free-day downloader, generic CSV/Parquet adapter with column mapping, FNSPID loader.
5. **WP5 Report + docs + pipeline**: narrative report sections, run_all update, README, BUILD_STATUS, beginner RUNBOOK.

## Shared contracts (all packages must follow)
- Variants: `HEURISTIC`, `AVELLANEDA_STOIKOV` (baselines), `SAC_LOB_ONLY`, `SAC_LOB_TRANSFORMER`, `SAC_LOB_SENTIMENT`, `SAC_FULL`.
- Run id for learned runs: `<variant>_seed<N>` (lowercase variant), model: `artifacts/rl/<variant>/seed<N>/best_model.zip`.
- Feature artifacts: `artifacts/features/lob_<split>.parquet` (validated, time-sorted raw LOB rows) and
  `artifacts/features/<variant>_<split>.npy` (float32, row i aligned to lob row i) + `artifacts/features/<variant>_columns.json`,
  `artifacts/features/scaler.joblib` (fit on train only). Splits: `train`, `validation`, `test`.
- Env observation = `[market_features | private_state]`; private_state always appended by the env (inventory fraction, scaled mtm pnl,
  remaining-episode fraction, last quoted bid/ask distance in ticks normalized, last quoted spread normalized).
- Per-step log dict (env `info["step_log"]`): `t, timestamp, mid, bid_prices[2], ask_prices[2], bid_volumes[2], ask_volumes[2],
  inventory, cash, mtm_pnl, fills[{side,price,volume}], placed_volume, canceled_volume`.
- `run_backtest` result adds `steps`: list (per episode) of lists of step-log dicts. DB table `steps` stores them (run_id, episode, step, ...).
- Metrics keys (flat floats in `metrics`): mean_pnl, median_pnl, std_pnl, map, max_abs_inventory, max_drawdown, pnl_over_map, sharpe_like,
  n_fills, fill_rate, cancel_ratio, avg_quoted_spread, avg_quote_distance, adverse_selection_ratio, spread_capture_pnl, inventory_pnl, fees_paid.
  Undefined values (e.g. PnL/MAP with MAP=0) are omitted or null, never inf/NaN.
- Everything tunable lives in YAML config (extend `AppConfig`); no hardcoded hyperparameters.
- Honesty rules from the spec still apply (no fabricated results; synthetic data labeled; fills are approximated).
