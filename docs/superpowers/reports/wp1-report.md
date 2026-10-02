# WP1 report: simulator realism + trader metrics

## Changes
- Env observation = `[market_features | private_state]`; `PRIVATE_STATE_DIM = 6` (`PRIVATE_STATE_NAMES` in `env/market_making_env.py`):
  `[inventory/max_inv (clip -1..1), tanh(mtm/pnl_scale), remaining_frac, last_bid_dist, last_ask_dist, last_spread]` (distances in ticks / max_spread_ticks). Private block is `obs[-6:]`.
- Quote persistence per (side, level); unchanged price + desired >= remaining keeps remaining volume (no cancel); changed price / reduced volume = cancel remaining + place new. Partial fills reduce remaining. Reward cancellation penalty uses the new canceled volume.
- Latency: `simulator.latency_steps`; new/changed orders become fill-eligible after N steps.
- `simulator.rebate_per_unit` (default 0.0), `simulator.pnl_scale` (50.0); `AccountState.rebates`. New config sections `evaluation` (adverse_horizon_steps, adverse_threshold_ticks) and `baselines` (gamma, kappa, volatility_window); `compute_lob_features(..., vol_window=20)`.
- Step log `info["step_log"]`: contract fields plus `next_mid` (mid after the step, the fill-time mid); each fill also has `mid` (= next_mid). `t` is the 0-based step in the episode; `timestamp` ISO string of the raw row at quote time. Logged `bid/ask_volumes` = outstanding volume after reconciliation, before fills.
- `run_backtest` returns `steps`, and `orders` (per-step placed/canceled/outstanding). Supports AS via `is_heuristic=True`.
- `summarize_run(backtest_result, config)` (config = AppConfig or dict). Keys: mean_pnl, median_pnl, std_pnl, map, total_pnl, max_abs_inventory, pnl_over_map*, sharpe_like*, n_fills, fees_paid, rebates_received, spread_capture_pnl, inventory_pnl, max_drawdown, cancel_ratio*, fill_rate*, avg_quoted_spread*, avg_quote_distance*, adverse_selection_ratio* (*omitted when undefined). Spread/distance in ticks; attribution values are totals over all episodes; max_drawdown = worst per-episode drawdown of mtm curve.
  Deviation from contract text: `inventory_pnl = total_pnl - spread_capture + fees_paid - rebates_received` (sign fixed so the decomposition is exact; identical when fees/rebates are 0).
- `AvellanedaStoikovMarketMaker` (`agents/avellaneda_stoikov.py`): needs `rolling_vol_index`, `midprice_index` (feature columns), tick/grid params, `episode_length_steps`; optional `feature_mean/feature_scale` to un-standardise features (WP2 must pass scaler params if features are scaled). Remaining time is read from `obs[-4]`.
- `utils/io.file_sha256`, `evaluation/run_metadata.build_run_metadata`, `evaluation/plots.write_plots`; `write_run_artifacts(..., training_curve=None)` writes orders.csv, quotes.csv, training_curve.csv, plots/{equity_curve,episode_pnl,inventory,adverse_selection}.png.
- DB: `steps` table + index `ix_steps_run_episode`; `write_run_to_db` bulk-inserts steps with replace-on-rerun. `init_db` creates it; existing DB files get it on next `init_db`.
- `scripts/evaluate.py`: `--variant HEURISTIC|AVELLANEDA_STOIKOV` (SAC needs `--model-path`); `train_agent.py` passes rebate/latency/pnl_scale.

## Tests
No existing test needed changes. 17 new tests in `tests/test_wp1.py`; full suite 88 passed. Smoke runs of HEURISTIC and AVELLANEDA_STOIKOV on fast_demo produced all artifacts and 1800 steps rows each (smoke runs and DB rows deleted afterwards).

## Notes
- On synthetic data the quotes follow the mid every snapshot, so cancel_ratio is ~0.95+; spread_capture is negative because fills are priced against the next snapshot's mid. These are properties of the snapshot approximation, not tuned.
- AS default kappa=100 (per price unit) was chosen so the spread is a few ticks at tick 0.01; retune for other price scales.

## Fixes (review follow-up)

- Quote persistence now tops up: same price with desired > remaining raises the slot to the desired volume and counts the difference as `placed_volume` (top-ups get no queue-priority modeling; the original order's latency/eligibility is kept). desired < remaining cancels only the difference (counted as canceled) and keeps the order. Desired volumes remain clamped by current inventory, so per-side outstanding never exceeds the inventory headroom.
- Step logs carry `fees_paid` and `rebates_received` from the env's per-fill accounting; `summarize_run` sums them and falls back to config rates only for older logs lacking the fields.
- `vol_window=cfg.baselines.volatility_window` is passed in `train_agent.py` and `train_transformer.py` (`evaluate.py` already did).
- `observation_space` has real bounds on the private block; market features stay unbounded.
- New tests: inventory fuzz (random + adversarial, latency 0/1), volume conservation, PnL decomposition cross-check with an independent inventory-PnL computation, top-up, latency + persistence.
