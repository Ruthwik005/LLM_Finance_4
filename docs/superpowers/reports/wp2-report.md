# WP2 report: training pipeline (features -> Transformer -> SAC -> multi-seed experiment)

Code + smoke tests only. The long real training is run by the user. All numbers below were measured on this
machine (Apple Silicon laptop, CPU torch) with synthetic data; real-data values will differ.

## Architecture

```
raw LOB (+ news) --build_features(base)--> artifacts/features/{lob_<split>.parquet, lob_features_<split>.npy,
                                            sentiment_raw_<split>.npy, scaler.joblib, feature_meta.json}
                 --train_transformer-----> artifacts/transformer/{checkpoint.pt, training_curve.csv}
                 --build_features(fusion)-> artifacts/features/<variant>_<split>.npy + <variant>_columns.json
                 --run_experiments-------> artifacts/rl/<variant>/seed<N>/{best_model.zip,last_model.zip,training_curve.csv}
                                           artifacts/evaluation/<run_id>/... , <experiment_id>_summary.{json,csv}
                                           DB: runs/metrics/episodes/fills/steps + experiment_aggregates
```
Module map: `features/pipeline.py` (stages 1+2, `ensure_features`), `features/news_alignment.py`
(`align_sentiment_fast`, reference kept), `features/temporal_dataset.py` (`compute_horizon_labels`, lazy `WindowDataset`,
reference `build_windows` kept), `models/transformer_train.py` (`fit_transformer`), `models/embedding.py`
(`embed_all`, `load_frozen_transformer`), `training/env_factory.py` (`make_env`, `evenly_spaced_starts`),
`training/train_sac.py` (`train_sac_agent`, `ValidationCheckpointCallback`), `training/experiment.py` (`run_experiment`).

Key design points
- LOB features are computed ONCE on the full time-sorted validated data (past-only features => slicing by split
  afterwards is leak-free, no cold-start at split starts); splits are the existing day-based chronological split.
- `StandardScaler` fit on TRAIN rows only; applied to all splits, clipped to +-`features.clip_value`, float32.
- Embedding / probability / sentiment columns are standardized the same way (mean/std from the TRAIN split only,
  `extras_scaler.json`; constant columns such as `transformer_signal_valid` are only centered), then clipped.
- Variant column order: `[LOB | sentiment | embedding | probs | signal_valid]`; the LOB block is identical in all
  variants so Avellaneda-Stoikov indices/scale from `feature_meta.json` (`rolling_vol_index`, `midprice_index`,
  `feature_mean`, `feature_scale`) apply to every variant. `build_fused_dataset` is called with private placeholder dim 0
  (env appends the private state itself); it gained optional `signal_valid` and `clip_value` args.
- Transformer embeddings are per split (rows before the first full window use edge padding of row 0 and have
  `transformer_signal_valid = 0`); frozen = eval mode + no grad (checkpoint unchanged is tested).
- Env change (only one): `reset(options={"start_index": i})`. Train env: random starts from a wrapper with its own seeded
  RNG (the base env reseeds from OS entropy when `reset()` gets no seed, which broke reproducibility). Val/test:
  `evenly_spaced_starts` (non-overlapping whenever they fit). `run_backtest(..., start_indices=[...])` added.
- Checkpoint selection: best mean validation episode reward (deterministic policy on fixed val windows, evaluated every
  `sac.eval_freq` steps and once at the end). Test windows are only used after training with `best_model.zip`.
- Baselines (HEURISTIC, AVELLANEDA_STOIKOV) run once per experiment on the same test windows (run ids `heuristic`,
  `avellaneda_stoikov`). Aggregation: mean +- sample std (ddof=1) per variant/metric -> JSON/CSV and DB table
  `experiment_aggregates(experiment_id, variant, metric, mean, std NULL if n<2, n_seeds)`, replace-on-rerun.
  `comparisons_vs_SAC_LOB_ONLY` holds Welch t-test p-values and Cohen's d, labeled exploratory / not a claim of significance.
- Filesystem names use LOWERCASE variants (`sac_full_train.npy`, `artifacts/rl/sac_full/seed0/`), matching the lowercase run ids.
- Other small edits: stable sorts (validate_lob, compute_lob_features, chronological_split, reference alignment) so row
  alignment is deterministic with tied timestamps; `build_run_metadata` reads `features.*`; `training_curve.csv` of a run now
  has columns `timesteps,train_reward_mean,val_reward_mean,val_pnl_mean,val_map,ent_coef,critic_loss,actor_loss,is_best`
  (was `step,reward`; WP3 should read the new columns); `scripts/train_agent.py` / `evaluate.py` build missing features
  on demand and still write `artifacts/rl/model.zip` (Makefile targets keep working).

## Commands for the real run
```bash
source .venv/bin/activate
# 0. data (WP4 tooling) -> data/processed/lob.parquet (+ news.parquet), config e.g. configs/real_binance.yaml
python scripts/prepare_data.py   --config configs/real_binance.yaml
# 1. features (FinBERT runs here if artifacts/sentiment/news_sentiment.parquet is missing; --device auto = cuda/mps/cpu)
python scripts/build_features.py --config configs/real_binance.yaml --stage base
# 2. Transformer (train split only; early stopping on validation; curve in artifacts/transformer/training_curve.csv)
python scripts/train_transformer.py --config configs/real_binance.yaml
# 3. embeddings + fusion for all four variants
python scripts/build_features.py --config configs/real_binance.yaml --stage fusion
# 4. experiment: inspect, then run (resumable)
python scripts/run_experiments.py --config configs/real_binance.yaml --dry-run
python scripts/run_experiments.py --config configs/real_binance.yaml --experiment-id real_v1 --resume
```
Smoke test: `build_features.py --stub-sentiment` (labelled `synthetic_stub`, never used without the flag) then
`run_experiments.py --fast --artifacts-root /tmp/x --db-path /tmp/x/t.db`. Flags: `--variants`, `--seeds`, `--fast`,
`--dry-run`, `--resume`, `--artifacts-root`, `--db-path`, `--data-mode`.

## Config knobs (all in YAML; CLI only overrides)
`features.{vol_window,return_lags,clip_value,label_threshold_bps}`; `transformer.{lr,weight_decay,patience,num_workers,
window_stride,class_weights,embed_batch_size,epochs,batch_size,...}`; `sac.{net_arch,eval_freq,n_eval_episodes,
checkpoint_freq,device,total_timesteps,...}`; `experiment.{name,experiment_id,variants,seeds,n_val_episodes,n_test_episodes,
window_selection}`. Recommended real-run values:
- `sac.total_timesteps` 300k-1M (about 17-55 min per run at ~300 steps/s; see below). Start with 300k, 5 seeds.
- `sac.eval_freq` 10k-20k (each validation pass = `n_eval_episodes * episode_steps` env steps, ~0.1 s per 600 steps for the
  untrained-size policy; negligible next to training). `n_eval_episodes` 5-10 (more windows = less noisy selection).
- `simulator.episode_minutes` 20 (2400 steps at 500 ms); `experiment.n_test_episodes` 30; `experiment.seeds` [0..4].
  With few test days, check `evenly_spaced_starts` does not warn about overlapping windows.
- `buffer_size` >= total_timesteps if RAM allows (obs ~ (n_features+6) floats; 1M x ~100 floats ~ 0.8 GB for obs + next_obs).
- Transformer: `epochs` 20-30 with `patience` 3-5, `class_weights: true` (the flat class dominates short horizons; the
  log prints the validation majority-class accuracy to compare against), `window_stride` 5-20 for multi-million-row data
  (stride 1 gives millions of near-duplicate windows per epoch), `label_threshold_bps` set from the instrument's typical
  1-minute move so classes are not 90% flat. `sac.device: cpu` is usually fastest for these small MLPs.

## Measurements (this machine)
- Alignment: reference vs `align_sentiment_fast` identical (randomized tests, ties/empty/future news). 1M market rows x 10k news,
  3 windows: 0.3 s (target was < 2 min). 1M rows x 300k news (up to 4.5k articles in a window): 14 s.
- Base feature build, 1M rows x 10k news (stub), full pipeline incl. parquet/npy writes: 3 s; peak RSS ~2.9 GB (includes the
  synthetic generator's frames; the pipeline itself holds the float64 feature matrix, ~0.3 GB per 1M rows x 40 cols, plus copies).
- `embed_all`: ~17k rows/s on CPU for d_model 64 / 2 layers / seq 64 (1M rows ~1 min); memory O(batch*seq*F).
- SAC (net 256x256, batch 256, 1 gradient step per env step, CPU): ~300 steps/s (both LOB-only and FULL); env alone ~10k steps/s.
  => 300k steps ~ 17 min, 1M ~ 55 min per run; 4 variants x 5 seeds x 300k ~ 5.5 h sequential.
- fast_demo smoke (100k synthetic rows): base features 0.6 s, Transformer 2 epochs 4 s, fusion 2.4 s, `run_experiments --fast`
  (2 variants x 2 seeds x 600 steps + 2 baselines, 3 test windows) ~10 s total. Same seeds reproduce identical metrics.
- FinBERT: ProsusAI/finbert downloaded and scored 20 headlines on CPU successfully (model load ~42 s incl. download).
- Tests: `pytest -q` 138+ passed in ~10 s.

## Known limitations
- The env indexes a pandas DataFrame per step (`iloc`), capping throughput at ~10k env steps/s (not the bottleneck vs SAC updates).
- Clipping standardized `midprice` at +-10 sigma means AS un-scaling cannot recover prices beyond that range (only affects the
  volatility-in-price-units term; test periods whose price drifts far from the train mean will saturate). Consider dropping raw
  `midprice` from features for real data.
- Train-split embeddings are in-sample for the Transformer (validation/test are out-of-sample); early stopping limits but does not remove this.
- Embeddings are computed per split with edge padding at each split start (31-63 rows flagged `signal_valid = 0`).
- Seeds share the same fixed test windows, so Welch tests compare seed-to-seed training variance only (small n, exploratory).
- Fills remain snapshot approximations (WP1); AS baseline hyper-parameters (gamma, kappa) are untuned.
- `--stub-sentiment` synthetic scores carry no information; any result using them is a pipeline check only.

## Make targets wanted (WP5; Makefile not touched here)
```
features:      python scripts/build_features.py --config $(CONFIG) --stage base
transformer:   python scripts/train_transformer.py --config $(CONFIG)
fusion:        python scripts/build_features.py --config $(CONFIG) --stage fusion
experiments:   python scripts/run_experiments.py --config $(CONFIG) --experiment-id $(EXP) --resume
experiments-dry: python scripts/run_experiments.py --config $(CONFIG) --dry-run
smoke-experiments: build_features --stub-sentiment, train_transformer, build_features --stage fusion, run_experiments --fast (temp --artifacts-root)
```

## Review fixes
- **C1 resume**: run ids/folders unchanged (latest wins). Each run's metadata.json now carries `experiment_id, fast_mode,
  total_timesteps, eval_freq, n_test_windows, test_windows, seed, variant, config_hash, feature_hash`. `--resume` skips a run
  only if metrics.json + metadata.json (+ `best_model.zip` for SAC) exist and all of those match (`check_run_completed`);
  otherwise it retrains and logs why. Without `--resume` everything retrains/overwrites (documented in `--help` and the
  module docstring). `--fast` defaults the experiment id to `<id>_fast` (explicit `--experiment-id` still wins), so smoke
  summaries do not overwrite real ones.
- **I1 feature cache**: `feature_meta.json` stores `lob_hash` (features config, split, depth, LOB sha) and
  `sentiment.hash` (sentiment config, news sha, sentiment cache sha); each `<variant>_columns.json` stores `feature_hash`
  (+ Transformer config subset and checkpoint sha). `ensure_features` reuses on match, rebuilds variant-level staleness
  automatically, and raises a clear error for stale BASE features unless `force=True` (`run_experiments.py --force-features`),
  because a base rebuild invalidates the Transformer. FORMAT CHANGE: `<variant>_columns.json` is now
  `{"columns": [...], "feature_hash": ...}`; use `pipeline.load_columns()` (legacy plain lists still load).
- **I2 level-free inputs**: `midprice` and `microprice` stay in the LOB frame (labels/backtests) but are excluded from model
  inputs via `model_feature_columns` / `features.exclude_model_columns`; new inputs `mid_vs_ma` (mid / rolling mean over
  `features.price_ma_window` - 1) and `microprice_offset`. `meta.lob_feature_columns` is now the model column list
  (`midprice_index` removed). Avellaneda-Stoikov no longer reads features: `predict(obs, inventory, max_inventory, mid=)`
  with the true raw mid supplied by `run_backtest` (it passes `mid=` only to agents whose `predict` accepts it, and calls
  `agent.reset()` per episode); AS keeps its own rolling mid window (`baselines.volatility_window`). Its constructor no
  longer takes feature indices/mean/scale (scripts updated). Clip fractions per split/column are stored in
  `feature_meta.json` (`clip_fractions`, `extras_clip_fractions`) and a warning is logged above `features.max_clipped_fraction` (1%).
- **I3 memory**: embedding/prob/sentiment blocks stay float32 and are standardized in place per split in chunks (float64 only
  per chunk); variant arrays are written straight into `np.lib.format.open_memmap` (no DataFrames). Output is bit-identical to the
  `build_fused_dataset` path (tested).
- **I4**: labels are IGNORE when the target row is later than `h * transformer.label_max_gap_factor` (default 1.5) after ts[i].
- **Minor**: fallback saves `best_model.zip` unconditionally; `max_inventory` arg removed from `evaluate_on_windows` /
  `train_sac_agent`; `features.volume_window` drives `rolling_traded_volume`; dead line removed; test that checkpoint selection
  uses the VALIDATION env. `extras_scaler.json` now merges with previously built groups instead of dropping them.
- Tests: 162 pass (~22 s). Intentionally changed: AS tests (new constructor/`mid=`), `test_scaler_fit_on_train_only` (uses
  `mid_vs_ma`; test-split mean check relaxed to `> 0`), columns.json readers. New: `tests/test_wp2_review.py`.
- Smoke in temp dirs (100k synthetic rows): base 1.7 s, Transformer 2.6 s, fusion 2.3 s, `run_experiments --fast` 2 variants x 2 seeds 10.9 s,
  `--resume` rerun 1.5 s (all 6 runs skipped).
