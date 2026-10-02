# Spec Compliance

Maps every requirement of the build specification (`LLM_Enhanced_Market_Making_End_to_End_Build_Spec.md`) to its status in this repository.

Status vocabulary:

- **Done**: implemented and covered by a test or a command that was run.
- **Done-with-note**: implemented, but with a deviation or limitation stated in the note.
- **User-run**: the code is ready, but the result needs real data or a long training run that only you can do (see `docs/RUNBOOK.md`). Not verified here.
- **Not built (optional)**: not implemented; the spec marks it optional or second-phase.

Tests live in `tests/`. "Verified" means the command was run during development on a laptop with synthetic data. No real-data experiment has been run, so **no financial claim of any kind is supported by this repository today**.

## Section 3: must-have list

| # | Requirement | Status | Where / note |
|---|---|---|---|
| 1 | Historical LOB CSV/Parquet adapter | Done | `data/lob_loader.py`, `data/adapters.py` (`load_lob_with_mapping`), `tests/test_adapters.py`, `tests/test_data.py` |
| 2 | Historical news CSV/Parquet adapter | Done | `data/news_loader.py`, `adapters.normalize_news`, `scripts/prepare_fnspid.py` |
| 3 | Data validation + chronological split | Done | `data/validators.py`, `data/splits.py`, `tests/test_data.py` |
| 4 | LOB feature engineering | Done | `features/lob_features.py`, `tests/test_features.py` |
| 5 | FinBERT inference with caching | Done-with-note | `features/sentiment.py`, `tests/test_sentiment.py` (cache round trip). Real `ProsusAI/finbert` scored 20 headlines in a smoke test (WP2). A full-scale FNSPID scoring run is User-run |
| 6 | Transformer temporal encoder + supervised pretraining | Done | `models/transformer_encoder.py`, `models/transformer_train.py`, `tests/test_transformer.py`, `tests/test_wp2_training.py`. Quality on real data is User-run |
| 7 | Feature fusion | Done | `features/fusion.py`, `features/pipeline.py`, `tests/test_fusion.py`, `tests/test_wp2_features.py` |
| 8 | Gymnasium-compatible environment | Done | `env/market_making_env.py`, `tests/test_env.py` (`check_env`) |
| 9 | Continuous multi-price-level action | Done | 4-dim action -> 2 bid + 2 ask levels (`tests/test_env.py`, `tests/test_wp1.py`) |
| 10 | SAC training | Done | `training/train_sac.py`, `tests/test_wp2_training.py` (smoke budgets only; long training is User-run) |
| 11 | Inventory/risk-aware reward | Done | `env/reward.py` (truncated and quadratic), `tests/test_reward.py` |
| 12 | At least two baselines and three ablations | Done | Baselines: `HEURISTIC`, `AVELLANEDA_STOIKOV`. Ablations: `SAC_LOB_ONLY`, `SAC_LOB_TRANSFORMER`, `SAC_LOB_SENTIMENT`, `SAC_FULL` (`training/experiment.py`). Run on synthetic data; real runs are User-run |
| 13 | Backtest/evaluation pipeline | Done | `evaluation/backtest.py`, `evaluation/metrics.py`, `scripts/run_experiments.py` |
| 14 | Metrics persisted to disk | Done-with-note | `metrics.json`, `episodes.csv`, `fills.csv`, `orders.csv`, `quotes.csv`, SQLite. `metrics.csv` per run is **not** written |
| 15 | Streamlit dashboard | Done-with-note | **Deviation:** a React (Vite) dashboard replaces Streamlit (`frontend/`, `scripts/launch_demo.py`). `make dashboard` does not exist; use `make demo` or `make frontend` |
| 16 | FastAPI read-only results API | Done | `api/main.py`, `tests/test_api.py`, `tests/test_api_wp3.py` |
| 17 | Automated Markdown final report | Done | `evaluation/report.py`, `tests/test_report.py`. The report in `artifacts/reports/` exists only after `make report`; a report on real data is User-run |
| 18 | Unit/integration tests | Done | `pytest -q`; `tests/test_end_to_end.py`, `tests/test_wp2_training.py::test_run_experiment_end_to_end`, `tests/test_run_all.py` |
| 19 | Reproducible seeds/configuration | Done-with-note | seeds and hashes in run metadata (`config_hash`, `feature_hash`); no bit-for-bit determinism across hardware is promised (spec says the same) |
| 20 | README with setup and commands | Done | `README.md`, `docs/RUNBOOK.md` |

## Section 36: definition of done

| Item | Status | Note |
|---|---|---|
| Fresh install instructions work | Done-with-note | `make install` (Python 3.11 venv) and `make install-frontend` are documented. A clean-machine install was not re-run in this work package |
| Unit and integration tests pass | Done | `pytest -q` (see `docs/superpowers/reports/wp5-report.md` for the count). Frontend tests: `cd frontend && npx vitest run` |
| Pipeline can run end to end | Done-with-note | `python scripts/run_all.py --config configs/fast_demo.yaml --fast --stub-sentiment` verified on synthetic data in temp dirs. On real data: User-run |
| FinBERT output cached and correctly time-aligned | Done-with-note | cache + as-of alignment tested (`tests/test_news_alignment.py`, `tests/test_wp2_alignment.py`, which compares the fast aligner with the reference one); real FinBERT over a full news file is User-run |
| Transformer checkpoint can train/load | Done | `tests/test_transformer.py`, `tests/test_wp2_training.py`, `tests/test_wp2_review.py` (checkpoint bound to base features) |
| Gymnasium environment passes validation | Done | `tests/test_env.py::test_gymnasium_check_env_passes`, `tests/test_wp1.py::test_check_env_with_private_state` |
| SAC can train and evaluate | Done | smoke budgets verified; meaningful training is User-run |
| Two-level bids and asks from continuous actions | Done | `tests/test_env.py`, `tests/test_wp1.py` |
| Inventory and cash accounting tested | Done | `tests/test_accounting.py`, `tests/test_wp1.py` (fuzz, conservation, PnL decomposition) |
| Required ablations run or are marked unavailable | Done | all four variants run on synthetic data; a variant without news is recorded N/A with a reason (`run_all.py`, report tables). Real ablations are User-run |
| Metrics generated from actual runs | Done | SQLite + run folders; report reads only those |
| Dashboard displays results | Done-with-note | React dashboard, Vitest smoke renders of every screen; API + dev server start verified with `scripts/launch_demo.py`. Not Streamlit |
| API displays results | Done | `tests/test_api.py`, `tests/test_api_wp3.py` |
| Final report generated from actual artifacts | Done-with-note | generator verified on the synthetic demo; the report that matters (real data) is User-run |
| README contains one-command demo instructions | Done | `make demo-data` then `make demo` (two commands) or `python scripts/run_all.py ...` |
| Synthetic/demo data clearly labeled | Done | `data_mode=synthetic_smoke_test` in metadata, UI chip, report banner, `*_stub` sentiment files, `run_all.py --stub-sentiment` forces the label. Caveat: for other config file names the data mode must be passed with `--data-mode` |
| No future-data leakage | Done | `tests/test_news_alignment.py::test_future_news_cannot_leak_backward`, `tests/test_wp2_alignment.py`; scaler fit on train only (`tests/test_wp2_features.py::test_scaler_fit_on_train_only`); LOB features use past rows only; checkpoint selection on validation only |
| No fabricated performance claims | Done-with-note | the report generator derives every number from the DB/artifacts and a test greps generated text for forbidden phrases. Read the final real report yourself before submitting |

## Section 42: acceptance scenarios

| Scenario | Status | Evidence |
|---|---|---|
| A. No news available | Done | `tests/test_news_alignment.py::test_no_news_returns_default_neutral_features` (mask false, defaults 0); `run_all.py` news-optional mode skips sentiment variants (`tests/test_run_all.py::test_no_news_skips_sentiment_variants_with_reason`) |
| B. News at 10:00:01 invisible at 10:00:00 | Done | `test_future_news_cannot_leak_backward`, `test_news_at_or_before_timestamp_is_visible`, `test_news_published_exactly_at_timestamp_is_visible` |
| C. Inventory near long limit | Done | `tests/test_env.py::test_inventory_never_exceeds_max_long_limit`, `tests/test_wp1.py::test_fuzz_inventory_hard_limit` |
| D. Inventory near short limit | Done | `test_inventory_never_exceeds_max_short_limit`, fuzz test |
| E. No trade reaches passive quote | Done | `tests/test_execution.py::test_bid_does_not_fill_when_market_never_reaches_price` |
| F. Partial available volume | Done | `test_partial_fill_capped_by_available_trade_volume`, `tests/test_wp1.py::test_partial_fill_remainder_persists` |
| G. Identical seed/config reproducibility | Done-with-note | seed + environment recorded in metadata; same seeds gave identical metrics on one machine (`tests/test_wp2_training.py::test_sac_training_deterministic_and_saves_best`). Not promised across hardware |

## Section 29: required tests

| Requirement | Status | Test |
|---|---|---|
| Chronological split has no overlap | Done | `tests/test_data.py::test_chronological_split_has_no_overlap_and_is_ordered` |
| Timestamps sorted | Done | same test; stable sorts in `validate_lob` (`tests/test_wp2_alignment.py`) |
| Crossed books caught | Done | `tests/test_data.py::test_validate_lob_removes_crossed_book_rows` |
| Future news cannot leak | Done | see scenario B |
| No-news state returns neutral features | Done | see scenario A |
| FinBERT cache round trip | Done | `tests/test_sentiment.py::test_cache_sentiment_round_trip` |
| No NaN/inf after preprocessing | Done | `tests/test_features.py::test_compute_lob_features_has_no_nan_or_inf`, `tests/test_wp2_features.py::test_variant_artifacts_composition_finite_clipped` |
| Training scaler reused on validation/test | Done | `tests/test_features.py::test_scaler_fit_on_train_reused_on_validation` |
| Transformer shapes, forward pass, checkpoint | Done | `tests/test_transformer.py` |
| Buy / sell fill accounting | Done | `tests/test_accounting.py` |
| Partial fill cap, no overfill | Done | `tests/test_execution.py` |
| Cancellation count | Done | `tests/test_wp1.py::test_quote_persistence_no_cancel_when_unchanged`, `test_step_log_fields`, `test_top_up_when_desired_exceeds_remaining` |
| `check_env` passes | Done | `tests/test_env.py` |
| Observation in observation space | Done | `tests/test_env.py::test_observation_always_within_observation_space` |
| Action maps into valid quotes | Done | `tests/test_env.py`, `tests/test_wp1.py` |
| Inventory hard limit | Done | see scenarios C/D |
| Episode terminates correctly | Done | `tests/test_env.py::test_episode_terminates_at_configured_length` |
| Reward tests | Done | `tests/test_reward.py` |
| Metrics: MAP, PnL/MAP zero, drawdown, adverse selection | Done | `tests/test_metrics.py`, `tests/test_wp1.py::test_summarize_run_hand_computed`, `test_zero_fill_and_zero_map_edge_cases` |
| Integration: prepare -> sentiment -> transformer -> env -> SAC -> evaluate -> report | Done | `tests/test_end_to_end.py` (components) and `tests/test_wp2_training.py::test_run_experiment_end_to_end` (new multi-seed pipeline); the whole CLI chain is checked by `run_all.py` on synthetic data |

## Other spec sections

| Spec item | Status | Note |
|---|---|---|
| §1.1 corrections applied in docs/report | Done | README, report section 5 (all 8 items) |
| §9 Mode A / B / C | Done-with-note | B and C supported; A (matched real news) needs data you do not have in this project |
| §11.3 sentiment windows, mask, caps | Done | `features/news_alignment.py` |
| §12 Transformer horizons, early stopping, frozen | Done | `models/`, config `transformer.*` |
| §14.3 reference price `imm_stable` | Not built (optional) | the mode exists but falls back to midprice (documented in code; not claimed as IMM) |
| §14.4 never cross a session boundary | Not built | episodes are contiguous row windows; gaps in collected data are not detected (documented limitation). Training starts are random in the train split; validation/test windows are deterministic |
| §15.1 event-aware execution | Not built (optional) | requires message-level data |
| §15.2 quote persistence, placed/canceled counts | Done | `env/market_making_env.py`, `orders.csv` |
| §15.3 simulator disclaimer in every report | Done | report banner + environment section + limitations (`tests/test_report.py`) |
| §17.1-17.4 reward options | Done-with-note | both inventory penalties and cancellation penalty; `adverse_selection_penalty` config field exists but is not applied |
| §18 SAC with validation checkpoint selection | Done | `training/train_sac.py`; test set used only after selection |
| §19 imitation / behavior cloning | Not built (optional) | stated in report and README |
| §20.3 TD3, Avellaneda-Stoikov baselines | TD3: Not built (optional). A-S: Done | `agents/avellaneda_stoikov.py` (untuned gamma/kappa) |
| §21-22 metrics and adverse-selection definition | Done | `evaluation/metrics.py`; horizon/threshold from config |
| §23 run artifacts | Done-with-note | all listed files except `metrics.csv` |
| §24 API endpoints | Done | plus extra read-only endpoints |
| §25 dashboard pages | Done-with-note | React equivalents of every page; not Streamlit |
| §26 report with 24 sections | Done | `tests/test_report.py` checks all 24 headings |
| §27 CLI | Done-with-note | `make install/sample-data/prepare/sentiment/transformer/train/evaluate/report/api` exist; `make dashboard` is `make frontend`/`make demo`; `run_all.py` updated for the new pipeline |
| §28 YAML configuration | Done | `configs/` (extra sections: `features`, `evaluation`, `baselines`, `experiment`) |
| §30 logging, fail early | Done-with-note | stage logs with timing; fails early on empty splits, missing LOB columns and crossed books. "Action interval inconsistent with dataset frequency" is not checked |
| §31 performance practices | Done | Parquet, cached sentiment, precomputed embeddings, memmap fusion; env still indexes a DataFrame per step (~10k steps/s) |
| §32 integrity rules | Done-with-note | enforced in generated text and tests; human review still required |
| §33 limitations documented | Done | report section 21, README |
| §34 safety (no keys, `.env` ignored) | Done | `.env.example`, `.gitignore` |
| §35 daily plan acceptance (`pytest -q`, `run_all.py` on fast demo) | Done-with-note | verified on this machine |
| §38 demo path | Done-with-note | `make demo` serves saved artifacts; Episode Replay shows quotes/fills/inventory/PnL against the heuristic. It replays saved test episodes; it does not run live policy inference |
| §41 required files | Done-with-note | README, BUILD_STATUS, both configs exist; `artifacts/reports/FINAL_PROJECT_REPORT.md` appears after `make report` (artifacts are git-ignored) |
| §6 repository layout | Done-with-note | `pyproject.toml` replaced by `setup.py` + `setup.cfg`; `dashboard/` replaced by `frontend/`; extra modules added |
| §45 second-phase items (message-level replay, IRLU, multi-asset, MLflow, Docker, paper trading) | Not built (optional) | |
