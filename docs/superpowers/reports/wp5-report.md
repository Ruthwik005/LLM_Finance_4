# WP5 report: final report generator, pipeline orchestration, documentation

## Delivered
- `src/market_maker/evaluation/report.py` (+ `report_text.py`): 24-section `FINAL_PROJECT_REPORT.md` from the SQLite DB (runs, metrics,
  experiment_aggregates, steps) and run artifacts/metadata. Sources: `--db-path`/`$MARKET_MAKER_DB_PATH`, `--artifacts-root`/`$MARKET_MAKER_ARTIFACTS_ROOT`,
  `--experiment-id`, `--config`. Section 43 table (mean ± sample std with n for learned variants, single values for baselines, `N/A (reason)` otherwise),
  secondary table, four PNG figures in `reports/figures/` (PnL + equity, inventory/MAP, adverse selection, learning curves), programmatic conservative
  discussion (Welch p-values/Cohen's d from the experiment summary, "within seed-to-seed noise" unless p < 0.05, small-n caveat, measured diagnostics),
  mandatory synthetic banner and fills disclaimer, all §33 limitations, references. The DB is opened read-only and never created.
- `scripts/run_all.py` rewritten (`src/market_maker/orchestration.py`): prepare_data -> build_features -> train_transformer (+ fusion) -> run_experiments -> generate_report;
  flags `--config --fast --stub-sentiment --from-stage --only-stage --dry-run --resume --force-features --artifacts-root --db-path --experiment-id --variants --seeds --data-mode`.
  Real FinBERT unless `--stub-sentiment` (which also forces `data_mode=synthetic_smoke_test`). No news file and no cached scores -> `--no-sentiment`, sentiment variants dropped and recorded
  in `artifacts/logs/pipeline_status.json` (the report prints them as N/A with the reason). Fail fast with the stage's exit code and a clear message; per-stage timing.
- `scripts/launch_demo.py` (`make demo`), Makefile targets `features, train-transformer, experiments, experiments-fast, full-run, demo-data (new pipeline), demo, lint, test-py`.
- README, BUILD_STATUS, `docs/SPEC_COMPLIANCE.md`, `docs/RUNBOOK.md`.

## Verified (temp dirs only; the real demo data/DB were not modified except a new `artifacts/reports/` output)
- `run_all.py --fast --stub-sentiment --seeds 0 1` on a fresh synthetic dataset in a temp dir: all 5 stages ok (~28 s); `--resume --from-stage run_experiments` skipped all runs; no-news config produced N/A rows for the two sentiment variants.
- `launch_demo.py` with `API_PORT=8777 FRONTEND_PORT=5199`: API health 200, Vite 200, clean shutdown on SIGINT; busy port 8000 (Docker) gives the guidance message and exit 2.
- Live collector 18 s test (34 rows) + merge; Binance exchangeInfo tick size lookup; all `make -n` targets; `make lint` (compileall + oxlint, warnings only).
- `pytest -q`: 180 passed (28 s). New: 10 report tests, 9 run_all/launch_demo tests.

## Notes / concerns
- `make demo-data` itself was not executed (it regenerates `data/processed/lob.parquet` and rewrites shared run folders); its commands were verified as above in temp dirs.
- Data mode is inferred from the config file name (`fast_demo.yaml`) in `training/experiment.py`; copies of the synthetic config under other names need `--data-mode synthetic_smoke_test`.
- Episodes are not guarded against gaps in collected data (documented in the report, README, SPEC_COMPLIANCE).
- The IMM paper title in the references was written from memory and flagged in the report as "verify against the PDF".
- RUNBOOK numbers that are estimates (disk per day, hours per run) are marked as such; steps/sec comes from the WP2 measurement.
