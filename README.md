# LLM-Enhanced Financial Market Making

Research/backtesting MVP extending the multi-price-level IMM market-making framework with
FinBERT sentiment, a Transformer temporal encoder, and a Soft Actor-Critic (SAC) agent.

**Correction to the original slides:** IMM is already a multi-price-level method in the base
paper (two bid levels, two ask levels) — this project does not claim to introduce
multi-price-level quoting as a novel contribution; it inherits it from IMM.

## Architecture

Raw LOB/news → validation + chronological split → LOB features + FinBERT sentiment (cached)
+ Transformer embeddings (frozen) → feature fusion → Gymnasium `MarketMakingEnv` → SAC
(Stable-Baselines3) → backtest/evaluation written to SQLite + file artifacts → FastAPI
(reads SQLite) → React frontend (calls FastAPI) + auto-generated Markdown report.

## Data mode

**Mode B**: real crypto LOB (Binance trades/aggTrades + self-collected or Kaggle L2 depth
snapshots) is the primary trading dataset. FNSPID (real timestamped equity news) validates the
sentiment/alignment pipeline on its own ticker/period. These are NOT causally matched — no claim
is made that news improved trading on the crypto LOB. See `configs/default.yaml` for paths.

## Data schema

See `src/market_maker/schemas.py` (`LobRow`, `NewsRow`) for the exact required columns.

## Setup

```bash
make install
make install-frontend
make sample-data   # synthetic smoke-test data — labeled, not a performance result
```

## Fast demo (synthetic, minutes)

```bash
python scripts/run_all.py --config configs/fast_demo.yaml
```

## Full pipeline (real Mode-B data)

Place real LOB/news Parquet files at the paths in `configs/experiment_full.yaml`, then:

```bash
python scripts/run_all.py --config configs/experiment_full.yaml
```

## Artifacts

- `artifacts/evaluation/<run_id>/` — config, metadata, metrics, episodes, fills, plots, report (file artifacts)
- `artifacts/market_maker.db` — SQLite DB queried live by the API/frontend
- `artifacts/reports/FINAL_PROJECT_REPORT.md` — auto-generated final report

## Frontend / API

No results yet? `make demo-data` generates synthetic data, trains a small SAC agent and creates two demo runs
(labeled as synthetic smoke-test data in the UI).

In separate terminals:

```bash
make api         # FastAPI, http://localhost:8000/docs  (port busy? `make api PORT=8765` and set VITE_API_URL=http://localhost:8765 in frontend/.env.local)
make frontend    # React (Vite) dev server, http://localhost:5173
```

## Testing

```bash
make test
```

## Known limitations

- Fills/queue priority are approximated from snapshot data, not a message-level exchange reconstruction.
- Mode B means LOB and news come from different assets/periods; no causal sentiment-to-PnL claim is made.
- The Transformer encoder is frozen during SAC training (a simplification vs. IMM's joint representation+policy learning).
- Ablation variants (`SAC_LOB_TRANSFORMER`, `SAC_LOB_SENTIMENT`, `SAC_FULL`) require wiring sentiment/transformer artifacts from Tasks 5/7; currently only `SAC_LOB_ONLY` is trainable end-to-end.

## Disclaimer

Research/backtesting only. No live trading, no broker credentials, no real-money execution.
