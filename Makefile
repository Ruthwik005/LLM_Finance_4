.PHONY: install install-frontend sample-data demo-data prepare sentiment transformer train-transformer features train evaluate evaluate-sac \
        experiments experiments-fast full-run report api frontend demo lint test test-py collect-lob download-public prepare-fnspid

# Config used by the single-step targets. Override: make features CONFIG=configs/real_binance.yaml
CONFIG ?= configs/fast_demo.yaml
REAL_CONFIG ?= configs/real_binance.yaml
EXP ?=
API_PORT ?= 8000
PORT ?= $(API_PORT)
export API_PORT
EXP_FLAG = $(if $(EXP),--experiment-id $(EXP),)

install:
	python3.11 -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt && pip install -e .

install-frontend:
	cd frontend && npm install

# --- demo (SYNTHETIC data: a software smoke test, not a performance result) -----------------------------------
# NOTE: sample-data (re)writes data/processed/lob.parquet and news.parquet with synthetic data.
sample-data:
	python -c "from market_maker.data.sample_data import generate_synthetic_lob, generate_synthetic_news; \
	lob = generate_synthetic_lob(100000, 5, 'BTCUSDT', 42); lob.to_parquet('data/processed/lob.parquet'); \
	news = generate_synthetic_news(200, 'BTCUSDT', lob, 42); news.to_parquet('data/processed/news.parquet')"

# New pipeline on synthetic data with stub sentiment; writes labeled-synthetic demo runs into artifacts/ and the DB.
demo-data: sample-data
	python scripts/run_all.py --config configs/fast_demo.yaml --fast --stub-sentiment --seeds 0 1 2
	@echo "Synthetic demo runs created (SOFTWARE SMOKE TEST ONLY). Now run: make demo"

# Starts the API (API_PORT, default 8000) + the React dev server, waits for health, Ctrl-C stops both.
demo:
	python scripts/launch_demo.py

# --- individual pipeline steps (CONFIG=...) -------------------------------------------------------------------
prepare:
	python scripts/prepare_data.py --config $(CONFIG)

sentiment:
	python scripts/run_finbert.py --config $(CONFIG)

features:
	python scripts/build_features.py --config $(CONFIG) --stage base

transformer train-transformer:
	python scripts/train_transformer.py --config $(CONFIG)

train:
	python scripts/train_agent.py --config $(CONFIG)

evaluate:
	python scripts/evaluate.py --config $(CONFIG)

evaluate-sac:
	python scripts/evaluate.py --config $(CONFIG) --model-path artifacts/rl/model.zip --run-id sac_lob_only --variant SAC_LOB_ONLY

# Variants x seeds from the config (resumable). Features/Transformer must exist (make features, make train-transformer).
experiments:
	python scripts/run_experiments.py --config $(CONFIG) $(EXP_FLAG) --resume

experiments-fast:
	python scripts/run_experiments.py --config $(CONFIG) $(EXP_FLAG) --fast

# Whole pipeline with the REAL config (long: hours). Real FinBERT, resumable. Try: make full-run DRY=--dry-run
full-run:
	python scripts/run_all.py --config $(REAL_CONFIG) --resume $(DRY)

report:
	python scripts/generate_report.py

api:
	uvicorn market_maker.api.main:app --reload --port $(PORT)

frontend:
	cd frontend && npm run dev

lint:
	python -m compileall -q src scripts tests
	cd frontend && npm run lint

test-py:
	pytest -q

test: test-py
	cd frontend && npx vitest run

# --- real-data tooling ------------------------------------------------------------------------------------------
collect-lob:
	python scripts/collect_binance_lob.py --duration-minutes 30 --merge --merge-out data/processed/lob_binance.parquet

download-public:
	python scripts/download_binance_public.py --symbol BTCUSDT --start 2023-06-01 --end 2023-06-01 --max-rows 1000000

prepare-fnspid:
	python scripts/prepare_fnspid.py --tickers AAPL --max-rows 5000
