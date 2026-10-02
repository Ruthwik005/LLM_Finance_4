.PHONY: collect-lob download-public prepare-fnspid install install-frontend demo-data sample-data prepare sentiment transformer train evaluate evaluate-sac report api frontend demo test

install:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt && pip install -e .

install-frontend:
	cd frontend && npm install

demo-data: sample-data prepare train evaluate evaluate-sac
	@echo "Demo runs created (heuristic_baseline, sac_lob_only). Start 'make api' and 'make frontend'."

sample-data:
	python -c "from market_maker.data.sample_data import generate_synthetic_lob, generate_synthetic_news; \
	lob = generate_synthetic_lob(100000, 5, 'BTCUSDT', 42); lob.to_parquet('data/processed/lob.parquet'); \
	news = generate_synthetic_news(200, 'BTCUSDT', lob, 42); news.to_parquet('data/processed/news.parquet')"

prepare:
	python scripts/prepare_data.py --config configs/fast_demo.yaml

sentiment:
	python scripts/run_finbert.py --config configs/fast_demo.yaml

transformer:
	python scripts/train_transformer.py --config configs/fast_demo.yaml

train:
	python scripts/train_agent.py --config configs/fast_demo.yaml

evaluate:
	python scripts/evaluate.py --config configs/fast_demo.yaml

evaluate-sac:
	python scripts/evaluate.py --config configs/fast_demo.yaml --model-path artifacts/rl/model.zip \
		--run-id sac_lob_only --variant SAC_LOB_ONLY

report:
	python scripts/generate_report.py

PORT ?= 8000

api:
	uvicorn market_maker.api.main:app --reload --port $(PORT)

frontend:
	cd frontend && npm run dev

# Run `make api` and `make frontend` in two separate terminals for the full demo.
demo:
	@echo "Run 'make api' in one terminal and 'make frontend' in another."

test:
	pytest -q
	cd frontend && npx vitest run

collect-lob:
	python scripts/collect_binance_lob.py --duration-minutes 30 --merge --merge-out data/processed/lob_binance.parquet

download-public:
	python scripts/download_binance_public.py --symbol BTCUSDT --start 2023-06-01 --end 2023-06-01 --max-rows 1000000

prepare-fnspid:
	python scripts/prepare_fnspid.py --tickers AAPL --max-rows 5000
