.PHONY: install install-frontend sample-data prepare sentiment transformer train evaluate report api frontend demo test

install:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt && pip install -e .

install-frontend:
	cd frontend && npm install

sample-data:
	python -c "from market_maker.data.sample_data import generate_synthetic_lob, generate_synthetic_news; \
	lob = generate_synthetic_lob(50000, 5, 'BTCUSDT', 42); lob.to_parquet('data/processed/lob.parquet'); \
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

report:
	python scripts/generate_report.py

api:
	uvicorn market_maker.api.main:app --reload

frontend:
	cd frontend && npm run dev

# Run `make api` and `make frontend` in two separate terminals for the full demo.
demo:
	@echo "Run 'make api' in one terminal and 'make frontend' in another."

test:
	pytest -q
	cd frontend && npx vitest run
