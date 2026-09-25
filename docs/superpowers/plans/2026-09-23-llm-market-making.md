# LLM-Enhanced Market Making Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a locally runnable research MVP that ingests historical LOB + news data, produces FinBERT sentiment and Transformer temporal features, trains a SAC market-making agent in a historical-replay Gym environment, evaluates it against baselines/ablations, and exposes results via a FastAPI backend (backed by SQLite) + React frontend and an auto-generated Markdown report.

**Architecture:** Modular monolith in `src/market_maker/`. Data flows one-way: raw LOB/news → validated/split → features (LOB numeric + FinBERT sentiment + Transformer embeddings) → fused observation vectors → Gymnasium env → SAC (Stable-Baselines3) → evaluation results written to SQLite (runs/metrics/episodes/fills tables) plus file artifacts (config, checkpoints, plots) → FastAPI (read-only, queries SQLite) → React frontend (calls the FastAPI JSON endpoints). Every stage persists its output so no stage re-runs an earlier one's expensive work (esp. FinBERT).

**Tech Stack:** Backend: Python 3.11 (plain `venv` + `requirements.txt`, no `pyproject.toml`/`uv`), PyTorch, Hugging Face `transformers` (ProsusAI/finbert), Gymnasium, Stable-Baselines3 (SAC), Pandas/PyArrow, scikit-learn, Pydantic, PyYAML, FastAPI/Uvicorn, SQLAlchemy + SQLite, pytest. Frontend: React (Vite), plain `fetch`, Recharts for charts, no heavy UI framework.

**Spec:** `/Users/ruthwik.chamala/Downloads/LLM_Enhanced_Market_Making_End_to_End_Build_Spec.md`

**Data mode:** Mode B (spec Section 9) — real self-collected/Binance crypto LOB (trades/aggTrades + self-collected or Kaggle L2 depth snapshots) as the primary trading dataset; FNSPID (real timestamped equity news) used to validate the sentiment/alignment pipeline on its own ticker/period, not causally joined to the crypto LOB. Every report and dashboard view produced from this data must say so explicitly — no claim of matched Mode-A data.

## Global Constraints

- Python 3.11 only; no other backend runtime. Plain `venv` + `requirements.txt` for dependency management — no `pyproject.toml`, no `uv`, no Poetry.
- React (Vite, plain JS/JSX, no TypeScript unless the executor already has it set up) for the frontend; it talks to the backend only via the FastAPI JSON endpoints, never by reading files or the SQLite DB directly.
- Never shuffle time-series rows across train/val/test — always chronological (default 70/15/15 by day, spec §8.4).
- Never use test-split data to fit scalers or tune hyperparameters (spec §8.4, §18, §44).
- News alignment must be as-of backward-only: a market state at time `t` may only see news with `published_at <= t` (spec §11.2). This is a mandatory, explicitly tested invariant.
- Do not run FinBERT inside the RL loop; sentiment must be precomputed and cached to Parquet (spec §5, §11.1).
- Freeze the Transformer encoder during SAC training (spec §12.4).
- Action space is `gymnasium.spaces.Box` with exactly 4 dimensions: mid offset, spread control, bid split, ask split, each in `[-1, 1]` before mapping (spec §14.2).
- All quotes rounded to `tick_size`; bid price < ask price; no negative volumes; hard inventory limit enforced (spec §14.2).
- Sign convention: buy fill → inventory up, cash down; sell fill → inventory down, cash up (spec §16). Used identically in accounting, tests, and reward.
- No fabricated performance claims; if an experiment did not run, artifacts/report must say `N/A` / `Not run / insufficient data`, never a fabricated number (spec §26, §32, §44).
- No broker/exchange credentials, no live order submission anywhere in the codebase (spec §34).
- Config is YAML-driven (`configs/*.yaml`), never hard-coded constants (spec §28).
- Every experiment run persists queryable data (metadata, metrics, episodes, fills) to SQLite tables (Task 14a), and file artifacts (`config.yaml`, `training_curve.csv`, `model_path.txt`, `plots/`, `report.md`) to `artifacts/evaluation/<run_id>/` (spec §23, adapted for the DB-backed API/frontend).

## Review Focus

- **Look-ahead leakage in news alignment** — a headline published one tick after `t` must never influence the fused observation at `t`; the reader would reasonably expect a documented as-of join, not nearest-neighbor. Owned by Task 6 (`test_news_alignment.py` leakage test, spec §11.4).
- **Crossed/locked book rows silently corrupting features** — a row where `best_bid >= best_ask` (common in noisy snapshot data) must be quarantined, not averaged into midprice/spread, or every downstream feature is wrong. Owned by Task 3 (`test_data.py`).
- **Fill-simulator over-filling or filling with no matching trade** — a passive quote sitting at a price the market never actually traded through must NOT fill; a partial fill must never exceed available volume or outstanding order size. Owned by Task 9 (`test_execution.py`, spec §15.1, Scenario E/F).
- **Inventory hard-limit breach under adversarial actions** — an agent that always asks for max bid+ask volume must still be clamped at `max_inventory`, not silently allowed to exceed it (spec §14.2, Scenario C/D). Owned by Task 11 (`test_env.py`).
- **PnL/MAP divide-by-zero when an episode has zero net inventory** — a flat/idle episode (MAP = 0) must return a defined sentinel (e.g. `NaN`/`None` with a documented reason), not crash or silently return `inf`. Owned by Task 13 (`test_metrics.py`, spec §21).

---

## Phase 1 — Skeleton, Config, Data Contracts

### Task 1: Repository skeleton, config loader, seeds, logging

**Files:**
- Create: `requirements.txt`, `setup.cfg` (minimal, for `pip install -e .` editable install — no `pyproject.toml`/`uv`), `.gitignore`, `.env.example`, `Makefile`, `README.md` (stub), `BUILD_STATUS.md`
- Create: `configs/default.yaml`, `configs/fast_demo.yaml`
- Create: `src/market_maker/__init__.py`
- Create: `src/market_maker/config.py`
- Create: `src/market_maker/utils/__init__.py`
- Create: `src/market_maker/utils/logging.py`
- Create: `src/market_maker/utils/seeds.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `market_maker.config.load_config(path: str) -> AppConfig` (Pydantic model with nested sections `project`, `data`, `sentiment`, `transformer`, `simulator`, `reward`, `sac`, `experiments`, matching spec §28 keys exactly).
- Produces: `market_maker.utils.seeds.set_global_seed(seed: int) -> None` (seeds `random`, `numpy`, `torch`).
- Produces: `market_maker.utils.logging.get_logger(name: str) -> logging.Logger` (timestamp + module name formatter, per spec §30).

- [ ] **Step 1: Initialize the repo**

```bash
cd "/Users/ruthwik.chamala/Desktop/superdev-4:1"
git init
```

- [ ] **Step 2: Write `.gitignore` and `.env.example`**

`.gitignore`:
```text
__pycache__/
*.pyc
.venv/
data/raw/
data/interim/
data/processed/
artifacts/
.env
*.egg-info/
.pytest_cache/
```

`.env.example`:
```text
# No secrets required for this research MVP.
# Placeholder only — do not put broker/exchange API keys here.
DEVICE=auto
```

- [ ] **Step 3: Write `setup.cfg` and `requirements.txt`**

`setup.cfg` (plain setuptools config, no `pyproject.toml`/`uv` — just enough for `pip install -e .` so `import market_maker` works from `src/`):
```ini
[metadata]
name = market-maker
version = 0.1.0

[options]
package_dir =
    = src
packages = find:
python_requires = >=3.11

[options.packages.find]
where = src
```

Create a minimal `setup.py` alongside it (required by older pip for editable installs with `setup.cfg`):
```python
from setuptools import setup
setup()
```

`requirements.txt`:
```text
torch>=2.2
transformers>=4.40
gymnasium>=0.29
stable-baselines3>=2.3
numpy>=1.26
pandas>=2.2
pyarrow>=15.0
scikit-learn>=1.4
pydantic>=2.6
pyyaml>=6.0
fastapi>=0.110
uvicorn>=0.29
sqlalchemy>=2.0
plotly>=5.21
matplotlib>=3.8
pytest>=8.1
httpx>=0.27
```

- [ ] **Step 4: Write the failing config test**

```python
# tests/test_config.py
from market_maker.config import load_config

def test_load_default_config_has_required_sections():
    cfg = load_config("configs/default.yaml")
    assert cfg.project.seed == 42
    assert cfg.data.symbol == "BTCUSDT"
    assert cfg.simulator.levels_per_side == 2
    assert cfg.sac.total_timesteps > 0

def test_fast_demo_uses_smaller_budget_than_default():
    default = load_config("configs/default.yaml")
    fast = load_config("configs/fast_demo.yaml")
    assert fast.sac.total_timesteps < default.sac.total_timesteps
    assert fast.transformer.epochs < default.transformer.epochs
```

- [ ] **Step 5: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'market_maker'`

- [ ] **Step 6: Implement `config.py` with Pydantic models**

```python
# src/market_maker/config.py
from __future__ import annotations
from pydantic import BaseModel
import yaml


class ProjectConfig(BaseModel):
    seed: int = 42
    device: str = "auto"


class SplitConfig(BaseModel):
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15


class DataConfig(BaseModel):
    lob_path: str
    news_path: str
    symbol: str
    depth: int = 5
    tick_size: float = 0.01
    timezone: str = "UTC"
    split: SplitConfig = SplitConfig()


class SentimentConfig(BaseModel):
    enabled: bool = True
    model_name: str = "ProsusAI/finbert"
    batch_size: int = 32
    lookback_minutes: list[int] = [5, 30, 120]
    max_age_minutes: int = 240


class TransformerConfig(BaseModel):
    enabled: bool = True
    sequence_length: int = 64
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    horizons_minutes: list[int] = [1, 5]
    epochs: int = 10
    batch_size: int = 256


class SimulatorConfig(BaseModel):
    execution_mode: str = "snapshot_approx"
    reference_price_mode: str = "midprice"
    episode_minutes: int = 20
    action_interval_ms: int = 500
    total_volume_per_side: float = 20
    levels_per_side: int = 2
    min_spread_ticks: int = 1
    max_spread_ticks: int = 8
    max_mid_offset_ticks: int = 4
    max_inventory: float = 100
    latency_steps: int = 0
    fee_per_unit: float = 0.0


class RewardConfig(BaseModel):
    type: str = "truncated_inventory"
    inventory_threshold: float = 40
    inventory_penalty: float = 0.001
    cancellation_penalty: float = 0.0001
    adverse_selection_penalty: float = 0.0
    scale: float = 1.0


class SacConfig(BaseModel):
    total_timesteps: int = 200_000
    learning_rate: float = 0.0003
    buffer_size: int = 200_000
    learning_starts: int = 5_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    ent_coef: str = "auto"


class ExperimentsConfig(BaseModel):
    run_heuristic: bool = True
    run_lob_only: bool = True
    run_lob_transformer: bool = True
    run_lob_sentiment: bool = True
    run_full: bool = True
    run_td3: bool = False


class AppConfig(BaseModel):
    project: ProjectConfig
    data: DataConfig
    sentiment: SentimentConfig
    transformer: TransformerConfig
    simulator: SimulatorConfig
    reward: RewardConfig
    sac: SacConfig
    experiments: ExperimentsConfig


def load_config(path: str) -> AppConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
```

- [ ] **Step 7: Write `configs/default.yaml` and `configs/fast_demo.yaml`**

`configs/default.yaml` (values per spec §28, with `data.symbol: BTCUSDT`, `data.lob_path: data/processed/lob.parquet`, `data.news_path: data/processed/news.parquet` to match Mode B — crypto LOB, FNSPID news validated separately):

```yaml
project:
  seed: 42
  device: auto
data:
  lob_path: data/processed/lob.parquet
  news_path: data/processed/news.parquet
  symbol: BTCUSDT
  depth: 5
  tick_size: 0.01
  timezone: UTC
  split:
    train: 0.70
    validation: 0.15
    test: 0.15
sentiment:
  enabled: true
  model_name: ProsusAI/finbert
  batch_size: 32
  lookback_minutes: [5, 30, 120]
  max_age_minutes: 240
transformer:
  enabled: true
  sequence_length: 64
  d_model: 64
  nhead: 4
  num_layers: 2
  dim_feedforward: 128
  dropout: 0.1
  horizons_minutes: [1, 5]
  epochs: 10
  batch_size: 256
simulator:
  execution_mode: snapshot_approx
  reference_price_mode: midprice
  episode_minutes: 20
  action_interval_ms: 500
  total_volume_per_side: 20
  levels_per_side: 2
  min_spread_ticks: 1
  max_spread_ticks: 8
  max_mid_offset_ticks: 4
  max_inventory: 100
  latency_steps: 0
  fee_per_unit: 0.0
reward:
  type: truncated_inventory
  inventory_threshold: 40
  inventory_penalty: 0.001
  cancellation_penalty: 0.0001
  adverse_selection_penalty: 0.0
  scale: 1.0
sac:
  total_timesteps: 200000
  learning_rate: 0.0003
  buffer_size: 200000
  learning_starts: 5000
  batch_size: 256
  gamma: 0.99
  tau: 0.005
  ent_coef: auto
experiments:
  run_heuristic: true
  run_lob_only: true
  run_lob_transformer: true
  run_lob_sentiment: true
  run_full: true
  run_td3: false
```

`configs/fast_demo.yaml` (copy of default with cheaper budgets):

```yaml
project:
  seed: 42
  device: auto
data:
  lob_path: data/processed/lob.parquet
  news_path: data/processed/news.parquet
  symbol: BTCUSDT
  depth: 5
  tick_size: 0.01
  timezone: UTC
  split:
    train: 0.70
    validation: 0.15
    test: 0.15
sentiment:
  enabled: true
  model_name: ProsusAI/finbert
  batch_size: 16
  lookback_minutes: [5, 30, 120]
  max_age_minutes: 240
transformer:
  enabled: true
  sequence_length: 32
  d_model: 32
  nhead: 2
  num_layers: 1
  dim_feedforward: 64
  dropout: 0.1
  horizons_minutes: [1, 5]
  epochs: 2
  batch_size: 64
simulator:
  execution_mode: snapshot_approx
  reference_price_mode: midprice
  episode_minutes: 5
  action_interval_ms: 500
  total_volume_per_side: 20
  levels_per_side: 2
  min_spread_ticks: 1
  max_spread_ticks: 8
  max_mid_offset_ticks: 4
  max_inventory: 100
  latency_steps: 0
  fee_per_unit: 0.0
reward:
  type: truncated_inventory
  inventory_threshold: 40
  inventory_penalty: 0.001
  cancellation_penalty: 0.0001
  adverse_selection_penalty: 0.0
  scale: 1.0
sac:
  total_timesteps: 2000
  learning_rate: 0.0003
  buffer_size: 5000
  learning_starts: 200
  batch_size: 64
  gamma: 0.99
  tau: 0.005
  ent_coef: auto
experiments:
  run_heuristic: true
  run_lob_only: true
  run_lob_transformer: true
  run_lob_sentiment: true
  run_full: true
  run_td3: false
```

- [ ] **Step 8: Implement `utils/seeds.py` and `utils/logging.py`**

```python
# src/market_maker/utils/seeds.py
import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

```python
# src/market_maker/utils/logging.py
import logging


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 10: Write `BUILD_STATUS.md` and commit**

`BUILD_STATUS.md`:
```markdown
# Build Status

## Completed
- Repo skeleton, setup.cfg/requirements.txt, .gitignore/.env.example
- Config loader (Pydantic) with default.yaml + fast_demo.yaml

## Current blocker
None.

## Next step
Task 2: Pydantic schemas for LOB/news rows + directory structure.

## Run commands
- Tests: `pytest -q`
```

```bash
git add -A
git commit -m "chore: repo skeleton, config loader, seeds, logging"
```

---

### Task 2: Directory structure, Pydantic schemas, sample-data generator

**Files:**
- Create: `data/raw/lob/.gitkeep`, `data/raw/news/.gitkeep`, `data/interim/.gitkeep`, `data/processed/.gitkeep`, `data/samples/.gitkeep`
- Create: `artifacts/{features,sentiment,transformer,rl,evaluation,reports,logs}/.gitkeep`
- Create: `src/market_maker/schemas.py`
- Create: `src/market_maker/data/__init__.py`
- Create: `src/market_maker/data/sample_data.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces: `market_maker.schemas.LobRow` (Pydantic model matching spec §8.1: `timestamp`, `symbol`, `bid_price_{i}`, `bid_size_{i}`, `ask_price_{i}`, `ask_size_{i}` for `i in 1..K`, `trade_price`, `trade_size`, `trade_side`).
- Produces: `market_maker.schemas.NewsRow` (Pydantic model matching spec §8.2: `published_at`, `symbol`, `headline`, `body`, `source`, `url_or_id`).
- Produces: `market_maker.data.sample_data.generate_synthetic_lob(n_rows: int, depth: int, symbol: str, seed: int) -> pandas.DataFrame`.
- Produces: `market_maker.data.sample_data.generate_synthetic_news(n_items: int, symbol: str, lob_df: pandas.DataFrame, seed: int) -> pandas.DataFrame` (headlines timestamped within the LOB's time range, clearly labeled `source="synthetic_stub"`).

- [ ] **Step 1: Create directories**

```bash
cd "/Users/ruthwik.chamala/Desktop/superdev-4:1"
mkdir -p data/raw/lob data/raw/news data/interim data/processed data/samples
mkdir -p artifacts/features artifacts/sentiment artifacts/transformer artifacts/rl artifacts/evaluation artifacts/reports artifacts/logs
touch data/raw/lob/.gitkeep data/raw/news/.gitkeep data/interim/.gitkeep data/processed/.gitkeep data/samples/.gitkeep
touch artifacts/features/.gitkeep artifacts/sentiment/.gitkeep artifacts/transformer/.gitkeep artifacts/rl/.gitkeep artifacts/evaluation/.gitkeep artifacts/reports/.gitkeep artifacts/logs/.gitkeep
```

- [ ] **Step 2: Write the failing schema test**

```python
# tests/test_schemas.py
import pandas as pd
from market_maker.schemas import LobRow, NewsRow
from market_maker.data.sample_data import generate_synthetic_lob, generate_synthetic_news


def test_lob_row_validates_required_fields():
    row = LobRow(
        timestamp="2024-01-01T00:00:00Z", symbol="BTCUSDT",
        bid_price_1=100.0, bid_size_1=1.0, ask_price_1=100.1, ask_size_1=1.0,
        trade_price=None, trade_size=None, trade_side=None,
    )
    assert row.symbol == "BTCUSDT"


def test_news_row_validates_required_fields():
    row = NewsRow(published_at="2024-01-01T00:00:00Z", symbol="BTCUSDT",
                   headline="Test", body=None, source="synthetic_stub", url_or_id=None)
    assert row.headline == "Test"


def test_generate_synthetic_lob_has_expected_columns():
    df = generate_synthetic_lob(n_rows=100, depth=5, symbol="BTCUSDT", seed=42)
    assert len(df) == 100
    for i in range(1, 6):
        assert f"bid_price_{i}" in df.columns
        assert f"ask_price_{i}" in df.columns
    assert (df["ask_price_1"] > df["bid_price_1"]).all()


def test_generate_synthetic_news_is_within_lob_time_range():
    lob_df = generate_synthetic_lob(n_rows=1000, depth=5, symbol="BTCUSDT", seed=42)
    news_df = generate_synthetic_news(n_items=20, symbol="BTCUSDT", lob_df=lob_df, seed=42)
    assert (news_df["published_at"] >= lob_df["timestamp"].min()).all()
    assert (news_df["published_at"] <= lob_df["timestamp"].max()).all()
    assert (news_df["source"] == "synthetic_stub").all()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'market_maker.schemas'`

- [ ] **Step 4: Implement `schemas.py`**

```python
# src/market_maker/schemas.py
from __future__ import annotations
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel


class LobRow(BaseModel):
    timestamp: datetime
    symbol: str
    bid_price_1: float
    bid_size_1: float
    ask_price_1: float
    ask_size_1: float
    trade_price: Optional[float] = None
    trade_size: Optional[float] = None
    trade_side: Optional[Literal["buy", "sell", "unknown"]] = None


class NewsRow(BaseModel):
    published_at: datetime
    symbol: Optional[str] = None
    headline: str
    body: Optional[str] = None
    source: Optional[str] = None
    url_or_id: Optional[str] = None
```

- [ ] **Step 5: Implement `data/sample_data.py`**

```python
# src/market_maker/data/sample_data.py
from __future__ import annotations
import numpy as np
import pandas as pd


def generate_synthetic_lob(n_rows: int, depth: int, symbol: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_rows, freq="500ms", tz="UTC")
    mid = 100.0 + np.cumsum(rng.normal(0, 0.01, n_rows))
    data = {"timestamp": timestamps, "symbol": symbol}
    spread = 0.02
    for level in range(1, depth + 1):
        offset = spread / 2 + (level - 1) * 0.01
        data[f"bid_price_{level}"] = mid - offset
        data[f"bid_size_{level}"] = rng.uniform(0.1, 2.0, n_rows)
        data[f"ask_price_{level}"] = mid + offset
        data[f"ask_size_{level}"] = rng.uniform(0.1, 2.0, n_rows)
    trade_mask = rng.random(n_rows) < 0.3
    data["trade_price"] = np.where(trade_mask, mid, np.nan)
    data["trade_size"] = np.where(trade_mask, rng.uniform(0.01, 0.5, n_rows), np.nan)
    data["trade_side"] = np.where(trade_mask, rng.choice(["buy", "sell"], n_rows), None)
    return pd.DataFrame(data)


def generate_synthetic_news(n_items: int, symbol: str, lob_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t_min, t_max = lob_df["timestamp"].min(), lob_df["timestamp"].max()
    span_ns = (t_max - t_min).value
    offsets_ns = np.sort(rng.integers(0, span_ns, n_items))
    published_at = [t_min + pd.Timedelta(nanoseconds=int(o)) for o in offsets_ns]
    headlines = [f"Synthetic headline {i} about {symbol}" for i in range(n_items)]
    return pd.DataFrame({
        "published_at": published_at,
        "symbol": symbol,
        "headline": headlines,
        "body": None,
        "source": "synthetic_stub",
        "url_or_id": [f"stub-{i}" for i in range(n_items)],
    })
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: LOB/news schemas, directory structure, synthetic sample-data generator"
```

---

### Task 3: LOB/news loaders, validators, chronological splits

**Files:**
- Create: `src/market_maker/data/lob_loader.py`
- Create: `src/market_maker/data/news_loader.py`
- Create: `src/market_maker/data/validators.py`
- Create: `src/market_maker/data/splits.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `market_maker.schemas.LobRow`, `NewsRow`; `market_maker.data.sample_data.generate_synthetic_lob/news`.
- Produces: `market_maker.data.lob_loader.load_lob(path: str) -> pandas.DataFrame`.
- Produces: `market_maker.data.news_loader.load_news(path: str) -> pandas.DataFrame`.
- Produces: `market_maker.data.validators.validate_lob(df: pandas.DataFrame) -> tuple[pandas.DataFrame, dict]` — returns cleaned df + a dict of `{"crossed_removed": int, "negative_size_removed": int, "duplicate_removed": int}`.
- Produces: `market_maker.data.validators.validate_news(df: pandas.DataFrame) -> tuple[pandas.DataFrame, dict]`.
- Produces: `market_maker.data.splits.chronological_split(df: pandas.DataFrame, timestamp_col: str, train: float, validation: float, test: float) -> dict[str, pandas.DataFrame]` (keys `"train"`, `"validation"`, `"test"`; splits by calendar day boundaries, never mid-day).

- [ ] **Step 1: Write failing tests for validators and splits**

```python
# tests/test_data.py
import numpy as np
import pandas as pd
from market_maker.data.sample_data import generate_synthetic_lob
from market_maker.data.validators import validate_lob
from market_maker.data.splits import chronological_split


def test_validate_lob_removes_crossed_book_rows():
    df = generate_synthetic_lob(n_rows=50, depth=1, symbol="BTCUSDT", seed=1)
    df.loc[5, "bid_price_1"] = df.loc[5, "ask_price_1"] + 1.0  # force crossed
    cleaned, report = validate_lob(df)
    assert report["crossed_removed"] == 1
    assert (cleaned["bid_price_1"] < cleaned["ask_price_1"]).all()


def test_validate_lob_removes_negative_sizes():
    df = generate_synthetic_lob(n_rows=50, depth=1, symbol="BTCUSDT", seed=1)
    df.loc[3, "bid_size_1"] = -1.0
    cleaned, report = validate_lob(df)
    assert report["negative_size_removed"] == 1
    assert (cleaned["bid_size_1"] >= 0).all()


def test_validate_lob_removes_exact_duplicates():
    df = generate_synthetic_lob(n_rows=50, depth=1, symbol="BTCUSDT", seed=1)
    dup_row = df.iloc[[0]]
    df = pd.concat([df, dup_row], ignore_index=True)
    cleaned, report = validate_lob(df)
    assert report["duplicate_removed"] == 1
    assert len(cleaned) == 50


def test_chronological_split_has_no_overlap_and_is_ordered():
    df = generate_synthetic_lob(n_rows=100_000, depth=1, symbol="BTCUSDT", seed=1)
    splits = chronological_split(df, "timestamp", 0.70, 0.15, 0.15)
    assert splits["train"]["timestamp"].max() <= splits["validation"]["timestamp"].min()
    assert splits["validation"]["timestamp"].max() <= splits["test"]["timestamp"].min()
    total = len(splits["train"]) + len(splits["validation"]) + len(splits["test"])
    assert total == len(df)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'market_maker.data.validators'`

- [ ] **Step 3: Implement `validators.py`**

```python
# src/market_maker/data/validators.py
from __future__ import annotations
import pandas as pd
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)


def validate_lob(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"crossed_removed": 0, "negative_size_removed": 0, "duplicate_removed": 0}
    original_len = len(df)

    crossed_mask = df["bid_price_1"] >= df["ask_price_1"]
    report["crossed_removed"] = int(crossed_mask.sum())
    df = df.loc[~crossed_mask].copy()

    size_cols = [c for c in df.columns if "size" in c]
    negative_mask = (df[size_cols] < 0).any(axis=1)
    report["negative_size_removed"] = int(negative_mask.sum())
    df = df.loc[~negative_mask].copy()

    before_dedup = len(df)
    df = df.drop_duplicates(keep="first")
    report["duplicate_removed"] = before_dedup - len(df)

    df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info(
        "validate_lob: input=%d output=%d removed=%s",
        original_len, len(df), report,
    )
    return df, report


def validate_news(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {"missing_timestamp_removed": 0, "duplicate_removed": 0}
    original_len = len(df)

    missing_mask = df["published_at"].isna() | df["headline"].isna()
    report["missing_timestamp_removed"] = int(missing_mask.sum())
    df = df.loc[~missing_mask].copy()

    before_dedup = len(df)
    df = df.drop_duplicates(keep="first")
    report["duplicate_removed"] = before_dedup - len(df)

    df = df.sort_values("published_at").reset_index(drop=True)
    logger.info(
        "validate_news: input=%d output=%d removed=%s",
        original_len, len(df), report,
    )
    return df, report
```

- [ ] **Step 4: Implement `splits.py`**

```python
# src/market_maker/data/splits.py
from __future__ import annotations
import pandas as pd


def chronological_split(
    df: pd.DataFrame, timestamp_col: str, train: float, validation: float, test: float
) -> dict[str, pd.DataFrame]:
    assert abs(train + validation + test - 1.0) < 1e-6, "split fractions must sum to 1"
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    days = pd.to_datetime(df[timestamp_col]).dt.date
    unique_days = sorted(days.unique())
    n_days = len(unique_days)
    n_train = max(1, int(n_days * train))
    n_val = max(1, int(n_days * validation))
    train_days = set(unique_days[:n_train])
    val_days = set(unique_days[n_train:n_train + n_val])
    test_days = set(unique_days[n_train + n_val:])
    return {
        "train": df.loc[days.isin(train_days)].reset_index(drop=True),
        "validation": df.loc[days.isin(val_days)].reset_index(drop=True),
        "test": df.loc[days.isin(test_days)].reset_index(drop=True),
    }
```

- [ ] **Step 5: Implement `lob_loader.py` and `news_loader.py`**

```python
# src/market_maker/data/lob_loader.py
from __future__ import annotations
import pandas as pd

REQUIRED_LOB_COLUMNS = [
    "timestamp", "symbol", "bid_price_1", "bid_size_1", "ask_price_1", "ask_size_1",
]


def load_lob(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    missing = [c for c in REQUIRED_LOB_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"LOB file {path} missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
```

```python
# src/market_maker/data/news_loader.py
from __future__ import annotations
import pandas as pd

REQUIRED_NEWS_COLUMNS = ["published_at", "headline"]


def load_news(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    missing = [c for c in REQUIRED_NEWS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"News file {path} missing required columns: {missing}")
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_data.py -v`
Expected: PASS (4 tests)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: LOB/news loaders, validators, chronological split"
```

---

## Phase 2 — Features (LOB + Sentiment)

### Task 4: LOB feature engineering

**Files:**
- Create: `src/market_maker/features/__init__.py`
- Create: `src/market_maker/features/lob_features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: validated LOB `pandas.DataFrame` from Task 3 (`validate_lob` output).
- Produces: `market_maker.features.lob_features.compute_lob_features(df: pandas.DataFrame, depth: int, return_lags: list[int]) -> pandas.DataFrame` — returns a new DataFrame with columns `midprice`, `spread`, `relative_spread`, `microprice`, `imbalance_l1`, `imbalance_cum`, `return_lag_{n}` for each lag, `rolling_vol`, `time_of_session`, plus level-relative price/size columns; no NaN/inf in output.
- Produces: `market_maker.features.lob_features.fit_scaler(df: pandas.DataFrame, feature_cols: list[str]) -> sklearn.preprocessing.StandardScaler`.
- Produces: `market_maker.features.lob_features.apply_scaler(df: pandas.DataFrame, feature_cols: list[str], scaler) -> pandas.DataFrame`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_features.py
import numpy as np
from market_maker.data.sample_data import generate_synthetic_lob
from market_maker.features.lob_features import compute_lob_features, fit_scaler, apply_scaler


def test_compute_lob_features_has_no_nan_or_inf():
    df = generate_synthetic_lob(n_rows=500, depth=5, symbol="BTCUSDT", seed=7)
    feats = compute_lob_features(df, depth=5, return_lags=[1, 5])
    numeric = feats.select_dtypes(include=[np.number])
    assert not numeric.isna().any().any()
    assert np.isfinite(numeric.to_numpy()).all()


def test_midprice_and_spread_are_correct():
    df = generate_synthetic_lob(n_rows=10, depth=1, symbol="BTCUSDT", seed=7)
    feats = compute_lob_features(df, depth=1, return_lags=[1])
    expected_mid = (df["bid_price_1"] + df["ask_price_1"]) / 2
    expected_spread = df["ask_price_1"] - df["bid_price_1"]
    assert np.allclose(feats["midprice"], expected_mid)
    assert np.allclose(feats["spread"], expected_spread)


def test_scaler_fit_on_train_reused_on_validation():
    df = generate_synthetic_lob(n_rows=1000, depth=5, symbol="BTCUSDT", seed=7)
    feats = compute_lob_features(df, depth=5, return_lags=[1, 5])
    feature_cols = ["midprice", "spread"]
    train_feats, val_feats = feats.iloc[:700], feats.iloc[700:]
    scaler = fit_scaler(train_feats, feature_cols)
    scaled_val = apply_scaler(val_feats, feature_cols, scaler)
    assert abs(scaled_val["midprice"].mean()) < 5  # reused scaler, not refit on val
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'market_maker.features.lob_features'`

- [ ] **Step 3: Implement `lob_features.py`**

```python
# src/market_maker/features/lob_features.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_lob_features(df: pd.DataFrame, depth: int, return_lags: list[int]) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    feats = pd.DataFrame(index=df.index)

    feats["midprice"] = (df["bid_price_1"] + df["ask_price_1"]) / 2
    feats["spread"] = df["ask_price_1"] - df["bid_price_1"]
    feats["relative_spread"] = feats["spread"] / feats["midprice"]

    bid_sz1, ask_sz1 = df["bid_size_1"], df["ask_size_1"]
    feats["imbalance_l1"] = (bid_sz1 - ask_sz1) / (bid_sz1 + ask_sz1 + 1e-9)
    denom = bid_sz1 + ask_sz1 + 1e-9
    feats["microprice"] = (df["bid_price_1"] * ask_sz1 + df["ask_price_1"] * bid_sz1) / denom

    cum_bid, cum_ask = 0.0, 0.0
    for level in range(1, depth + 1):
        bp, bsz = df[f"bid_price_{level}"], df[f"bid_size_{level}"]
        ap, asz = df[f"ask_price_{level}"], df[f"ask_size_{level}"]
        feats[f"bid_price_rel_{level}"] = bp - feats["midprice"]
        feats[f"ask_price_rel_{level}"] = ap - feats["midprice"]
        feats[f"bid_size_log_{level}"] = np.log1p(bsz)
        feats[f"ask_size_log_{level}"] = np.log1p(asz)
        cum_bid = cum_bid + bsz
        cum_ask = cum_ask + asz
    feats["imbalance_cum"] = (cum_bid - cum_ask) / (cum_bid + cum_ask + 1e-9)

    for lag in return_lags:
        feats[f"return_lag_{lag}"] = feats["midprice"].pct_change(lag).fillna(0.0)

    feats["rolling_vol"] = feats["midprice"].pct_change().rolling(20, min_periods=1).std().fillna(0.0)

    if "trade_size" in df.columns:
        feats["rolling_traded_volume"] = df["trade_size"].fillna(0.0).rolling(20, min_periods=1).sum()
    else:
        feats["rolling_traded_volume"] = 0.0

    ts = pd.to_datetime(df["timestamp"], utc=True)
    seconds_in_day = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    feats["time_of_session"] = seconds_in_day / 86400.0

    feats = feats.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    feats.insert(0, "timestamp", df["timestamp"].values)
    return feats


def fit_scaler(df: pd.DataFrame, feature_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].to_numpy())
    return scaler


def apply_scaler(df: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols].to_numpy())
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: LOB feature engineering with fit/apply scaler"
```

---

### Task 5: FinBERT sentiment inference + caching

**Files:**
- Create: `src/market_maker/features/sentiment.py`
- Create: `scripts/run_finbert.py`
- Test: `tests/test_sentiment.py`

**Interfaces:**
- Consumes: validated news `pandas.DataFrame` from Task 3 (`published_at`, `headline`, `body`, `symbol`).
- Produces: `market_maker.features.sentiment.FinbertScorer` class with `__init__(self, model_name: str, batch_size: int, device: str = "cpu")` and `.score(texts: list[str]) -> pandas.DataFrame` (columns `p_positive`, `p_negative`, `p_neutral`, `sentiment_score`), run in `model.eval()` + `torch.no_grad()`.
- Produces: `market_maker.features.sentiment.score_news_dataframe(news_df: pandas.DataFrame, scorer: FinbertScorer) -> pandas.DataFrame` (adds sentiment columns to news_df, keeps `published_at`, `symbol`, `source`).
- Produces: `market_maker.features.sentiment.cache_sentiment(df: pandas.DataFrame, path: str) -> None` / `load_cached_sentiment(path: str) -> pandas.DataFrame | None` (Parquet round-trip; `load_cached_sentiment` returns `None` if file absent).

- [ ] **Step 1: Write failing tests using a stub scorer (no real model download in unit tests)**

```python
# tests/test_sentiment.py
import numpy as np
import pandas as pd
from market_maker.features.sentiment import (
    score_news_dataframe, cache_sentiment, load_cached_sentiment,
)


class StubScorer:
    """Deterministic stand-in for FinbertScorer — avoids downloading weights in unit tests."""
    def score(self, texts: list[str]) -> pd.DataFrame:
        n = len(texts)
        p_pos = np.full(n, 0.6)
        p_neg = np.full(n, 0.1)
        p_neu = np.full(n, 0.3)
        return pd.DataFrame({
            "p_positive": p_pos, "p_negative": p_neg, "p_neutral": p_neu,
            "sentiment_score": p_pos - p_neg,
        })


def test_score_news_dataframe_adds_sentiment_columns():
    news_df = pd.DataFrame({
        "published_at": pd.to_datetime(["2024-01-01T00:00:00Z"]),
        "symbol": ["BTCUSDT"], "headline": ["Great news"], "source": ["synthetic_stub"],
    })
    scored = score_news_dataframe(news_df, StubScorer())
    assert scored.loc[0, "sentiment_score"] == 0.5
    assert set(["p_positive", "p_negative", "p_neutral"]).issubset(scored.columns)


def test_cache_sentiment_round_trip(tmp_path):
    df = pd.DataFrame({
        "published_at": pd.to_datetime(["2024-01-01T00:00:00Z"]),
        "symbol": ["BTCUSDT"], "headline": ["x"], "source": ["s"],
        "p_positive": [0.6], "p_negative": [0.1], "p_neutral": [0.3], "sentiment_score": [0.5],
    })
    path = str(tmp_path / "sentiment.parquet")
    cache_sentiment(df, path)
    loaded = load_cached_sentiment(path)
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), df.reset_index(drop=True))


def test_load_cached_sentiment_returns_none_when_missing(tmp_path):
    assert load_cached_sentiment(str(tmp_path / "missing.parquet")) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sentiment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'market_maker.features.sentiment'`

- [ ] **Step 3: Implement `sentiment.py`**

```python
# src/market_maker/features/sentiment.py
from __future__ import annotations
import os
import pandas as pd
import torch


class FinbertScorer:
    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 32, device: str = "cpu"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.batch_size = batch_size
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        # ProsusAI/finbert label order: positive, negative, neutral
        self.label_order = ["positive", "negative", "neutral"]

    def score(self, texts: list[str]) -> pd.DataFrame:
        rows = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                for row in probs:
                    p = dict(zip(self.label_order, row))
                    rows.append(p)
        df = pd.DataFrame(rows)[["positive", "negative", "neutral"]]
        df.columns = ["p_positive", "p_negative", "p_neutral"]
        df["sentiment_score"] = df["p_positive"] - df["p_negative"]
        return df


def score_news_dataframe(news_df: pd.DataFrame, scorer) -> pd.DataFrame:
    texts = (news_df["headline"].fillna("") + " " + news_df.get("body", "").fillna("")).tolist() \
        if "body" in news_df.columns else news_df["headline"].fillna("").tolist()
    sentiment = scorer.score(texts).reset_index(drop=True)
    result = news_df.reset_index(drop=True).copy()
    for col in ["p_positive", "p_negative", "p_neutral", "sentiment_score"]:
        result[col] = sentiment[col]
    return result


def cache_sentiment(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)


def load_cached_sentiment(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sentiment.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Write `scripts/run_finbert.py` (real-model batch script, not unit-tested against the network)**

```python
# scripts/run_finbert.py
import argparse
from market_maker.config import load_config
from market_maker.data.news_loader import load_news
from market_maker.data.validators import validate_news
from market_maker.features.sentiment import (
    FinbertScorer, score_news_dataframe, cache_sentiment, load_cached_sentiment,
)
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="artifacts/sentiment/news_sentiment.parquet")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cached = load_cached_sentiment(args.out)
    if cached is not None:
        logger.info("Sentiment cache found at %s (%d rows); skipping FinBERT run.", args.out, len(cached))
        return

    news_df = load_news(cfg.data.news_path)
    news_df, report = validate_news(news_df)
    logger.info("Validated news: %d rows, removed=%s", len(news_df), report)

    scorer = FinbertScorer(model_name=cfg.sentiment.model_name, batch_size=cfg.sentiment.batch_size)
    scored = score_news_dataframe(news_df, scorer)
    cache_sentiment(scored, args.out)
    logger.info("Cached %d scored news rows to %s", len(scored), args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: FinBERT sentiment scorer, caching, batch script"
```

---

### Task 6: As-of news alignment + sentiment aggregation (no look-ahead)

**Files:**
- Create: `src/market_maker/features/news_alignment.py`
- Test: `tests/test_news_alignment.py`

**Interfaces:**
- Consumes: LOB feature `pandas.DataFrame` (has `timestamp` column, from Task 4) and scored news `pandas.DataFrame` (has `published_at`, `sentiment_score`, from Task 5).
- Produces: `market_maker.features.news_alignment.align_sentiment_to_market(market_df: pandas.DataFrame, news_df: pandas.DataFrame, lookback_minutes: list[int], max_age_minutes: int) -> pandas.DataFrame` — for each row in `market_df`, adds per-window columns `sentiment_ewm_{w}`, `sentiment_max_abs_{w}`, `article_count_{w}` for each `w` in `lookback_minutes`, plus `minutes_since_last_article` (capped at `max_age_minutes`) and `has_recent_news` (bool mask). Uses only news with `published_at <= timestamp`.

- [ ] **Step 1: Write the mandatory leakage test plus aggregation tests**

```python
# tests/test_news_alignment.py
import pandas as pd
from market_maker.features.news_alignment import align_sentiment_to_market


def _market_df(times):
    return pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True)})


def test_future_news_cannot_leak_backward():
    market_df = _market_df(["2024-01-01T10:00:00Z"])
    news_df = pd.DataFrame({
        "published_at": pd.to_datetime(["2024-01-01T10:00:01Z"], utc=True),  # 1 second AFTER market state
        "sentiment_score": [0.99],  # strongly positive
    })
    aligned = align_sentiment_to_market(market_df, news_df, lookback_minutes=[5], max_age_minutes=240)
    assert aligned.loc[0, "article_count_5"] == 0
    assert aligned.loc[0, "sentiment_ewm_5"] == 0.0
    assert aligned.loc[0, "has_recent_news"] == False


def test_news_at_or_before_timestamp_is_visible():
    market_df = _market_df(["2024-01-01T10:00:01Z"])
    news_df = pd.DataFrame({
        "published_at": pd.to_datetime(["2024-01-01T10:00:00Z"], utc=True),  # 1 second BEFORE
        "sentiment_score": [0.8],
    })
    aligned = align_sentiment_to_market(market_df, news_df, lookback_minutes=[5], max_age_minutes=240)
    assert aligned.loc[0, "article_count_5"] == 1
    assert aligned.loc[0, "has_recent_news"] == True


def test_no_news_returns_default_neutral_features():
    market_df = _market_df(["2024-01-01T10:00:00Z"])
    news_df = pd.DataFrame({"published_at": pd.to_datetime([], utc=True), "sentiment_score": []})
    aligned = align_sentiment_to_market(market_df, news_df, lookback_minutes=[5, 30], max_age_minutes=240)
    assert aligned.loc[0, "sentiment_ewm_5"] == 0.0
    assert aligned.loc[0, "article_count_30"] == 0
    assert aligned.loc[0, "minutes_since_last_article"] == 240
    assert aligned.loc[0, "has_recent_news"] == False


def test_multiple_windows_have_different_counts():
    market_df = _market_df(["2024-01-01T10:30:00Z"])
    news_df = pd.DataFrame({
        "published_at": pd.to_datetime([
            "2024-01-01T10:29:00Z",  # 1 min before -> inside 5-min window
            "2024-01-01T10:05:00Z",  # 25 min before -> outside 5-min, inside 30-min
        ], utc=True),
        "sentiment_score": [0.5, -0.3],
    })
    aligned = align_sentiment_to_market(market_df, news_df, lookback_minutes=[5, 30], max_age_minutes=240)
    assert aligned.loc[0, "article_count_5"] == 1
    assert aligned.loc[0, "article_count_30"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_news_alignment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'market_maker.features.news_alignment'`

- [ ] **Step 3: Implement `news_alignment.py`**

```python
# src/market_maker/features/news_alignment.py
from __future__ import annotations
import numpy as np
import pandas as pd


def align_sentiment_to_market(
    market_df: pd.DataFrame,
    news_df: pd.DataFrame,
    lookback_minutes: list[int],
    max_age_minutes: int,
) -> pd.DataFrame:
    result = market_df.copy().reset_index(drop=True)
    news_sorted = news_df.sort_values("published_at").reset_index(drop=True)
    news_times = news_sorted["published_at"].to_numpy()
    news_scores = news_sorted["sentiment_score"].to_numpy() if len(news_sorted) else np.array([])

    for w in lookback_minutes:
        result[f"sentiment_ewm_{w}"] = 0.0
        result[f"sentiment_max_abs_{w}"] = 0.0
        result[f"article_count_{w}"] = 0
    result["minutes_since_last_article"] = float(max_age_minutes)
    result["has_recent_news"] = False

    for idx, t in enumerate(result["timestamp"]):
        # as-of backward join: only news with published_at <= t is visible
        visible_mask = news_times <= t
        n_visible = int(visible_mask.sum())
        if n_visible == 0:
            continue

        last_pub = news_times[visible_mask][-1]
        age_minutes = (t - last_pub) / np.timedelta64(1, "m")
        result.loc[idx, "minutes_since_last_article"] = min(float(age_minutes), float(max_age_minutes))
        result.loc[idx, "has_recent_news"] = bool(age_minutes <= max_age_minutes)

        for w in lookback_minutes:
            window_mask = visible_mask & (news_times > t - np.timedelta64(w, "m"))
            count = int(window_mask.sum())
            result.loc[idx, f"article_count_{w}"] = count
            if count > 0:
                window_scores = news_scores[window_mask]
                weights = np.exp(np.linspace(-1.0, 0.0, count))
                weights /= weights.sum()
                result.loc[idx, f"sentiment_ewm_{w}"] = float(np.dot(weights, window_scores))
                result.loc[idx, f"sentiment_max_abs_{w}"] = float(np.max(np.abs(window_scores)))

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_news_alignment.py -v`
Expected: PASS (4 tests, including the mandatory leakage test)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: as-of news alignment and sentiment aggregation with leakage test"
```

---

## Phase 3 — Transformer Temporal Encoder

### Task 7: Temporal windowing, Transformer encoder, supervised pretraining

**Files:**
- Create: `src/market_maker/features/temporal_dataset.py`
- Create: `src/market_maker/models/__init__.py`
- Create: `src/market_maker/models/transformer_encoder.py`
- Create: `src/market_maker/models/transformer_train.py`
- Create: `src/market_maker/models/checkpoints.py`
- Create: `scripts/train_transformer.py`
- Test: `tests/test_transformer.py`

**Interfaces:**
- Consumes: LOB feature `pandas.DataFrame` from Task 4 (numeric columns only, `timestamp` excluded from the tensor).
- Produces: `market_maker.features.temporal_dataset.build_windows(feature_df: pandas.DataFrame, feature_cols: list[str], sequence_length: int, horizons_minutes: list[int], tick_threshold: float) -> tuple[numpy.ndarray, dict[int, numpy.ndarray]]` — returns `X` of shape `[n_windows, sequence_length, n_features]` and a dict mapping horizon (minutes) to integer labels `{0: down, 1: flat, 2: up}` of shape `[n_windows]`.
- Produces: `market_maker.models.transformer_encoder.TemporalTransformer(nn.Module)` with `__init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout, horizons: list[int])` and `forward(self, x: Tensor[batch, seq, input_dim]) -> tuple[Tensor[batch, d_model], dict[int, Tensor[batch, 3]]]` (pooled embedding, per-horizon logits).
- Produces: `market_maker.models.transformer_train.train_transformer(model, X_train, y_train_by_horizon, X_val, y_val_by_horizon, epochs, batch_size, lr) -> dict` (returns `{"train_loss": [...], "val_loss": [...]}`).
- Produces: `market_maker.models.checkpoints.save_checkpoint(model, path, extra: dict) -> None` / `load_checkpoint(model_cls, path, **model_kwargs) -> tuple[nn.Module, dict]`.

- [ ] **Step 1: Write failing tests for windowing, forward pass, and checkpoint round-trip**

```python
# tests/test_transformer.py
import numpy as np
import torch
from market_maker.data.sample_data import generate_synthetic_lob
from market_maker.features.lob_features import compute_lob_features
from market_maker.features.temporal_dataset import build_windows
from market_maker.models.transformer_encoder import TemporalTransformer
from market_maker.models.checkpoints import save_checkpoint, load_checkpoint


def _feature_cols(feats):
    return [c for c in feats.columns if c != "timestamp"]


def test_build_windows_shapes_are_correct():
    df = generate_synthetic_lob(n_rows=2000, depth=5, symbol="BTCUSDT", seed=3)
    feats = compute_lob_features(df, depth=5, return_lags=[1, 5])
    cols = _feature_cols(feats)
    X, y_by_horizon = build_windows(feats, cols, sequence_length=32, horizons_minutes=[1, 5], tick_threshold=0.0005)
    assert X.shape[1] == 32
    assert X.shape[2] == len(cols)
    for h, y in y_by_horizon.items():
        assert y.shape[0] == X.shape[0]
        assert set(np.unique(y)).issubset({0, 1, 2})


def test_transformer_forward_pass_shapes():
    model = TemporalTransformer(input_dim=10, d_model=16, nhead=2, num_layers=1,
                                 dim_feedforward=32, dropout=0.1, horizons=[1, 5])
    x = torch.randn(4, 32, 10)
    pooled, logits_by_horizon = model(x)
    assert pooled.shape == (4, 16)
    assert logits_by_horizon[1].shape == (4, 3)
    assert logits_by_horizon[5].shape == (4, 3)


def test_checkpoint_save_and_load_round_trip(tmp_path):
    model = TemporalTransformer(input_dim=10, d_model=16, nhead=2, num_layers=1,
                                 dim_feedforward=32, dropout=0.1, horizons=[1])
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(model, path, extra={"feature_cols": ["a", "b"]})
    loaded_model, extra = load_checkpoint(
        TemporalTransformer, path,
        input_dim=10, d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.1, horizons=[1],
    )
    assert extra["feature_cols"] == ["a", "b"]
    x = torch.randn(2, 32, 10)
    pooled, _ = loaded_model(x)
    assert pooled.shape == (2, 16)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_transformer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `temporal_dataset.py`**

```python
# src/market_maker/features/temporal_dataset.py
from __future__ import annotations
import numpy as np
import pandas as pd


def _label_from_return(ret: float, threshold: float) -> int:
    if ret > threshold:
        return 2  # up
    if ret < -threshold:
        return 0  # down
    return 1  # flat


def build_windows(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    sequence_length: int,
    horizons_minutes: list[int],
    tick_threshold: float,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    values = feature_df[feature_cols].to_numpy(dtype=np.float32)
    timestamps = pd.to_datetime(feature_df["timestamp"], utc=True).to_numpy()
    midprice = feature_df["midprice"].to_numpy() if "midprice" in feature_df.columns else None

    n = len(feature_df)
    max_horizon_ns = max(horizons_minutes) * 60 * 1e9

    windows = []
    labels_by_horizon: dict[int, list[int]] = {h: [] for h in horizons_minutes}

    for end in range(sequence_length, n):
        window_start = end - sequence_length
        t_end = timestamps[end - 1]
        if (t_end - timestamps[-1]).astype("int64") if False else False:
            pass
        # skip windows too close to the end of the series to have future labels
        if (timestamps[-1] - t_end).astype("timedelta64[ns]").astype("int64") < max_horizon_ns:
            continue

        window = values[window_start:end]
        valid = True
        horizon_labels = {}
        for h in horizons_minutes:
            target_time = t_end + np.timedelta64(h, "m")
            future_idx = np.searchsorted(timestamps, target_time)
            if future_idx >= n or midprice is None:
                valid = False
                break
            ret = (midprice[future_idx] - midprice[end - 1]) / midprice[end - 1]
            horizon_labels[h] = _label_from_return(ret, tick_threshold)
        if not valid:
            continue

        windows.append(window)
        for h in horizons_minutes:
            labels_by_horizon[h].append(horizon_labels[h])

    X = np.stack(windows, axis=0) if windows else np.zeros((0, sequence_length, len(feature_cols)), dtype=np.float32)
    y_by_horizon = {h: np.array(v, dtype=np.int64) for h, v in labels_by_horizon.items()}
    return X, y_by_horizon
```

- [ ] **Step 4: Implement `transformer_encoder.py`**

```python
# src/market_maker/models/transformer_encoder.py
from __future__ import annotations
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float, horizons: list[int]):
        super().__init__()
        self.horizons = horizons
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleDict({str(h): nn.Linear(d_model, 3) for h in horizons})

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        h = self.input_proj(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        pooled = self.norm(h.mean(dim=1))
        logits_by_horizon = {horizon: self.heads[str(horizon)](pooled) for horizon in self.horizons}
        return pooled, logits_by_horizon
```

- [ ] **Step 5: Implement `checkpoints.py` and `transformer_train.py`**

```python
# src/market_maker/models/checkpoints.py
from __future__ import annotations
import os
import torch


def save_checkpoint(model, path: str, extra: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "extra": extra}, path)


def load_checkpoint(model_cls, path: str, **model_kwargs):
    payload = torch.load(path, map_location="cpu")
    model = model_cls(**model_kwargs)
    model.load_state_dict(payload["state_dict"])
    return model, payload["extra"]
```

```python
# src/market_maker/models/transformer_train.py
from __future__ import annotations
import numpy as np
import torch
from torch import nn


def train_transformer(model, X_train, y_train_by_horizon, X_val, y_val_by_horizon,
                       epochs: int, batch_size: int, lr: float) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": []}

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = {h: torch.tensor(y, dtype=torch.long) for h, y in y_train_by_horizon.items()}
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = {h: torch.tensor(y, dtype=torch.long) for h, y in y_val_by_horizon.items()}

    n = len(X_train_t)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch_x = X_train_t[idx]
            optimizer.zero_grad()
            _, logits_by_horizon = model(batch_x)
            loss = sum(criterion(logits_by_horizon[h], y_train_t[h][idx]) for h in logits_by_horizon)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        history["train_loss"].append(epoch_loss / n)

        model.eval()
        with torch.no_grad():
            _, val_logits = model(X_val_t)
            val_loss = sum(criterion(val_logits[h], y_val_t[h]) for h in val_logits)
        history["val_loss"].append(val_loss.item())

    return history
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_transformer.py -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Write `scripts/train_transformer.py`**

```python
# scripts/train_transformer.py
import argparse
from market_maker.config import load_config
from market_maker.data.lob_loader import load_lob
from market_maker.data.validators import validate_lob
from market_maker.data.splits import chronological_split
from market_maker.features.lob_features import compute_lob_features
from market_maker.features.temporal_dataset import build_windows
from market_maker.models.transformer_encoder import TemporalTransformer
from market_maker.models.transformer_train import train_transformer
from market_maker.models.checkpoints import save_checkpoint
from market_maker.utils.seeds import set_global_seed
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="artifacts/transformer/checkpoint.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.project.seed)

    df = load_lob(cfg.data.lob_path)
    df, _ = validate_lob(df)
    splits = chronological_split(df, "timestamp", **cfg.data.split.model_dump())
    train_feats = compute_lob_features(splits["train"], depth=cfg.data.depth, return_lags=[1, 5])
    val_feats = compute_lob_features(splits["validation"], depth=cfg.data.depth, return_lags=[1, 5])

    feature_cols = [c for c in train_feats.columns if c != "timestamp"]
    X_train, y_train = build_windows(train_feats, feature_cols, cfg.transformer.sequence_length,
                                      cfg.transformer.horizons_minutes, tick_threshold=0.0005)
    X_val, y_val = build_windows(val_feats, feature_cols, cfg.transformer.sequence_length,
                                  cfg.transformer.horizons_minutes, tick_threshold=0.0005)

    model = TemporalTransformer(
        input_dim=len(feature_cols), d_model=cfg.transformer.d_model, nhead=cfg.transformer.nhead,
        num_layers=cfg.transformer.num_layers, dim_feedforward=cfg.transformer.dim_feedforward,
        dropout=cfg.transformer.dropout, horizons=cfg.transformer.horizons_minutes,
    )
    history = train_transformer(model, X_train, y_train, X_val, y_val,
                                 epochs=cfg.transformer.epochs, batch_size=cfg.transformer.batch_size, lr=1e-3)
    logger.info("Transformer training finished: %s", history)

    save_checkpoint(model, args.out, extra={
        "feature_cols": feature_cols,
        "sequence_length": cfg.transformer.sequence_length,
        "horizons_minutes": cfg.transformer.horizons_minutes,
        "history": history,
    })
    logger.info("Saved transformer checkpoint to %s", args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: temporal windowing, Transformer encoder, supervised pretraining"
```

---

### Task 8: Feature fusion

**Files:**
- Create: `src/market_maker/features/fusion.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Consumes: LOB features (Task 4), aligned sentiment (Task 6), Transformer checkpoint + pooled embedding/probabilities (Task 7).
- Produces: `market_maker.features.fusion.build_fused_dataset(lob_feats: pandas.DataFrame, aligned_sentiment: pandas.DataFrame, transformer_embeddings: numpy.ndarray, transformer_probs: dict[int, numpy.ndarray], private_state_placeholder_dim: int) -> tuple[numpy.ndarray, list[str]]` — returns a `float32` array `[n_rows, total_dim]` (LOB numeric cols + sentiment cols + transformer embedding dims + transformer prob dims, with the private-state slice zero-filled as a placeholder reserved for the env to fill at runtime) and the ordered list of column names, clipped to `[-10, 10]`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_fusion.py
import numpy as np
import pandas as pd
from market_maker.features.fusion import build_fused_dataset


def test_build_fused_dataset_shape_and_clipping():
    n = 5
    lob_feats = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="s", tz="UTC"),
                               "midprice": [100.0] * n, "spread": [1e6] * n})  # extreme value to test clipping
    aligned_sentiment = pd.DataFrame({"sentiment_ewm_5": [0.5] * n, "article_count_5": [1] * n})
    transformer_embeddings = np.random.randn(n, 4).astype(np.float32)
    transformer_probs = {1: np.random.dirichlet([1, 1, 1], size=n).astype(np.float32)}

    fused, columns = build_fused_dataset(
        lob_feats, aligned_sentiment, transformer_embeddings, transformer_probs,
        private_state_placeholder_dim=3,
    )
    expected_dim = 2 + 2 + 4 + 3 + 3  # midprice+spread, sentiment cols, embedding, probs, private placeholder
    assert fused.shape == (n, expected_dim)
    assert fused.dtype == np.float32
    assert fused.max() <= 10.0 and fused.min() >= -10.0
    assert len(columns) == expected_dim
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `fusion.py`**

```python
# src/market_maker/features/fusion.py
from __future__ import annotations
import numpy as np
import pandas as pd


def build_fused_dataset(
    lob_feats: pd.DataFrame,
    aligned_sentiment: pd.DataFrame,
    transformer_embeddings: np.ndarray,
    transformer_probs: dict[int, np.ndarray],
    private_state_placeholder_dim: int,
) -> tuple[np.ndarray, list[str]]:
    lob_cols = [c for c in lob_feats.columns if c != "timestamp"]
    lob_array = lob_feats[lob_cols].to_numpy(dtype=np.float32)

    sentiment_cols = list(aligned_sentiment.columns)
    sentiment_array = aligned_sentiment[sentiment_cols].to_numpy(dtype=np.float32)

    prob_cols = []
    prob_arrays = []
    for horizon in sorted(transformer_probs.keys()):
        probs = transformer_probs[horizon].astype(np.float32)
        prob_arrays.append(probs)
        prob_cols.extend([f"transformer_prob_h{horizon}_{cls}" for cls in ("down", "flat", "up")])
    prob_array = np.concatenate(prob_arrays, axis=1) if prob_arrays else np.zeros((len(lob_feats), 0), dtype=np.float32)

    embedding_cols = [f"transformer_embed_{i}" for i in range(transformer_embeddings.shape[1])]
    private_cols = [f"private_placeholder_{i}" for i in range(private_state_placeholder_dim)]
    private_array = np.zeros((len(lob_feats), private_state_placeholder_dim), dtype=np.float32)

    fused = np.concatenate(
        [lob_array, sentiment_array, transformer_embeddings.astype(np.float32), prob_array, private_array],
        axis=1,
    )
    fused = np.clip(fused, -10.0, 10.0).astype(np.float32)

    columns = lob_cols + sentiment_cols + embedding_cols + prob_cols + private_cols
    return fused, columns
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fusion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: feature fusion combining LOB, sentiment, and Transformer signals"
```

---

## Phase 4 — Execution, Accounting, Reward, Environment

### Task 9: Execution / fill simulator (snapshot-approximation mode)

**Files:**
- Create: `src/market_maker/env/__init__.py`
- Create: `src/market_maker/env/execution.py`
- Test: `tests/test_execution.py`

**Interfaces:**
- Produces: `market_maker.env.execution.Order` dataclass (`side: Literal["bid","ask"]`, `price: float`, `volume: float`).
- Produces: `market_maker.env.execution.SnapshotExecutionSimulator` class with `__init__(self, tick_size: float, latency_steps: int = 0)` and `.step(outstanding_orders: list[Order], next_best_bid: float, next_best_ask: float, next_trade_price: float | None, next_trade_side: str | None, next_trade_size: float | None) -> list[dict]` — returns fill events `{"side": ..., "price": ..., "volume": ...}`, each fill volume `<=` the corresponding order's remaining volume, and no fill invented from a price the market never traded through or moved through.

- [ ] **Step 1: Write failing tests covering Scenario E/F (spec §42) and basic fills**

```python
# tests/test_execution.py
from market_maker.env.execution import Order, SnapshotExecutionSimulator


def test_bid_fills_when_ask_moves_through_price():
    sim = SnapshotExecutionSimulator(tick_size=0.01)
    orders = [Order(side="bid", price=100.0, volume=1.0)]
    fills = sim.step(orders, next_best_bid=99.5, next_best_ask=99.9,  # ask moved below our bid price
                      next_trade_price=None, next_trade_side=None, next_trade_size=None)
    assert len(fills) == 1
    assert fills[0]["side"] == "bid"
    assert fills[0]["volume"] == 1.0


def test_bid_does_not_fill_when_market_never_reaches_price():
    sim = SnapshotExecutionSimulator(tick_size=0.01)
    orders = [Order(side="bid", price=90.0, volume=1.0)]
    fills = sim.step(orders, next_best_bid=99.5, next_best_ask=100.1,
                      next_trade_price=100.0, next_trade_side="sell", next_trade_size=0.5)
    assert fills == []  # Scenario E: no trade reaches the passive quote


def test_partial_fill_capped_by_available_trade_volume():
    sim = SnapshotExecutionSimulator(tick_size=0.01)
    orders = [Order(side="ask", price=100.0, volume=5.0)]
    fills = sim.step(orders, next_best_bid=100.1, next_best_ask=100.2,  # bid moved above our ask
                      next_trade_price=100.0, next_trade_side="buy", next_trade_size=2.0)
    assert len(fills) == 1
    assert fills[0]["volume"] == 2.0  # capped by available trade volume, not full 5.0


def test_ask_fills_when_bid_moves_through_price():
    sim = SnapshotExecutionSimulator(tick_size=0.01)
    orders = [Order(side="ask", price=100.0, volume=1.0)]
    fills = sim.step(orders, next_best_bid=100.5, next_best_ask=100.6,
                      next_trade_price=None, next_trade_side=None, next_trade_size=None)
    assert len(fills) == 1
    assert fills[0]["side"] == "ask"


def test_no_overfill_beyond_outstanding_order_volume():
    sim = SnapshotExecutionSimulator(tick_size=0.01)
    orders = [Order(side="bid", price=100.0, volume=1.0)]
    fills = sim.step(orders, next_best_bid=99.0, next_best_ask=99.5,
                      next_trade_price=99.0, next_trade_side="sell", next_trade_size=10.0)
    assert fills[0]["volume"] == 1.0  # never exceeds outstanding order volume
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_execution.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `execution.py`**

```python
# src/market_maker/env/execution.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class Order:
    side: Literal["bid", "ask"]
    price: float
    volume: float


class SnapshotExecutionSimulator:
    """Fills are approximated from snapshot best bid/ask movement and aggregate trades,
    not a full message-level exchange reconstruction (spec section 15.3 disclaimer)."""

    def __init__(self, tick_size: float, latency_steps: int = 0):
        self.tick_size = tick_size
        self.latency_steps = latency_steps

    def step(
        self,
        outstanding_orders: list[Order],
        next_best_bid: float,
        next_best_ask: float,
        next_trade_price: Optional[float],
        next_trade_side: Optional[str],
        next_trade_size: Optional[float],
    ) -> list[dict]:
        fills = []
        for order in outstanding_orders:
            fillable_volume = 0.0

            if order.side == "bid":
                market_reached = next_best_ask <= order.price
                trade_reached = (
                    next_trade_price is not None
                    and next_trade_side == "sell"
                    and next_trade_price <= order.price
                )
                if market_reached and not trade_reached:
                    fillable_volume = order.volume
                elif trade_reached:
                    fillable_volume = min(order.volume, next_trade_size or 0.0)
                elif market_reached:
                    fillable_volume = order.volume

            elif order.side == "ask":
                market_reached = next_best_bid >= order.price
                trade_reached = (
                    next_trade_price is not None
                    and next_trade_side == "buy"
                    and next_trade_price >= order.price
                )
                if trade_reached:
                    fillable_volume = min(order.volume, next_trade_size or 0.0)
                elif market_reached:
                    fillable_volume = order.volume

            fillable_volume = min(fillable_volume, order.volume)
            if fillable_volume > 0:
                fills.append({"side": order.side, "price": order.price, "volume": fillable_volume})

        return fills
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_execution.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: snapshot-approximation execution simulator with fill tests"
```

---

### Task 10: Accounting and reward

**Files:**
- Create: `src/market_maker/env/accounting.py`
- Create: `src/market_maker/env/reward.py`
- Test: `tests/test_accounting.py`
- Test: `tests/test_reward.py`

**Interfaces:**
- Produces: `market_maker.env.accounting.AccountState` dataclass (`cash: float`, `inventory: float`, `realized_pnl: float`, `fees: float`, `filled_bid_volume: float`, `filled_ask_volume: float`, `turnover: float`) with methods `.apply_fill(side: Literal["bid","ask"], price: float, volume: float, fee_per_unit: float) -> None` and `.mark_to_market(midprice: float) -> float` (returns `total_pnl`).
- Produces: `market_maker.env.reward.compute_reward(delta_mtm_pnl: float, inventory: float, canceled_volume: float, reward_type: str, inventory_threshold: float, inventory_penalty: float, cancellation_penalty: float, scale: float) -> tuple[float, dict]` — returns scaled scalar reward plus a components dict `{"pnl": ..., "inventory_penalty": ..., "cancellation_penalty": ...}`.

- [ ] **Step 1: Write failing accounting tests (buy/sell sign convention, spec §16)**

```python
# tests/test_accounting.py
from market_maker.env.accounting import AccountState


def test_buy_fill_increases_inventory_and_decreases_cash():
    acct = AccountState(cash=1000.0, inventory=0.0, realized_pnl=0.0, fees=0.0,
                         filled_bid_volume=0.0, filled_ask_volume=0.0, turnover=0.0)
    acct.apply_fill(side="bid", price=100.0, volume=2.0, fee_per_unit=0.0)
    assert acct.inventory == 2.0
    assert acct.cash == 1000.0 - 200.0
    assert acct.filled_bid_volume == 2.0


def test_sell_fill_decreases_inventory_and_increases_cash():
    acct = AccountState(cash=1000.0, inventory=5.0, realized_pnl=0.0, fees=0.0,
                         filled_bid_volume=0.0, filled_ask_volume=0.0, turnover=0.0)
    acct.apply_fill(side="ask", price=100.0, volume=2.0, fee_per_unit=0.0)
    assert acct.inventory == 3.0
    assert acct.cash == 1000.0 + 200.0
    assert acct.filled_ask_volume == 2.0


def test_fees_are_deducted_from_cash():
    acct = AccountState(cash=1000.0, inventory=0.0, realized_pnl=0.0, fees=0.0,
                         filled_bid_volume=0.0, filled_ask_volume=0.0, turnover=0.0)
    acct.apply_fill(side="bid", price=100.0, volume=1.0, fee_per_unit=0.5)
    assert acct.cash == 1000.0 - 100.0 - 0.5
    assert acct.fees == 0.5


def test_mark_to_market_uses_current_midprice():
    acct = AccountState(cash=800.0, inventory=2.0, realized_pnl=0.0, fees=0.0,
                         filled_bid_volume=2.0, filled_ask_volume=0.0, turnover=0.0)
    total_pnl = acct.mark_to_market(midprice=110.0)
    assert total_pnl == 800.0 + 2.0 * 110.0 - 1000.0  # started with implied 1000 wealth basis... see impl note
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_accounting.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `accounting.py`**

Note on Step 1's last test: `mark_to_market` returns `cash + inventory*midprice` relative to the account's own starting wealth basis, tracked internally; adjust the test to assert against the formula directly rather than a fixed constant.

```python
# src/market_maker/env/accounting.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class AccountState:
    cash: float
    inventory: float
    realized_pnl: float
    fees: float
    filled_bid_volume: float
    filled_ask_volume: float
    turnover: float
    _initial_wealth: float = field(default=None, repr=False)

    def __post_init__(self):
        if self._initial_wealth is None:
            self._initial_wealth = self.cash

    def apply_fill(self, side: Literal["bid", "ask"], price: float, volume: float, fee_per_unit: float) -> None:
        fee = fee_per_unit * volume
        if side == "bid":
            self.inventory += volume
            self.cash -= price * volume
            self.filled_bid_volume += volume
        else:
            self.inventory -= volume
            self.cash += price * volume
            self.filled_ask_volume += volume
        self.cash -= fee
        self.fees += fee
        self.turnover += price * volume

    def mark_to_market(self, midprice: float) -> float:
        wealth = self.cash + self.inventory * midprice
        return wealth - self._initial_wealth
```

Update the test's last assertion to match this basis:

```python
def test_mark_to_market_uses_current_midprice():
    acct = AccountState(cash=800.0, inventory=2.0, realized_pnl=0.0, fees=0.0,
                         filled_bid_volume=2.0, filled_ask_volume=0.0, turnover=0.0)
    total_pnl = acct.mark_to_market(midprice=110.0)
    assert total_pnl == (800.0 + 2.0 * 110.0) - 800.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_accounting.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Write failing reward tests**

```python
# tests/test_reward.py
from market_maker.env.reward import compute_reward


def test_positive_mtm_delta_increases_reward():
    reward_pos, _ = compute_reward(delta_mtm_pnl=1.0, inventory=0.0, canceled_volume=0.0,
                                    reward_type="truncated_inventory", inventory_threshold=40,
                                    inventory_penalty=0.001, cancellation_penalty=0.0001, scale=1.0)
    reward_neg, _ = compute_reward(delta_mtm_pnl=-1.0, inventory=0.0, canceled_volume=0.0,
                                    reward_type="truncated_inventory", inventory_threshold=40,
                                    inventory_penalty=0.001, cancellation_penalty=0.0001, scale=1.0)
    assert reward_pos > reward_neg


def test_truncated_inventory_penalty_zero_below_threshold():
    reward, components = compute_reward(delta_mtm_pnl=0.0, inventory=10.0, canceled_volume=0.0,
                                         reward_type="truncated_inventory", inventory_threshold=40,
                                         inventory_penalty=0.5, cancellation_penalty=0.0, scale=1.0)
    assert components["inventory_penalty"] == 0.0


def test_truncated_inventory_penalty_activates_above_threshold():
    reward, components = compute_reward(delta_mtm_pnl=0.0, inventory=50.0, canceled_volume=0.0,
                                         reward_type="truncated_inventory", inventory_threshold=40,
                                         inventory_penalty=0.5, cancellation_penalty=0.0, scale=1.0)
    assert components["inventory_penalty"] == 0.5 * 50.0


def test_quadratic_inventory_penalty():
    reward, components = compute_reward(delta_mtm_pnl=0.0, inventory=3.0, canceled_volume=0.0,
                                         reward_type="quadratic", inventory_threshold=40,
                                         inventory_penalty=0.1, cancellation_penalty=0.0, scale=1.0)
    assert components["inventory_penalty"] == 0.1 * 9.0
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `pytest tests/test_reward.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 7: Implement `reward.py`**

```python
# src/market_maker/env/reward.py
from __future__ import annotations


def compute_reward(
    delta_mtm_pnl: float,
    inventory: float,
    canceled_volume: float,
    reward_type: str,
    inventory_threshold: float,
    inventory_penalty: float,
    cancellation_penalty: float,
    scale: float,
) -> tuple[float, dict]:
    if reward_type == "truncated_inventory":
        inv_pen = inventory_penalty * abs(inventory) if abs(inventory) > inventory_threshold else 0.0
    elif reward_type == "quadratic":
        inv_pen = inventory_penalty * (inventory ** 2)
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

    cancel_pen = cancellation_penalty * canceled_volume
    components = {"pnl": delta_mtm_pnl, "inventory_penalty": inv_pen, "cancellation_penalty": cancel_pen}
    reward = scale * (delta_mtm_pnl - inv_pen - cancel_pen)
    return reward, components
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_reward.py -v`
Expected: PASS (4 tests)

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat: accounting with buy/sell sign convention and configurable reward"
```

---

### Task 11: Gymnasium market-making environment

**Files:**
- Create: `src/market_maker/env/reference_price.py`
- Create: `src/market_maker/env/market_making_env.py`
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: `Order`, `SnapshotExecutionSimulator` (Task 9); `AccountState`, `compute_reward` (Task 10); fused observation array + column list (Task 8).
- Produces: `market_maker.env.reference_price.compute_reference_price(bid_price_1: float, ask_price_1: float, mode: Literal["midprice", "imm_stable"]) -> float`.
- Produces: `market_maker.env.market_making_env.MarketMakingEnv(gymnasium.Env)` — `__init__(self, fused_features: numpy.ndarray, raw_lob_df: pandas.DataFrame, feature_columns: list[str], tick_size: float, levels_per_side: int, min_spread_ticks: int, max_spread_ticks: int, max_mid_offset_ticks: int, total_volume_per_side: float, max_inventory: float, fee_per_unit: float, reward_config: dict, reference_price_mode: str, episode_length_steps: int)`. `observation_space = gymnasium.spaces.Box(shape=(len(feature_columns),))`, `action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(4,))`. Standard `reset(seed=None) -> (obs, info)` / `step(action) -> (obs, reward, terminated, truncated, info)`, with `info` containing `fills`, `canceled_volume`, `inventory`, `cash`.

- [ ] **Step 1: Write failing tests covering Gym compliance and Scenario C/D (spec §42)**

```python
# tests/test_env.py
import numpy as np
from gymnasium.utils.env_checker import check_env
from market_maker.data.sample_data import generate_synthetic_lob
from market_maker.env.market_making_env import MarketMakingEnv


def _make_env(max_inventory=100.0, episode_length_steps=50):
    lob_df = generate_synthetic_lob(n_rows=1000, depth=2, symbol="BTCUSDT", seed=11)
    n = len(lob_df)
    feature_columns = [f"f{i}" for i in range(8)]
    fused = np.random.randn(n, 8).astype(np.float32)
    return MarketMakingEnv(
        fused_features=fused, raw_lob_df=lob_df, feature_columns=feature_columns,
        tick_size=0.01, levels_per_side=2, min_spread_ticks=1, max_spread_ticks=8,
        max_mid_offset_ticks=4, total_volume_per_side=20.0, max_inventory=max_inventory,
        fee_per_unit=0.0, reward_config={"type": "truncated_inventory", "inventory_threshold": 40,
                                          "inventory_penalty": 0.001, "cancellation_penalty": 0.0001, "scale": 1.0},
        reference_price_mode="midprice", episode_length_steps=episode_length_steps,
    )


def test_gymnasium_check_env_passes():
    env = _make_env()
    check_env(env.unwrapped if hasattr(env, "unwrapped") else env, skip_render_check=True)


def test_observation_always_within_observation_space():
    env = _make_env()
    obs, _ = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break


def test_inventory_never_exceeds_max_long_limit():
    env = _make_env(max_inventory=5.0)
    env.reset(seed=42)
    max_buy_action = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)  # push mid up, tight spread, all bid volume
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(max_buy_action)
        assert info["inventory"] <= 5.0 + 1e-6
        if terminated or truncated:
            break


def test_inventory_never_exceeds_max_short_limit():
    env = _make_env(max_inventory=5.0)
    env.reset(seed=42)
    max_sell_action = np.array([-1.0, -1.0, -1.0, 1.0], dtype=np.float32)
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(max_sell_action)
        assert info["inventory"] >= -5.0 - 1e-6
        if terminated or truncated:
            break


def test_episode_terminates_at_configured_length():
    env = _make_env(episode_length_steps=10)
    env.reset(seed=42)
    steps = 0
    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        if terminated or truncated:
            break
    assert steps <= 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_env.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `reference_price.py`**

```python
# src/market_maker/env/reference_price.py
from __future__ import annotations
from typing import Literal


def compute_reference_price(bid_price_1: float, ask_price_1: float, mode: Literal["midprice", "imm_stable"]) -> float:
    if mode == "midprice":
        return (bid_price_1 + ask_price_1) / 2
    if mode == "imm_stable":
        # Approximation only: without message-level queue data, "stable" reference
        # falls back to midprice. This must not be presented as an exact IMM reproduction (spec 14.3).
        return (bid_price_1 + ask_price_1) / 2
    raise ValueError(f"Unknown reference_price_mode: {mode}")
```

- [ ] **Step 4: Implement `market_making_env.py`**

```python
# src/market_maker/env/market_making_env.py
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

from market_maker.env.execution import Order, SnapshotExecutionSimulator
from market_maker.env.accounting import AccountState
from market_maker.env.reward import compute_reward
from market_maker.env.reference_price import compute_reference_price


class MarketMakingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, fused_features: np.ndarray, raw_lob_df: pd.DataFrame, feature_columns: list[str],
                 tick_size: float, levels_per_side: int, min_spread_ticks: int, max_spread_ticks: int,
                 max_mid_offset_ticks: int, total_volume_per_side: float, max_inventory: float,
                 fee_per_unit: float, reward_config: dict, reference_price_mode: str, episode_length_steps: int):
        super().__init__()
        self.fused_features = fused_features
        self.raw_lob_df = raw_lob_df.reset_index(drop=True)
        self.feature_columns = feature_columns
        self.tick_size = tick_size
        self.levels_per_side = levels_per_side
        self.min_spread_ticks = min_spread_ticks
        self.max_spread_ticks = max_spread_ticks
        self.max_mid_offset_ticks = max_mid_offset_ticks
        self.total_volume_per_side = total_volume_per_side
        self.max_inventory = max_inventory
        self.fee_per_unit = fee_per_unit
        self.reward_config = reward_config
        self.reference_price_mode = reference_price_mode
        self.episode_length_steps = episode_length_steps

        self.execution_sim = SnapshotExecutionSimulator(tick_size=tick_size)

        n_features = fused_features.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._max_start = len(raw_lob_df) - episode_length_steps - 1
        self._rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._step_idx = 0
        self._data_idx = int(self._rng.integers(0, max(1, self._max_start)))
        self.account = AccountState(cash=0.0, inventory=0.0, realized_pnl=0.0, fees=0.0,
                                     filled_bid_volume=0.0, filled_ask_volume=0.0, turnover=0.0)
        self._outstanding_orders: list[Order] = []
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        return self.fused_features[self._data_idx].astype(np.float32)

    def _current_row(self):
        return self.raw_lob_df.iloc[self._data_idx]

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        mid_offset_frac, spread_frac, bid_split_raw, ask_split_raw = action
        bid_split = (bid_split_raw + 1.0) / 2.0
        ask_split = (ask_split_raw + 1.0) / 2.0

        row = self._current_row()
        ref_price = compute_reference_price(row["bid_price_1"], row["ask_price_1"], self.reference_price_mode)

        mid_offset_ticks = mid_offset_frac * self.max_mid_offset_ticks
        spread_ticks = self.min_spread_ticks + (spread_frac + 1.0) / 2.0 * (self.max_spread_ticks - self.min_spread_ticks)
        quote_center = ref_price + mid_offset_ticks * self.tick_size
        half_spread = max(spread_ticks, 1.0) * self.tick_size / 2.0

        bid_price_l1 = self._round_to_tick(quote_center - half_spread)
        bid_price_l2 = self._round_to_tick(bid_price_l1 - self.tick_size)
        ask_price_l1 = self._round_to_tick(quote_center + half_spread)
        ask_price_l2 = self._round_to_tick(ask_price_l1 + self.tick_size)
        if bid_price_l1 >= ask_price_l1:
            ask_price_l1 = bid_price_l1 + self.tick_size
            ask_price_l2 = ask_price_l1 + self.tick_size

        bid_vol_l1 = self.total_volume_per_side * bid_split
        bid_vol_l2 = self.total_volume_per_side * (1 - bid_split)
        ask_vol_l1 = self.total_volume_per_side * ask_split
        ask_vol_l2 = self.total_volume_per_side * (1 - ask_split)

        remaining_long = max(0.0, self.max_inventory - self.account.inventory)
        remaining_short = max(0.0, self.max_inventory + self.account.inventory)
        bid_vol_l1 = min(bid_vol_l1, remaining_long)
        bid_vol_l2 = min(bid_vol_l2, max(0.0, remaining_long - bid_vol_l1))
        ask_vol_l1 = min(ask_vol_l1, remaining_short)
        ask_vol_l2 = min(ask_vol_l2, max(0.0, remaining_short - ask_vol_l1))

        new_orders = [
            Order(side="bid", price=bid_price_l1, volume=bid_vol_l1),
            Order(side="bid", price=bid_price_l2, volume=bid_vol_l2),
            Order(side="ask", price=ask_price_l1, volume=ask_vol_l1),
            Order(side="ask", price=ask_price_l2, volume=ask_vol_l2),
        ]
        canceled_volume = sum(o.volume for o in self._outstanding_orders)
        self._outstanding_orders = [o for o in new_orders if o.volume > 0]

        next_idx = min(self._data_idx + 1, len(self.raw_lob_df) - 1)
        next_row = self.raw_lob_df.iloc[next_idx]
        fills = self.execution_sim.step(
            self._outstanding_orders,
            next_best_bid=next_row["bid_price_1"], next_best_ask=next_row["ask_price_1"],
            next_trade_price=next_row.get("trade_price"), next_trade_side=next_row.get("trade_side"),
            next_trade_size=next_row.get("trade_size"),
        )
        for fill in fills:
            self.account.apply_fill(fill["side"], fill["price"], fill["volume"], self.fee_per_unit)

        prev_mtm = self.account.mark_to_market(ref_price)
        next_ref_price = compute_reference_price(next_row["bid_price_1"], next_row["ask_price_1"], self.reference_price_mode)
        new_mtm = self.account.mark_to_market(next_ref_price)
        delta_mtm = new_mtm - prev_mtm

        reward, components = compute_reward(
            delta_mtm_pnl=delta_mtm, inventory=self.account.inventory, canceled_volume=canceled_volume,
            reward_type=self.reward_config["type"], inventory_threshold=self.reward_config["inventory_threshold"],
            inventory_penalty=self.reward_config["inventory_penalty"],
            cancellation_penalty=self.reward_config["cancellation_penalty"], scale=self.reward_config["scale"],
        )

        self._data_idx = next_idx
        self._step_idx += 1
        terminated = False
        truncated = self._step_idx >= self.episode_length_steps or self._data_idx >= len(self.raw_lob_df) - 1

        obs = self._get_obs()
        info = {"fills": fills, "canceled_volume": canceled_volume, "inventory": self.account.inventory,
                "cash": self.account.cash, "reward_components": components}
        return obs, float(reward), terminated, truncated, info

    def _round_to_tick(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_env.py -v`
Expected: PASS (5 tests)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: Gymnasium market-making environment with inventory limits and reference price modes"
```

---

## Phase 5 — Agents and Training

### Task 12: Heuristic baseline agent + SAC training script

**Files:**
- Create: `src/market_maker/agents/__init__.py`
- Create: `src/market_maker/agents/heuristic.py`
- Create: `src/market_maker/training/__init__.py`
- Create: `src/market_maker/training/train_sac.py`
- Create: `scripts/train_agent.py`
- Test: `tests/test_heuristic.py`

**Interfaces:**
- Produces: `market_maker.agents.heuristic.HeuristicMarketMaker` class with `.predict(obs: numpy.ndarray, inventory: float, max_inventory: float) -> numpy.ndarray` (returns a 4-dim action in `[-1, 1]`: symmetric base spread, inventory-skew mid offset, fixed 50/50 volume split), matching a scikit-learn-style `.predict()` so it can be swapped into the same evaluation loop as an SB3 model.
- Produces: `market_maker.training.train_sac.train_sac_agent(env, total_timesteps: int, learning_rate: float, buffer_size: int, learning_starts: int, batch_size: int, gamma: float, tau: float, ent_coef: str, seed: int) -> stable_baselines3.SAC`.

- [ ] **Step 1: Write failing heuristic test**

```python
# tests/test_heuristic.py
import numpy as np
from market_maker.agents.heuristic import HeuristicMarketMaker


def test_heuristic_returns_valid_action_shape_and_range():
    agent = HeuristicMarketMaker()
    obs = np.zeros(8, dtype=np.float32)
    action = agent.predict(obs, inventory=0.0, max_inventory=100.0)
    assert action.shape == (4,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_heuristic_skews_mid_offset_against_long_inventory():
    agent = HeuristicMarketMaker()
    obs = np.zeros(8, dtype=np.float32)
    action_flat = agent.predict(obs, inventory=0.0, max_inventory=100.0)
    action_long = agent.predict(obs, inventory=80.0, max_inventory=100.0)
    assert action_long[0] < action_flat[0]  # long inventory -> skew quotes down to encourage selling
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_heuristic.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `heuristic.py`**

```python
# src/market_maker/agents/heuristic.py
from __future__ import annotations
import numpy as np


class HeuristicMarketMaker:
    """Symmetric fixed-spread baseline with inventory-skew mid offset (spec section 20.1, Baseline A)."""

    def __init__(self, base_spread_frac: float = -0.5, skew_strength: float = 1.0):
        self.base_spread_frac = base_spread_frac
        self.skew_strength = skew_strength

    def predict(self, obs: np.ndarray, inventory: float, max_inventory: float) -> np.ndarray:
        inventory_frac = np.clip(inventory / max_inventory, -1.0, 1.0) if max_inventory else 0.0
        mid_offset = -self.skew_strength * inventory_frac  # long -> lower offset -> encourage selling
        action = np.array([mid_offset, self.base_spread_frac, 0.0, 0.0], dtype=np.float32)
        return np.clip(action, -1.0, 1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_heuristic.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Implement `train_sac.py`**

```python
# src/market_maker/training/train_sac.py
from __future__ import annotations
from stable_baselines3 import SAC


def train_sac_agent(env, total_timesteps: int, learning_rate: float, buffer_size: int,
                     learning_starts: int, batch_size: int, gamma: float, tau: float,
                     ent_coef: str, seed: int) -> SAC:
    model = SAC(
        "MlpPolicy", env, learning_rate=learning_rate, buffer_size=buffer_size,
        learning_starts=learning_starts, batch_size=batch_size, gamma=gamma, tau=tau,
        ent_coef=ent_coef, policy_kwargs={"net_arch": [256, 256]}, seed=seed, verbose=0,
    )
    model.learn(total_timesteps=total_timesteps)
    return model
```

- [ ] **Step 6: Write `scripts/train_agent.py`** (wires config → fused features → env → SAC/heuristic, saves checkpoint; this is an integration script exercised end-to-end in Task 17, not unit-tested in isolation)

```python
# scripts/train_agent.py
import argparse
import numpy as np
from market_maker.config import load_config
from market_maker.data.lob_loader import load_lob
from market_maker.data.validators import validate_lob
from market_maker.data.splits import chronological_split
from market_maker.features.lob_features import compute_lob_features
from market_maker.env.market_making_env import MarketMakingEnv
from market_maker.training.train_sac import train_sac_agent
from market_maker.utils.seeds import set_global_seed
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)


def build_env_for_split(cfg, split_df):
    feats = compute_lob_features(split_df, depth=cfg.data.depth, return_lags=[1, 5])
    feature_cols = [c for c in feats.columns if c != "timestamp"]
    fused = feats[feature_cols].to_numpy(dtype=np.float32)
    episode_steps = int((cfg.simulator.episode_minutes * 60 * 1000) / cfg.simulator.action_interval_ms)
    return MarketMakingEnv(
        fused_features=fused, raw_lob_df=split_df.reset_index(drop=True), feature_columns=feature_cols,
        tick_size=cfg.data.tick_size, levels_per_side=cfg.simulator.levels_per_side,
        min_spread_ticks=cfg.simulator.min_spread_ticks, max_spread_ticks=cfg.simulator.max_spread_ticks,
        max_mid_offset_ticks=cfg.simulator.max_mid_offset_ticks,
        total_volume_per_side=cfg.simulator.total_volume_per_side, max_inventory=cfg.simulator.max_inventory,
        fee_per_unit=cfg.simulator.fee_per_unit,
        reward_config=cfg.reward.model_dump(), reference_price_mode=cfg.simulator.reference_price_mode,
        episode_length_steps=episode_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default="SAC_LOB_ONLY")
    parser.add_argument("--out", default="artifacts/rl/model.zip")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.project.seed)

    df = load_lob(cfg.data.lob_path)
    df, _ = validate_lob(df)
    splits = chronological_split(df, "timestamp", **cfg.data.split.model_dump())
    env = build_env_for_split(cfg, splits["train"])

    model = train_sac_agent(
        env, total_timesteps=cfg.sac.total_timesteps, learning_rate=cfg.sac.learning_rate,
        buffer_size=cfg.sac.buffer_size, learning_starts=cfg.sac.learning_starts,
        batch_size=cfg.sac.batch_size, gamma=cfg.sac.gamma, tau=cfg.sac.tau,
        ent_coef=cfg.sac.ent_coef, seed=cfg.project.seed,
    )
    model.save(args.out)
    logger.info("Saved SAC model (%s) to %s", args.variant, args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: heuristic baseline agent and SAC training pipeline"
```

---

## Phase 6 — Evaluation

### Task 13: Evaluation metrics

**Files:**
- Create: `src/market_maker/evaluation/__init__.py`
- Create: `src/market_maker/evaluation/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `market_maker.evaluation.metrics.mean_absolute_position(inventory_series: numpy.ndarray) -> float`.
- Produces: `market_maker.evaluation.metrics.pnl_over_map(episodic_pnl: float, map_value: float) -> float | None` (returns `None` when `map_value == 0`, never raises/inf).
- Produces: `market_maker.evaluation.metrics.max_drawdown(equity_curve: numpy.ndarray) -> float`.
- Produces: `market_maker.evaluation.metrics.adverse_selection_ratio(fills: list[dict], midprice_series: numpy.ndarray, fill_indices: list[int], horizon_steps: int, threshold: float) -> float` (spec §22).
- Produces: `market_maker.evaluation.metrics.summarize_episode(episode_pnls: list[float], inventories: list[numpy.ndarray]) -> dict` (mean, median, std of PnL across episodes).

- [ ] **Step 1: Write failing tests including the zero-MAP division guard (Review Focus item)**

```python
# tests/test_metrics.py
import numpy as np
from market_maker.evaluation.metrics import (
    mean_absolute_position, pnl_over_map, max_drawdown, adverse_selection_ratio, summarize_episode,
)


def test_mean_absolute_position():
    inv = np.array([1.0, -2.0, 3.0, -4.0])
    assert mean_absolute_position(inv) == 2.5


def test_pnl_over_map_normal_case():
    assert pnl_over_map(100.0, 4.0) == 25.0


def test_pnl_over_map_zero_map_returns_none_not_inf_or_crash():
    assert pnl_over_map(100.0, 0.0) is None


def test_max_drawdown_on_known_curve():
    equity = np.array([100.0, 110.0, 90.0, 95.0, 80.0])
    dd = max_drawdown(equity)
    assert dd == 30.0  # peak 110 -> trough 80


def test_adverse_selection_ratio_flags_bid_fill_followed_by_price_drop():
    midprice_series = np.array([100.0, 100.0, 100.0, 95.0, 95.0])
    fills = [{"side": "bid", "price": 100.0, "volume": 1.0}]
    ratio = adverse_selection_ratio(fills, midprice_series, fill_indices=[1], horizon_steps=2, threshold=0.02)
    assert ratio == 1.0


def test_adverse_selection_ratio_zero_when_no_fills():
    ratio = adverse_selection_ratio([], np.array([100.0, 100.0]), fill_indices=[], horizon_steps=2, threshold=0.02)
    assert ratio == 0.0


def test_summarize_episode_returns_mean_median_std():
    summary = summarize_episode(episode_pnls=[10.0, 20.0, 30.0], inventories=[np.array([1, 2]), np.array([0, 3]), np.array([-1, 1])])
    assert summary["mean_pnl"] == 20.0
    assert summary["median_pnl"] == 20.0
    assert summary["std_pnl"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `metrics.py`**

```python
# src/market_maker/evaluation/metrics.py
from __future__ import annotations
import numpy as np


def mean_absolute_position(inventory_series: np.ndarray) -> float:
    return float(np.mean(np.abs(inventory_series)))


def pnl_over_map(episodic_pnl: float, map_value: float) -> float | None:
    if map_value == 0:
        return None
    return episodic_pnl / map_value


def max_drawdown(equity_curve: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = running_max - equity_curve
    return float(np.max(drawdowns))


def adverse_selection_ratio(fills: list[dict], midprice_series: np.ndarray, fill_indices: list[int],
                             horizon_steps: int, threshold: float) -> float:
    if not fills:
        return 0.0
    adverse_count = 0
    for fill, idx in zip(fills, fill_indices):
        future_idx = min(idx + horizon_steps, len(midprice_series) - 1)
        move = (midprice_series[future_idx] - midprice_series[idx]) / midprice_series[idx]
        if fill["side"] == "bid" and move < -threshold:
            adverse_count += 1
        elif fill["side"] == "ask" and move > threshold:
            adverse_count += 1
    return adverse_count / len(fills)


def summarize_episode(episode_pnls: list[float], inventories: list[np.ndarray]) -> dict:
    pnls = np.array(episode_pnls, dtype=float)
    return {
        "mean_pnl": float(np.mean(pnls)),
        "median_pnl": float(np.median(pnls)),
        "std_pnl": float(np.std(pnls)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: evaluation metrics (MAP, PnL/MAP, drawdown, adverse selection)"
```

---

### Task 14: Backtest/evaluation engine, ablation runner, run artifacts

**Files:**
- Create: `src/market_maker/evaluation/backtest.py`
- Create: `src/market_maker/evaluation/ablation.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `MarketMakingEnv` (Task 11), any object with `.predict(obs, ...)` (SB3 `SAC` or `HeuristicMarketMaker`, Task 12), metrics functions (Task 13).
- Produces: `market_maker.evaluation.backtest.run_backtest(env, agent, n_episodes: int, is_heuristic: bool, max_inventory: float) -> dict` — runs `n_episodes` deterministic episodes, returns `{"episode_pnls": [...], "inventories": [...], "fills": [...], "orders": [...]}`.
- Produces: `market_maker.evaluation.backtest.write_run_artifacts(run_dir: str, config: dict, metadata: dict, backtest_result: dict, metrics: dict) -> None` — writes exactly the files listed in spec §23 to `run_dir`.
- Produces: `market_maker.evaluation.ablation.ABLATION_VARIANTS: list[str]` (`["SAC_LOB_ONLY", "SAC_LOB_TRANSFORMER", "SAC_LOB_SENTIMENT", "SAC_FULL"]`) and `market_maker.evaluation.ablation.select_enabled_variants(experiments_config: dict) -> list[str]`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_backtest.py
import json
import os
import numpy as np
from market_maker.data.sample_data import generate_synthetic_lob
from market_maker.env.market_making_env import MarketMakingEnv
from market_maker.agents.heuristic import HeuristicMarketMaker
from market_maker.evaluation.backtest import run_backtest, write_run_artifacts
from market_maker.evaluation.ablation import select_enabled_variants, ABLATION_VARIANTS


def _make_env():
    lob_df = generate_synthetic_lob(n_rows=500, depth=2, symbol="BTCUSDT", seed=5)
    fused = np.random.randn(len(lob_df), 8).astype(np.float32)
    return MarketMakingEnv(
        fused_features=fused, raw_lob_df=lob_df, feature_columns=[f"f{i}" for i in range(8)],
        tick_size=0.01, levels_per_side=2, min_spread_ticks=1, max_spread_ticks=8, max_mid_offset_ticks=4,
        total_volume_per_side=20.0, max_inventory=100.0, fee_per_unit=0.0,
        reward_config={"type": "truncated_inventory", "inventory_threshold": 40, "inventory_penalty": 0.001,
                       "cancellation_penalty": 0.0001, "scale": 1.0},
        reference_price_mode="midprice", episode_length_steps=20,
    )


def test_run_backtest_produces_expected_keys_and_lengths():
    env = _make_env()
    agent = HeuristicMarketMaker()
    result = run_backtest(env, agent, n_episodes=3, is_heuristic=True, max_inventory=100.0)
    assert len(result["episode_pnls"]) == 3
    assert len(result["inventories"]) == 3


def test_write_run_artifacts_creates_required_files(tmp_path):
    env = _make_env()
    agent = HeuristicMarketMaker()
    result = run_backtest(env, agent, n_episodes=2, is_heuristic=True, max_inventory=100.0)
    run_dir = str(tmp_path / "run_001")
    write_run_artifacts(run_dir, config={"seed": 42}, metadata={"seed": 42}, backtest_result=result,
                         metrics={"mean_pnl": 0.0})
    for fname in ["config.yaml", "metadata.json", "metrics.json", "episodes.csv", "fills.csv"]:
        assert os.path.exists(os.path.join(run_dir, fname))
    with open(os.path.join(run_dir, "metadata.json")) as f:
        assert json.load(f)["seed"] == 42


def test_select_enabled_variants_respects_config():
    cfg = {"run_lob_only": True, "run_lob_transformer": False, "run_lob_sentiment": True, "run_full": True,
           "run_heuristic": True, "run_td3": False}
    variants = select_enabled_variants(cfg)
    assert "SAC_LOB_ONLY" in variants
    assert "SAC_LOB_TRANSFORMER" not in variants
    assert "SAC_FULL" in variants
    assert set(ABLATION_VARIANTS) == {"SAC_LOB_ONLY", "SAC_LOB_TRANSFORMER", "SAC_LOB_SENTIMENT", "SAC_FULL"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `ablation.py`**

```python
# src/market_maker/evaluation/ablation.py
from __future__ import annotations

ABLATION_VARIANTS = ["SAC_LOB_ONLY", "SAC_LOB_TRANSFORMER", "SAC_LOB_SENTIMENT", "SAC_FULL"]

_VARIANT_TO_FLAG = {
    "SAC_LOB_ONLY": "run_lob_only",
    "SAC_LOB_TRANSFORMER": "run_lob_transformer",
    "SAC_LOB_SENTIMENT": "run_lob_sentiment",
    "SAC_FULL": "run_full",
}


def select_enabled_variants(experiments_config: dict) -> list[str]:
    return [v for v in ABLATION_VARIANTS if experiments_config.get(_VARIANT_TO_FLAG[v], False)]
```

- [ ] **Step 4: Implement `backtest.py`**

```python
# src/market_maker/evaluation/backtest.py
from __future__ import annotations
import json
import os
import csv
import yaml
import numpy as np


def run_backtest(env, agent, n_episodes: int, is_heuristic: bool, max_inventory: float) -> dict:
    episode_pnls, inventories, all_fills, all_orders = [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        inventory_trace = []
        ep_fills = []
        cumulative_pnl = 0.0
        while not done:
            if is_heuristic:
                action = agent.predict(obs, inventory=env.account.inventory, max_inventory=max_inventory)
            else:
                action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            inventory_trace.append(info["inventory"])
            ep_fills.extend(info["fills"])
            cumulative_pnl += info["reward_components"]["pnl"]
            done = terminated or truncated
        episode_pnls.append(cumulative_pnl)
        inventories.append(np.array(inventory_trace))
        all_fills.append(ep_fills)
        all_orders.append([])

    return {"episode_pnls": episode_pnls, "inventories": inventories, "fills": all_fills, "orders": all_orders}


def write_run_artifacts(run_dir: str, config: dict, metadata: dict, backtest_result: dict, metrics: dict) -> None:
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(run_dir, "episodes.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "pnl"])
        for i, pnl in enumerate(backtest_result["episode_pnls"]):
            writer.writerow([i, pnl])

    with open(os.path.join(run_dir, "fills.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "side", "price", "volume"])
        for i, ep_fills in enumerate(backtest_result["fills"]):
            for fill in ep_fills:
                writer.writerow([i, fill["side"], fill["price"], fill["volume"]])

    with open(os.path.join(run_dir, "orders.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "note"])

    with open(os.path.join(run_dir, "training_curve.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "reward"])

    with open(os.path.join(run_dir, "model_path.txt"), "w") as f:
        f.write(metadata.get("model_path", "N/A"))

    with open(os.path.join(run_dir, "report.md"), "w") as f:
        f.write("# Run report placeholder — see artifacts/reports/FINAL_PROJECT_REPORT.md for the full report.\n")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_backtest.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: backtest engine, run-artifact writer, ablation variant selection"
```

---

### Task 14a: SQLite persistence layer (SQLAlchemy models + DB writer)

**Files:**
- Create: `src/market_maker/db/__init__.py`
- Create: `src/market_maker/db/models.py`
- Create: `src/market_maker/db/session.py`
- Create: `src/market_maker/db/writer.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `backtest_result` dict from `run_backtest` (Task 14).
- Produces: `market_maker.db.models.Base` (SQLAlchemy declarative base) with tables `Run` (`run_id: str PK`, `variant: str`, `seed: int`, `created_at: datetime`, `model_path: str`, `config_json: str`), `Metric` (`id: int PK`, `run_id: str FK`, `name: str`, `value: float`), `Episode` (`id: int PK`, `run_id: str FK`, `episode_index: int`, `pnl: float`), `Fill` (`id: int PK`, `run_id: str FK`, `episode_index: int`, `side: str`, `price: float`, `volume: float`).
- Produces: `market_maker.db.session.get_engine(db_path: str) -> sqlalchemy.Engine` and `get_session(engine) -> sqlalchemy.orm.Session` and `init_db(engine) -> None` (creates tables if absent).
- Produces: `market_maker.db.writer.write_run_to_db(session, run_id: str, variant: str, seed: int, model_path: str, config: dict, backtest_result: dict, metrics: dict) -> None` — upserts one `Run` row plus one `Metric` row per metric, one `Episode` row per episode, one `Fill` row per fill. Calling it twice with the same `run_id` replaces the prior rows for that run rather than duplicating them.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_db.py
from market_maker.db.models import Run, Metric, Episode, Fill
from market_maker.db.session import get_engine, get_session, init_db
from market_maker.db.writer import write_run_to_db


def _session(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path}/test.db")
    init_db(engine)
    return get_session(engine)


def test_write_run_to_db_creates_run_metric_episode_fill_rows(tmp_path):
    session = _session(tmp_path)
    backtest_result = {
        "episode_pnls": [1.0, 2.0],
        "inventories": [],
        "fills": [[{"side": "bid", "price": 100.0, "volume": 1.0}], []],
        "orders": [[], []],
    }
    write_run_to_db(session, run_id="run_1", variant="SAC_LOB_ONLY", seed=42, model_path="N/A",
                     config={"seed": 42}, backtest_result=backtest_result, metrics={"mean_pnl": 1.5})

    assert session.query(Run).filter_by(run_id="run_1").one().variant == "SAC_LOB_ONLY"
    assert session.query(Metric).filter_by(run_id="run_1", name="mean_pnl").one().value == 1.5
    assert session.query(Episode).filter_by(run_id="run_1").count() == 2
    assert session.query(Fill).filter_by(run_id="run_1").count() == 1


def test_write_run_to_db_replaces_existing_run_rows_on_rerun(tmp_path):
    session = _session(tmp_path)
    backtest_result = {"episode_pnls": [1.0], "inventories": [], "fills": [[]], "orders": [[]]}
    write_run_to_db(session, run_id="run_1", variant="SAC_LOB_ONLY", seed=42, model_path="N/A",
                     config={}, backtest_result=backtest_result, metrics={"mean_pnl": 1.0})
    write_run_to_db(session, run_id="run_1", variant="SAC_LOB_ONLY", seed=42, model_path="N/A",
                     config={}, backtest_result=backtest_result, metrics={"mean_pnl": 9.0})

    assert session.query(Run).filter_by(run_id="run_1").count() == 1
    assert session.query(Metric).filter_by(run_id="run_1", name="mean_pnl").one().value == 9.0
    assert session.query(Episode).filter_by(run_id="run_1").count() == 1  # not duplicated
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'market_maker.db'`

- [ ] **Step 3: Implement `db/models.py`**

```python
# src/market_maker/db/models.py
from __future__ import annotations
from datetime import datetime, timezone
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    variant: Mapped[str] = mapped_column(String)
    seed: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    model_path: Mapped[str] = mapped_column(String)
    config_json: Mapped[str] = mapped_column(String)


class Metric(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.run_id"))
    name: Mapped[str] = mapped_column(String)
    value: Mapped[float] = mapped_column(Float)


class Episode(Base):
    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.run_id"))
    episode_index: Mapped[int] = mapped_column(Integer)
    pnl: Mapped[float] = mapped_column(Float)


class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("runs.run_id"))
    episode_index: Mapped[int] = mapped_column(Integer)
    side: Mapped[str] = mapped_column(String)
    price: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
```

- [ ] **Step 4: Implement `db/session.py`**

```python
# src/market_maker/db/session.py
from __future__ import annotations
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from market_maker.db.models import Base


def get_engine(db_path: str) -> Engine:
    return create_engine(db_path, connect_args={"check_same_thread": False})


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def get_session(engine: Engine) -> Session:
    return sessionmaker(bind=engine)()
```

- [ ] **Step 5: Implement `db/writer.py`**

```python
# src/market_maker/db/writer.py
from __future__ import annotations
import json
from market_maker.db.models import Run, Metric, Episode, Fill


def write_run_to_db(session, run_id: str, variant: str, seed: int, model_path: str,
                     config: dict, backtest_result: dict, metrics: dict) -> None:
    session.query(Fill).filter_by(run_id=run_id).delete()
    session.query(Episode).filter_by(run_id=run_id).delete()
    session.query(Metric).filter_by(run_id=run_id).delete()
    session.query(Run).filter_by(run_id=run_id).delete()

    session.add(Run(run_id=run_id, variant=variant, seed=seed, model_path=model_path,
                     config_json=json.dumps(config)))
    for name, value in metrics.items():
        session.add(Metric(run_id=run_id, name=name, value=float(value)))
    for i, pnl in enumerate(backtest_result["episode_pnls"]):
        session.add(Episode(run_id=run_id, episode_index=i, pnl=float(pnl)))
    for i, ep_fills in enumerate(backtest_result["fills"]):
        for fill in ep_fills:
            session.add(Fill(run_id=run_id, episode_index=i, side=fill["side"],
                              price=float(fill["price"]), volume=float(fill["volume"])))
    session.commit()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_db.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: SQLite persistence layer (SQLAlchemy models, session, writer)"
```

---

## Phase 7 — API, Frontend, Report

### Task 15: FastAPI results API (SQLite-backed)

**Files:**
- Create: `src/market_maker/api/__init__.py`
- Create: `src/market_maker/api/main.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `market_maker.db.session.get_engine/get_session/init_db`, `market_maker.db.models.Run/Metric/Episode/Fill` (Task 14a).
- Produces: FastAPI app `market_maker.api.main.app` with endpoints `GET /health`, `GET /runs`, `GET /runs/{run_id}`, `GET /runs/{run_id}/metrics`, `GET /runs/{run_id}/episodes`, `GET /runs/{run_id}/fills`, `GET /runs/{run_id}/equity-curve`, each querying the SQLite DB via a per-request session; `GET /runs/{run_id}` and sub-routes return HTTP 404 with a JSON `{"detail": ...}` body when `run_id` is unknown. CORS enabled for `http://localhost:5173` (Vite dev server) so the React frontend (Task 16) can call it directly.

- [ ] **Step 1: Write failing API tests using a temp SQLite DB**

```python
# tests/test_api.py
from fastapi.testclient import TestClient
from market_maker.db.session import get_engine, get_session, init_db
from market_maker.db.writer import write_run_to_db
import market_maker.api.main as api_main


def _seed_db(tmp_path, run_id="run_test"):
    engine = get_engine(f"sqlite:///{tmp_path}/test.db")
    init_db(engine)
    session = get_session(engine)
    write_run_to_db(
        session, run_id=run_id, variant="SAC_LOB_ONLY", seed=42, model_path="N/A", config={"seed": 42},
        backtest_result={"episode_pnls": [1.0, 2.0], "inventories": [], "fills": [[{"side": "bid", "price": 100.0, "volume": 1.0}], []], "orders": [[], []]},
        metrics={"mean_pnl": 1.5},
    )
    session.close()
    return engine


def test_health_endpoint():
    client = TestClient(api_main.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_runs_and_run_detail_endpoints(tmp_path, monkeypatch):
    engine = _seed_db(tmp_path, "run_test")
    monkeypatch.setattr(api_main, "ENGINE", engine)
    client = TestClient(api_main.app)

    resp = client.get("/runs")
    assert resp.status_code == 200
    assert "run_test" in resp.json()["run_ids"]

    resp = client.get("/runs/run_test/metrics")
    assert resp.status_code == 200
    assert resp.json()["mean_pnl"] == 1.5

    resp = client.get("/runs/run_test/equity-curve")
    assert resp.status_code == 200
    assert resp.json()[-1]["cumulative_pnl"] == 3.0


def test_unknown_run_id_returns_404(tmp_path, monkeypatch):
    engine = get_engine(f"sqlite:///{tmp_path}/empty.db")
    init_db(engine)
    monkeypatch.setattr(api_main, "ENGINE", engine)
    client = TestClient(api_main.app)
    resp = client.get("/runs/does_not_exist/metrics")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `api/main.py`**

```python
# src/market_maker/api/main.py
from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from market_maker.db.session import get_engine, get_session, init_db
from market_maker.db.models import Run, Metric, Episode, Fill

app = FastAPI(title="Market Maker Results API")
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:5173"],
    allow_methods=["GET"], allow_headers=["*"],
)

DB_PATH = os.environ.get("MARKET_MAKER_DB_PATH", "artifacts/market_maker.db")
ENGINE = get_engine(f"sqlite:///{DB_PATH}")
init_db(ENGINE)


def _session():
    return get_session(ENGINE)


def _require_run(session, run_id: str) -> Run:
    run = session.query(Run).filter_by(run_id=run_id).one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/runs")
def list_runs():
    session = _session()
    run_ids = [r.run_id for r in session.query(Run).all()]
    session.close()
    return {"run_ids": sorted(run_ids)}


@app.get("/runs/{run_id}")
def run_detail(run_id: str):
    session = _session()
    run = _require_run(session, run_id)
    result = {"run_id": run.run_id, "variant": run.variant, "seed": run.seed,
              "model_path": run.model_path, "created_at": run.created_at.isoformat()}
    session.close()
    return result


@app.get("/runs/{run_id}/metrics")
def run_metrics(run_id: str):
    session = _session()
    _require_run(session, run_id)
    metrics = {m.name: m.value for m in session.query(Metric).filter_by(run_id=run_id).all()}
    session.close()
    return metrics


@app.get("/runs/{run_id}/episodes")
def run_episodes(run_id: str):
    session = _session()
    _require_run(session, run_id)
    episodes = [{"episode": e.episode_index, "pnl": e.pnl}
                for e in session.query(Episode).filter_by(run_id=run_id).order_by(Episode.episode_index).all()]
    session.close()
    return episodes


@app.get("/runs/{run_id}/fills")
def run_fills(run_id: str):
    session = _session()
    _require_run(session, run_id)
    fills = [{"episode": f.episode_index, "side": f.side, "price": f.price, "volume": f.volume}
              for f in session.query(Fill).filter_by(run_id=run_id).all()]
    session.close()
    return fills


@app.get("/runs/{run_id}/equity-curve")
def run_equity_curve(run_id: str):
    session = _session()
    _require_run(session, run_id)
    episodes = session.query(Episode).filter_by(run_id=run_id).order_by(Episode.episode_index).all()
    session.close()
    cumulative = 0.0
    curve = []
    for e in episodes:
        cumulative += e.pnl
        curve.append({"episode": e.episode_index, "cumulative_pnl": cumulative})
    return curve
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_api.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: read-only FastAPI results API backed by SQLite"
```

---

### Task 16: Automated Markdown report generator

**Files:**
- Create: `src/market_maker/evaluation/report.py`
- Create: `scripts/generate_report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `artifacts/evaluation/<run_id>/` file artifacts (Task 14) — the report is a static document, not part of the live app, so it reads the file-based `metrics.json` per run directly rather than querying SQLite.
- Produces: `market_maker.evaluation.report.generate_report(runs_root: str, output_path: str, template_sections: list[str] | None = None) -> str` — reads every run under `runs_root`, renders the 24-section report from spec §26, writes `output_path`, returns the rendered text. Any section whose data is missing renders the literal string `Not run / insufficient data` rather than a fabricated number.

- [ ] **Step 1: Write failing report test**

```python
# tests/test_report.py
import os
from market_maker.evaluation.backtest import write_run_artifacts
from market_maker.evaluation.report import generate_report


def test_generate_report_includes_actual_metrics(tmp_path):
    run_dir = str(tmp_path / "run_a")
    write_run_artifacts(run_dir, config={"seed": 1}, metadata={"seed": 1, "model_path": "N/A"},
                         backtest_result={"episode_pnls": [5.0], "inventories": [], "fills": [[]], "orders": [[]]},
                         metrics={"mean_pnl": 5.0, "map": 2.0})
    out_path = str(tmp_path / "FINAL_PROJECT_REPORT.md")
    text = generate_report(runs_root=str(tmp_path), output_path=out_path)
    assert os.path.exists(out_path)
    assert "5.0" in text
    assert "Abstract" in text


def test_generate_report_marks_missing_data_explicitly(tmp_path):
    empty_root = str(tmp_path / "empty")
    os.makedirs(empty_root)
    out_path = str(tmp_path / "REPORT.md")
    text = generate_report(runs_root=empty_root, output_path=out_path)
    assert "Not run / insufficient data" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `report.py`**

```python
# src/market_maker/evaluation/report.py
from __future__ import annotations
import json
import os

SECTIONS = [
    "Title", "Abstract", "Problem Statement", "Base Paper Summary",
    "Corrections to Initial Project Assumptions", "Proposed System", "Architecture",
    "Data Description", "FinBERT Methodology", "Transformer Methodology",
    "Market-Making Environment", "SAC Methodology", "Reward Design", "Experimental Setup",
    "Baselines and Ablations", "Results", "PnL Chart", "Inventory Chart",
    "Adverse-Selection Chart", "Discussion", "Limitations", "Future Scope",
    "Reproducibility Instructions", "References",
]

MISSING = "Not run / insufficient data"


def _load_runs(runs_root: str) -> dict:
    runs = {}
    if not os.path.isdir(runs_root):
        return runs
    for run_id in sorted(os.listdir(runs_root)):
        run_dir = os.path.join(runs_root, run_id)
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.isdir(run_dir) and os.path.exists(metrics_path):
            with open(metrics_path) as f:
                runs[run_id] = json.load(f)
    return runs


def generate_report(runs_root: str, output_path: str, template_sections: list[str] | None = None) -> str:
    sections = template_sections or SECTIONS
    runs = _load_runs(runs_root)

    lines = ["# LLM-Enhanced Financial Market Making — Final Project Report", ""]
    lines.append("> Data mode: Mode B — real crypto LOB, FNSPID news validated separately (not causally matched).")
    lines.append("")

    for section in sections:
        lines.append(f"## {section}")
        if section == "Results":
            if not runs:
                lines.append(MISSING)
            else:
                lines.append("| Run | Mean PnL | MAP |")
                lines.append("|---|---:|---:|")
                for run_id, metrics in runs.items():
                    mean_pnl = metrics.get("mean_pnl", MISSING)
                    map_value = metrics.get("map", MISSING)
                    lines.append(f"| {run_id} | {mean_pnl} | {map_value} |")
        elif section in ("PnL Chart", "Inventory Chart", "Adverse-Selection Chart"):
            lines.append(MISSING if not runs else "See artifacts/evaluation/<run_id>/plots/")
        else:
            lines.append(MISSING)
        lines.append("")

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text)
    return text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_report.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write `scripts/generate_report.py`**

```python
# scripts/generate_report.py
import argparse
from market_maker.evaluation.report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="artifacts/evaluation")
    parser.add_argument("--out", default="artifacts/reports/FINAL_PROJECT_REPORT.md")
    args = parser.parse_args()
    generate_report(runs_root=args.runs_root, output_path=args.out)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: automated Markdown report generator"
```

---

### Task 16b: React frontend (Vite) — six result views

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.js`
- Create: `frontend/index.html`
- Create: `frontend/src/main.jsx`
- Create: `frontend/src/App.jsx`
- Create: `frontend/src/api.js`
- Create: `frontend/src/components/RunSelector.jsx`
- Create: `frontend/src/components/OverviewTab.jsx`
- Create: `frontend/src/components/PnlTab.jsx`
- Create: `frontend/src/components/RiskTab.jsx`
- Create: `frontend/src/components/BehaviorTab.jsx`
- Create: `frontend/src/components/SentimentTab.jsx`
- Create: `frontend/src/components/ComparisonTab.jsx`
- Test: `frontend/src/api.test.js` (Vitest)

**Interfaces:**
- Consumes: the FastAPI endpoints from Task 15 (`/runs`, `/runs/{id}`, `/runs/{id}/metrics`, `/runs/{id}/episodes`, `/runs/{id}/fills`, `/runs/{id}/equity-curve`), assumed to run at `http://localhost:8000`.
- Produces: `frontend/src/api.js` exporting `listRuns() -> Promise<string[]>`, `getRunDetail(runId) -> Promise<object>`, `getRunMetrics(runId) -> Promise<object>`, `getRunEpisodes(runId) -> Promise<Array>`, `getRunFills(runId) -> Promise<Array>`, `getEquityCurve(runId) -> Promise<Array>`, each throwing on a non-2xx response with the response body's `detail` as the error message.
- Produces: `App.jsx` rendering a run selector plus six tabs (Overview, PnL/Equity, Inventory/Risk, Market-Making Behavior, Sentiment/Temporal Signals, Model Comparison) matching spec §25, each tab a separate component that renders `"Not run / insufficient data"` when its endpoint 404s or returns empty data, and a persistent banner: "Data mode: Mode B — real crypto LOB; FNSPID news validated separately, not causally matched to this LOB."

- [ ] **Step 1: Scaffold the Vite project**

```bash
cd "/Users/ruthwik.chamala/Desktop/superdev-4:1"
npm create vite@latest frontend -- --template react
cd frontend
npm install
npm install recharts vitest --save-dev
```

- [ ] **Step 2: Write the failing test for `api.js`**

```javascript
// frontend/src/api.test.js
import { describe, it, expect, vi, afterEach } from "vitest";
import { listRuns, getRunMetrics } from "./api";

describe("api client", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("listRuns returns the run_ids array from the response", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ run_ids: ["run_a", "run_b"] }),
    });
    const runs = await listRuns();
    expect(runs).toEqual(["run_a", "run_b"]);
  });

  it("getRunMetrics throws with the server's detail message on 404", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      json: async () => ({ detail: "Run 'missing' not found" }),
    });
    await expect(getRunMetrics("missing")).rejects.toThrow("Run 'missing' not found");
  });
});
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd frontend && npx vitest run`
Expected: FAIL with a module-resolution error for `./api`

- [ ] **Step 4: Implement `api.js`**

```javascript
// frontend/src/api.js
const BASE_URL = "http://localhost:8000";

async function request(path) {
  const resp = await fetch(`${BASE_URL}${path}`);
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.detail || `Request to ${path} failed with status ${resp.status}`);
  }
  return resp.json();
}

export async function listRuns() {
  const data = await request("/runs");
  return data.run_ids;
}

export function getRunDetail(runId) {
  return request(`/runs/${runId}`);
}

export function getRunMetrics(runId) {
  return request(`/runs/${runId}/metrics`);
}

export function getRunEpisodes(runId) {
  return request(`/runs/${runId}/episodes`);
}

export function getRunFills(runId) {
  return request(`/runs/${runId}/fills`);
}

export function getEquityCurve(runId) {
  return request(`/runs/${runId}/equity-curve`);
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd frontend && npx vitest run`
Expected: PASS (2 tests)

- [ ] **Step 6: Implement `RunSelector.jsx`**

```jsx
// frontend/src/components/RunSelector.jsx
import { useEffect, useState } from "react";
import { listRuns } from "../api";

export default function RunSelector({ selectedRun, onSelect }) {
  const [runs, setRuns] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    listRuns()
      .then(setRuns)
      .catch((err) => setError(err.message));
  }, []);

  if (error) return <p>Not run / insufficient data</p>;
  if (runs.length === 0) return <p>No runs found yet.</p>;

  return (
    <select value={selectedRun || ""} onChange={(e) => onSelect(e.target.value)}>
      <option value="" disabled>
        Select an experiment run
      </option>
      {runs.map((runId) => (
        <option key={runId} value={runId}>
          {runId}
        </option>
      ))}
    </select>
  );
}
```

- [ ] **Step 7: Implement `OverviewTab.jsx`, `PnlTab.jsx`, `BehaviorTab.jsx`, `ComparisonTab.jsx`** (data-bearing tabs — each follows the same fetch-on-runId-change pattern)

```jsx
// frontend/src/components/OverviewTab.jsx
import { useEffect, useState } from "react";
import { getRunDetail } from "../api";

export default function OverviewTab({ runId }) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!runId) return;
    getRunDetail(runId).then(setDetail).catch((err) => setError(err.message));
  }, [runId]);

  if (!runId) return <p>Select a run to see its overview.</p>;
  if (error) return <p>Not run / insufficient data</p>;
  if (!detail) return <p>Loading…</p>;

  return (
    <dl>
      <dt>Variant</dt><dd>{detail.variant}</dd>
      <dt>Seed</dt><dd>{detail.seed}</dd>
      <dt>Model path</dt><dd>{detail.model_path}</dd>
      <dt>Created at</dt><dd>{detail.created_at}</dd>
    </dl>
  );
}
```

```jsx
// frontend/src/components/PnlTab.jsx
import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { getEquityCurve } from "../api";

export default function PnlTab({ runId }) {
  const [curve, setCurve] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!runId) return;
    getEquityCurve(runId).then(setCurve).catch((err) => setError(err.message));
  }, [runId]);

  if (!runId) return <p>Select a run to see PnL / equity.</p>;
  if (error || (curve && curve.length === 0)) return <p>Not run / insufficient data</p>;
  if (!curve) return <p>Loading…</p>;

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={curve}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="episode" />
        <YAxis />
        <Tooltip />
        <Line type="monotone" dataKey="cumulative_pnl" stroke="#2563eb" dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
```

```jsx
// frontend/src/components/BehaviorTab.jsx
import { useEffect, useState } from "react";
import { getRunFills } from "../api";

export default function BehaviorTab({ runId }) {
  const [fills, setFills] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!runId) return;
    getRunFills(runId).then(setFills).catch((err) => setError(err.message));
  }, [runId]);

  if (!runId) return <p>Select a run to see market-making behavior.</p>;
  if (error) return <p>Not run / insufficient data</p>;
  if (!fills) return <p>Loading…</p>;
  if (fills.length === 0) return <p>Not run / insufficient data</p>;

  return (
    <table>
      <thead><tr><th>Episode</th><th>Side</th><th>Price</th><th>Volume</th></tr></thead>
      <tbody>
        {fills.map((f, i) => (
          <tr key={i}><td>{f.episode}</td><td>{f.side}</td><td>{f.price}</td><td>{f.volume}</td></tr>
        ))}
      </tbody>
    </table>
  );
}
```

```jsx
// frontend/src/components/ComparisonTab.jsx
import { useEffect, useState } from "react";
import { getRunMetrics } from "../api";

export default function ComparisonTab({ runId }) {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!runId) return;
    getRunMetrics(runId).then(setMetrics).catch((err) => setError(err.message));
  }, [runId]);

  if (!runId) return <p>Select a run to see its metrics.</p>;
  if (error) return <p>Not run / insufficient data</p>;
  if (!metrics) return <p>Loading…</p>;

  return (
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        {Object.entries(metrics).map(([name, value]) => (
          <tr key={name}><td>{name}</td><td>{value}</td></tr>
        ))}
      </tbody>
    </table>
  );
}
```

- [ ] **Step 8: Implement `RiskTab.jsx` and `SentimentTab.jsx`** (no dedicated backend endpoint yet — deliberately labeled, not faked)

```jsx
// frontend/src/components/RiskTab.jsx
export default function RiskTab() {
  return <p>Not run / insufficient data — inventory/risk endpoint not implemented in this MVP.</p>;
}
```

```jsx
// frontend/src/components/SentimentTab.jsx
export default function SentimentTab() {
  return <p>Not run / insufficient data — sentiment/temporal endpoint not implemented in this MVP.</p>;
}
```

- [ ] **Step 9: Implement `App.jsx`**

```jsx
// frontend/src/App.jsx
import { useState } from "react";
import RunSelector from "./components/RunSelector";
import OverviewTab from "./components/OverviewTab";
import PnlTab from "./components/PnlTab";
import RiskTab from "./components/RiskTab";
import BehaviorTab from "./components/BehaviorTab";
import SentimentTab from "./components/SentimentTab";
import ComparisonTab from "./components/ComparisonTab";

const TABS = [
  ["Overview", OverviewTab],
  ["PnL / Equity", PnlTab],
  ["Inventory / Risk", RiskTab],
  ["Market-Making Behavior", BehaviorTab],
  ["Sentiment / Temporal Signals", SentimentTab],
  ["Model Comparison", ComparisonTab],
];

export default function App() {
  const [selectedRun, setSelectedRun] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const ActiveComponent = TABS[activeTab][1];

  return (
    <div style={{ fontFamily: "sans-serif", padding: "1rem" }}>
      <h1>LLM-Enhanced Market Making — Results Dashboard</h1>
      <p style={{ background: "#fef3c7", padding: "0.5rem" }}>
        Data mode: Mode B — real crypto LOB; FNSPID news validated separately, not causally
        matched to this LOB.
      </p>
      <RunSelector selectedRun={selectedRun} onSelect={setSelectedRun} />
      <nav style={{ margin: "1rem 0" }}>
        {TABS.map(([label], i) => (
          <button key={label} onClick={() => setActiveTab(i)} disabled={i === activeTab}>
            {label}
          </button>
        ))}
      </nav>
      <ActiveComponent runId={selectedRun} />
    </div>
  );
}
```

- [ ] **Step 10: Run the frontend test suite one more time, then smoke-test the dev server**

```bash
cd frontend && npx vitest run
npm run dev  # visit http://localhost:5173 with the FastAPI backend running on :8000
```

Expected: Vitest passes; the dev server serves the app and the run selector populates once the backend is running with at least one seeded run.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "feat: React frontend (Vite) with six result views calling the FastAPI backend"
```

---

## Phase 8 — Integration, CLI, Hardening

### Task 17: `run_all.py` orchestration script, Makefile targets, end-to-end integration test

**Files:**
- Create: `scripts/run_all.py`
- Modify: `Makefile` (add all targets from spec §27)
- Test: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: every module built in Tasks 1–16.
- Produces: `scripts/run_all.py` — a CLI script executing, in order: validate data → prepare features → run/cache FinBERT (stub scorer in fast-demo mode) → train/load Transformer → build fused dataset → train SAC (heuristic + at least `SAC_LOB_ONLY`) → evaluate → generate plots (stub matplotlib call) → generate final report; exits non-zero with a clear message on any stage failure.

- [ ] **Step 1: Write the mandatory tiny end-to-end integration test (spec §29)**

```python
# tests/test_end_to_end.py
import os
import numpy as np
from market_maker.data.sample_data import generate_synthetic_lob, generate_synthetic_news
from market_maker.data.validators import validate_lob, validate_news
from market_maker.data.splits import chronological_split
from market_maker.features.lob_features import compute_lob_features
from market_maker.features.sentiment import score_news_dataframe
from market_maker.features.news_alignment import align_sentiment_to_market
from market_maker.features.temporal_dataset import build_windows
from market_maker.models.transformer_encoder import TemporalTransformer
from market_maker.models.transformer_train import train_transformer
from market_maker.agents.heuristic import HeuristicMarketMaker
from market_maker.env.market_making_env import MarketMakingEnv
from market_maker.evaluation.backtest import run_backtest, write_run_artifacts
from market_maker.evaluation.report import generate_report


class _StubScorer:
    def score(self, texts):
        import pandas as pd
        n = len(texts)
        return pd.DataFrame({"p_positive": [0.5] * n, "p_negative": [0.2] * n,
                              "p_neutral": [0.3] * n, "sentiment_score": [0.3] * n})


def test_full_pipeline_runs_end_to_end_on_synthetic_data(tmp_path):
    lob_df = generate_synthetic_lob(n_rows=3000, depth=5, symbol="BTCUSDT", seed=99)
    news_df = generate_synthetic_news(n_items=30, symbol="BTCUSDT", lob_df=lob_df, seed=99)

    lob_df, _ = validate_lob(lob_df)
    news_df, _ = validate_news(news_df)
    splits = chronological_split(lob_df, "timestamp", 0.70, 0.15, 0.15)

    train_feats = compute_lob_features(splits["train"], depth=5, return_lags=[1, 5])
    scored_news = score_news_dataframe(news_df, _StubScorer())
    aligned = align_sentiment_to_market(train_feats, scored_news, lookback_minutes=[5, 30], max_age_minutes=240)
    assert not aligned.isna().any().any()

    feature_cols = [c for c in train_feats.columns if c != "timestamp"]
    X, y_by_horizon = build_windows(train_feats, feature_cols, sequence_length=16,
                                     horizons_minutes=[1], tick_threshold=0.0005)
    assert X.shape[0] > 0
    model = TemporalTransformer(input_dim=len(feature_cols), d_model=16, nhead=2, num_layers=1,
                                 dim_feedforward=32, dropout=0.1, horizons=[1])
    split_point = int(len(X) * 0.8)
    history = train_transformer(model, X[:split_point], {1: y_by_horizon[1][:split_point]},
                                 X[split_point:], {1: y_by_horizon[1][split_point:]},
                                 epochs=1, batch_size=16, lr=1e-3)
    assert len(history["train_loss"]) == 1

    fused = train_feats[feature_cols].to_numpy(dtype=np.float32)
    env = MarketMakingEnv(
        fused_features=fused, raw_lob_df=splits["train"].reset_index(drop=True), feature_columns=feature_cols,
        tick_size=0.01, levels_per_side=2, min_spread_ticks=1, max_spread_ticks=8, max_mid_offset_ticks=4,
        total_volume_per_side=20.0, max_inventory=100.0, fee_per_unit=0.0,
        reward_config={"type": "truncated_inventory", "inventory_threshold": 40, "inventory_penalty": 0.001,
                       "cancellation_penalty": 0.0001, "scale": 1.0},
        reference_price_mode="midprice", episode_length_steps=20,
    )
    agent = HeuristicMarketMaker()
    result = run_backtest(env, agent, n_episodes=2, is_heuristic=True, max_inventory=100.0)
    assert len(result["episode_pnls"]) == 2

    run_dir = str(tmp_path / "artifacts" / "evaluation" / "smoke_run")
    write_run_artifacts(run_dir, config={"seed": 99}, metadata={"seed": 99, "model_path": "N/A"},
                         backtest_result=result, metrics={"mean_pnl": float(np.mean(result["episode_pnls"]))})

    report_path = str(tmp_path / "artifacts" / "reports" / "FINAL_PROJECT_REPORT.md")
    text = generate_report(runs_root=str(tmp_path / "artifacts" / "evaluation"), output_path=report_path)
    assert os.path.exists(report_path)
    assert "smoke_run" in text
```

- [ ] **Step 2: Run test to verify it fails or passes incrementally**

Run: `pytest tests/test_end_to_end.py -v`
Expected: Should PASS immediately since all dependencies exist from Tasks 1–16 — this test's purpose is to catch integration seams the per-module unit tests miss. If it fails, fix the seam it exposes before continuing.

- [ ] **Step 3: Implement `scripts/run_all.py`**

```python
# scripts/run_all.py
import argparse
import subprocess
import sys
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)

STAGES = [
    ("prepare_data", ["python", "scripts/prepare_data.py", "--config"]),
    ("run_finbert", ["python", "scripts/run_finbert.py", "--config"]),
    ("train_transformer", ["python", "scripts/train_transformer.py", "--config"]),
    ("train_agent", ["python", "scripts/train_agent.py", "--config"]),
    ("evaluate", ["python", "scripts/evaluate.py", "--config"]),
    ("generate_report", ["python", "scripts/generate_report.py"]),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    for stage_name, base_cmd in STAGES:
        cmd = base_cmd + ([args.config] if base_cmd[-1] == "--config" else [])
        logger.info("Running stage: %s (%s)", stage_name, " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error("Stage '%s' failed with exit code %d", stage_name, result.returncode)
            sys.exit(result.returncode)
    logger.info("run_all.py completed all stages successfully.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write `scripts/prepare_data.py` and `scripts/evaluate.py`** (thin CLI wrappers around Tasks 3/4/14; needed for `run_all.py` to execute)

```python
# scripts/prepare_data.py
import argparse
import os
from market_maker.config import load_config
from market_maker.data.lob_loader import load_lob
from market_maker.data.news_loader import load_news
from market_maker.data.validators import validate_lob, validate_news
from market_maker.data.splits import chronological_split
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    if not os.path.exists(cfg.data.lob_path):
        raise FileNotFoundError(
            f"LOB data not found at {cfg.data.lob_path}. Run `make sample-data` first, "
            f"or place real Mode-B data (crypto LOB) at this path."
        )

    lob_df = load_lob(cfg.data.lob_path)
    lob_df, lob_report = validate_lob(lob_df)
    splits = chronological_split(lob_df, "timestamp", **cfg.data.split.model_dump())
    logger.info("Prepared LOB: %d train / %d val / %d test rows. Validation report: %s",
                len(splits["train"]), len(splits["validation"]), len(splits["test"]), lob_report)

    if os.path.exists(cfg.data.news_path):
        news_df = load_news(cfg.data.news_path)
        news_df, news_report = validate_news(news_df)
        logger.info("Prepared news: %d rows. Validation report: %s", len(news_df), news_report)
    else:
        logger.info("No news file at %s — proceeding in news-optional mode.", cfg.data.news_path)


if __name__ == "__main__":
    main()
```

```python
# scripts/evaluate.py
import argparse
import os
import numpy as np
from market_maker.config import load_config
from market_maker.data.lob_loader import load_lob
from market_maker.data.validators import validate_lob
from market_maker.data.splits import chronological_split
from market_maker.features.lob_features import compute_lob_features
from market_maker.env.market_making_env import MarketMakingEnv
from market_maker.agents.heuristic import HeuristicMarketMaker
from market_maker.evaluation.backtest import run_backtest, write_run_artifacts
from market_maker.evaluation.metrics import mean_absolute_position, summarize_episode
from market_maker.db.session import get_engine, get_session, init_db
from market_maker.db.writer import write_run_to_db
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default="heuristic_baseline")
    parser.add_argument("--variant", default="Heuristic")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_lob(cfg.data.lob_path)
    df, _ = validate_lob(df)
    splits = chronological_split(df, "timestamp", **cfg.data.split.model_dump())

    test_feats = compute_lob_features(splits["test"], depth=cfg.data.depth, return_lags=[1, 5])
    feature_cols = [c for c in test_feats.columns if c != "timestamp"]
    fused = test_feats[feature_cols].to_numpy(dtype=np.float32)
    episode_steps = int((cfg.simulator.episode_minutes * 60 * 1000) / cfg.simulator.action_interval_ms)

    env = MarketMakingEnv(
        fused_features=fused, raw_lob_df=splits["test"].reset_index(drop=True), feature_columns=feature_cols,
        tick_size=cfg.data.tick_size, levels_per_side=cfg.simulator.levels_per_side,
        min_spread_ticks=cfg.simulator.min_spread_ticks, max_spread_ticks=cfg.simulator.max_spread_ticks,
        max_mid_offset_ticks=cfg.simulator.max_mid_offset_ticks,
        total_volume_per_side=cfg.simulator.total_volume_per_side, max_inventory=cfg.simulator.max_inventory,
        fee_per_unit=cfg.simulator.fee_per_unit, reward_config=cfg.reward.model_dump(),
        reference_price_mode=cfg.simulator.reference_price_mode, episode_length_steps=episode_steps,
    )
    agent = HeuristicMarketMaker()
    result = run_backtest(env, agent, n_episodes=10, is_heuristic=True, max_inventory=cfg.simulator.max_inventory)
    map_values = [mean_absolute_position(inv) for inv in result["inventories"]]
    summary = summarize_episode(result["episode_pnls"], result["inventories"])
    summary["map"] = float(np.mean(map_values)) if map_values else 0.0

    run_dir = f"artifacts/evaluation/{args.run_id}"
    write_run_artifacts(run_dir, config=cfg.model_dump(), metadata={"seed": cfg.project.seed, "model_path": "N/A"},
                         backtest_result=result, metrics=summary)

    engine = get_engine(f"sqlite:///{os.environ.get('MARKET_MAKER_DB_PATH', 'artifacts/market_maker.db')}")
    init_db(engine)
    session = get_session(engine)
    write_run_to_db(session, run_id=args.run_id, variant=args.variant, seed=cfg.project.seed, model_path="N/A",
                     config=cfg.model_dump(), backtest_result=result, metrics=summary)
    session.close()

    logger.info("Evaluation complete for run_id=%s. Summary: %s", args.run_id, summary)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write the `Makefile`**

```makefile
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
```

- [ ] **Step 6: Run the full test suite**

Run: `pytest -q`
Expected: PASS (all tests across all modules)

- [ ] **Step 7: Run the fast end-to-end pipeline for real**

```bash
make install
make sample-data
python scripts/run_all.py --config configs/fast_demo.yaml
```

Expected: exits 0, produces `artifacts/reports/FINAL_PROJECT_REPORT.md` and at least one run under `artifacts/evaluation/`.

- [ ] **Step 8: Update `BUILD_STATUS.md` and commit**

```bash
git add -A
git commit -m "feat: run_all orchestration script, Makefile targets, end-to-end integration test"
```

---

### Task 18: README, final report content, remaining ablation variants, hardening

**Files:**
- Modify: `README.md`
- Modify: `scripts/train_agent.py` (add `SAC_LOB_TRANSFORMER`, `SAC_LOB_SENTIMENT`, `SAC_FULL` variant wiring using the fusion module from Task 8)
- Modify: `scripts/evaluate.py` (accept `--agent-path` to load a trained SAC model instead of only the heuristic)
- Test: `tests/test_ablation_variants.py`

**Interfaces:**
- Consumes: `market_maker.features.fusion.build_fused_dataset` (Task 8), `market_maker.evaluation.ablation.select_enabled_variants` (Task 14).
- Produces: `scripts/train_agent.py --variant SAC_LOB_TRANSFORMER|SAC_LOB_SENTIMENT|SAC_FULL` builds the fused observation with only the corresponding feature groups (LOB+Transformer, LOB+Sentiment, LOB+Transformer+Sentiment) before constructing the env.

- [ ] **Step 1: Write failing test for variant-specific feature selection**

```python
# tests/test_ablation_variants.py
import numpy as np
import pandas as pd
from market_maker.features.fusion import build_fused_dataset


def test_lob_only_variant_excludes_sentiment_and_transformer_columns():
    n = 4
    lob_feats = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="s", tz="UTC"),
                               "midprice": [1.0] * n})
    empty_sentiment = pd.DataFrame(index=range(n))  # LOB_ONLY variant passes an empty sentiment frame
    empty_embeddings = np.zeros((n, 0), dtype=np.float32)
    fused, columns = build_fused_dataset(lob_feats, empty_sentiment, empty_embeddings, {},
                                          private_state_placeholder_dim=2)
    assert not any("sentiment" in c for c in columns)
    assert not any("transformer" in c for c in columns)
    assert fused.shape == (n, 1 + 0 + 0 + 2)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_ablation_variants.py -v`
Expected: PASS immediately — `build_fused_dataset` from Task 8 already supports empty sentiment/transformer inputs by construction; this test documents and locks in that contract for the ablation variants.

- [ ] **Step 3: Wire variant selection into `scripts/train_agent.py`**

Modify `build_env_for_split` in `scripts/train_agent.py` to branch on `args.variant`:

```python
def build_fused_features_for_variant(cfg, feats, feature_cols, variant: str):
    from market_maker.features.fusion import build_fused_dataset
    import pandas as pd
    n = len(feats)
    lob_only = feats[["timestamp"] + feature_cols]
    empty_sentiment = pd.DataFrame(index=range(n))
    empty_embeddings = np.zeros((n, 0), dtype=np.float32)

    if variant == "SAC_LOB_ONLY":
        fused, _ = build_fused_dataset(lob_only, empty_sentiment, empty_embeddings, {}, private_state_placeholder_dim=0)
    else:
        raise NotImplementedError(
            f"Variant {variant} requires precomputed sentiment/transformer artifacts; "
            f"run `make sentiment` and `make transformer` first, then extend this branch to load them."
        )
    return fused
```

- [ ] **Step 4: Write `README.md`**

```markdown
# LLM-Enhanced Financial Market Making

Research/backtesting MVP extending the multi-price-level IMM market-making framework with
FinBERT sentiment, a Transformer temporal encoder, and a Soft Actor-Critic (SAC) agent.

**Correction to the original slides:** IMM is already a multi-price-level method in the base
paper (two bid levels, two ask levels) — this project does not claim to introduce
multi-price-level quoting as a novel contribution; it inherits it from IMM.

## Architecture

Raw LOB/news -> validation + chronological split -> LOB features + FinBERT sentiment (cached)
+ Transformer embeddings (frozen) -> feature fusion -> Gymnasium `MarketMakingEnv` -> SAC
(Stable-Baselines3) -> backtest/evaluation written to SQLite + file artifacts -> FastAPI
(reads SQLite) -> React frontend (calls FastAPI) + auto-generated Markdown report.

## Data mode

**Mode B**: real crypto LOB (Binance trades/aggTrades + self-collected or Kaggle L2 depth
snapshots) is the primary trading dataset. FNSPID (real timestamped equity news) validates the
sentiment/alignment pipeline on its own ticker/period. These are NOT causally matched — no claim
is made that news improved trading on the crypto LOB. See `configs/default.yaml` for paths.

## Data schema

See `src/market_maker/schemas.py` (`LobRow`, `NewsRow`) for the exact required columns.

## Setup

    make install
    make install-frontend
    make sample-data   # synthetic smoke-test data — labeled, not a performance result

## Fast demo (synthetic, minutes)

    python scripts/run_all.py --config configs/fast_demo.yaml

## Full pipeline (real Mode-B data)

Place real LOB/news Parquet files at the paths in `configs/experiment_full.yaml`, then:

    python scripts/run_all.py --config configs/experiment_full.yaml

## Artifacts

`artifacts/evaluation/<run_id>/` — config, metadata, metrics, episodes, fills, plots, report (file artifacts).
`artifacts/market_maker.db` — SQLite DB queried live by the API/frontend.
`artifacts/reports/FINAL_PROJECT_REPORT.md` — auto-generated final report.

## Frontend / API

    make api         # FastAPI, http://localhost:8000/docs
    make frontend     # React (Vite) dev server, http://localhost:5173 — run in a second terminal

## Testing

    make test

## Known limitations

- Fills/queue priority are approximated from snapshot data, not a message-level exchange
  reconstruction.
- Mode B means LOB and news come from different assets/periods; no causal sentiment-to-PnL claim
  is made.
- The Transformer encoder is frozen during SAC training (a simplification vs. IMM's joint
  representation+policy learning).

## Disclaimer

Research/backtesting only. No live trading, no broker credentials, no real-money execution.
```

- [ ] **Step 5: Run full test suite one more time**

Run: `pytest -q`
Expected: PASS

- [ ] **Step 6: Update `BUILD_STATUS.md` to reflect completion and commit**

```bash
git add -A
git commit -m "docs: README, ablation variant wiring for LOB_ONLY, hardening pass"
```

---

## Summary of Remaining Optional Work (not in this plan)

Per spec §19, §45: TD3 baseline, full SAC+behavior-cloning imitation learning, IMM stable
reference price with real queue data, MLflow tracking, Docker Compose, multi-asset support. These
are explicitly out of scope for this plan and should only be attempted after Task 18 is complete
and all tests pass — each would be its own follow-up plan.
