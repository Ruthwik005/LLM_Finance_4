# RUNBOOK: finish this project from this repo

This guide is for someone who has never trained a reinforcement-learning agent.
Every command can be copied and pasted. Run commands from the project folder unless it says otherwise.

Contents: 0 Status, 1 Prerequisites, 2 Install and test, 3 Ten-minute demo, 4 Real data, 5 Configure, 6 Train, 7 Read results,
8 Final report, 9 GitHub, 10 Presenting and checklist, 11 Troubleshooting, 12 Glossary.

---

## 0. What you have and what is left

**Done (software):** data loading and checks, features, FinBERT sentiment, Transformer, a market-making simulator, the SAC agent,
two baselines, four ablations, metrics, a database, a read-only API, a dashboard, a report generator, and one-command pipelines.
The test suite checks all of it with small synthetic data.

**Not done (needs you):**

1. Collect real order-book data (hours to days of waiting).
2. Get real news (one command).
3. Choose settings for your instrument (tick size, order sizes, training length).
4. Run the long training (a few hours, mostly unattended).
5. Read the results honestly and write your conclusions.
6. Publish to GitHub and present.

**Important:** the numbers you see in the demo come from *synthetic* data. They only prove that the software runs.
They say nothing about real markets. Never quote them as results.

---

## 1. Install prerequisites (macOS)

You need Python 3.11, Node 20 or newer, and git. Check what you have:

```bash
python3.11 --version     # want: Python 3.11.x
node --version           # want: v20.19 or newer (22 LTS is fine)
npm --version
git --version
```

If something is missing, install [Homebrew](https://brew.sh) first, then:

```bash
brew install python@3.11 node git
```

Close and reopen Terminal, then run the four version commands again.

---

## 2. Open the project, install, run the tests

```bash
cd ~/Desktop/superdev-41          # or wherever you put the folder
make install                      # creates the .venv folder and installs Python packages (a few minutes)
source .venv/bin/activate         # run this in EVERY new Terminal window; your prompt starts with (.venv)
make install-frontend             # installs the dashboard's packages (npm)
pytest -q                         # runs the Python tests
```

Expected end of the `pytest -q` output (the count grows as tests are added; there must be no `failed`):

```text
........................................
NNN passed in NN.NNs
```

Optional frontend tests: `cd frontend && npx vitest run && cd ..`

---

## 3. Ten-minute demo (synthetic data)

```bash
source .venv/bin/activate
make demo-data
```

This creates fake order-book data, fake news scores ("stub sentiment"), trains tiny agents for a few hundred steps and writes the results to
`artifacts/` and the database. It takes about a minute. You should see lines like:

```text
... Stage prepare_data finished in 0.3s
... Stage build_features finished in 1.5s
... Stage train_transformer finished in 7.1s
... Stage run_experiments finished in 18.2s
... Stage generate_report finished in 0.6s
```

Then start the dashboard:

```bash
make demo
```

You should see:

```text
============================================================
API:       http://localhost:8000/docs   (health: http://localhost:8000/health)
Dashboard: http://localhost:5173
Press Ctrl-C to stop everything.
============================================================
```

Open the Dashboard URL in a browser. Press Ctrl-C in Terminal to stop.
If you get `port 8000 is already in use` (Docker often uses it), run `API_PORT=8765 make demo`.

**What each screen shows**

| Screen | What it shows | How to read it |
|---|---|---|
| Overview | key numbers of the selected run | pick a run in the top bar. Look at the data-mode label first: "synthetic" means a smoke test |
| Episode Replay | one test episode step by step: mid price, your bid/ask quotes (two levels each side), fills as triangles, inventory and PnL | press play. Compare with the heuristic run. Look for long periods where inventory sits at one side |
| PnL / Equity | cumulative and per-episode PnL | many small wins and one big loss is a bad sign |
| Inventory & Risk | position over time, MAP, max inventory, drawdown | the agent should not sit at the limit all the time |
| Market-Making Behavior | cancel ratio, fill rate, average spread, adverse selection, where PnL came from | high cancel ratio = lots of quote churn |
| Training | learning curve on the validation data and the "best checkpoint" | the chosen model is the one with the best validation reward, not the last one |
| Sentiment & Signals | sentiment, article counts and Transformer probabilities against the price | stub sentiment is random noise by design |
| Model Comparison | all variants, mean ± std over seeds, exploratory tests | if the ± ranges overlap heavily, there is no detectable difference |

Make the report from the demo data (it will carry the "SOFTWARE SMOKE TEST ONLY" banner):

```bash
make report
open artifacts/reports/FINAL_PROJECT_REPORT.md
```

---

## 4. Get REAL data

The real setup is "Mode B": a real crypto order book (BTCUSDT) plus real *equity* news (FNSPID). They are from different assets and
periods. Therefore sentiment cannot be shown to cause trading profit here. Say so in your presentation.

### 4a. Live Binance order-book collector (recommended)

Binance has no free historical order-book archive, so you record it yourself. It needs the internet and no account or key.

**How long?** The split is by calendar day (UTC): train 70%, validation 15%, test 15%, in time order.
With fewer than **4 UTC days** the test split is empty and the pipeline stops with an error.
Recommended: **at least 7 days**, continuous. The default test setting needs about 10 hours of test data
(30 windows of 20 minutes), which 7 days comfortably give (the last 2 days are test).

**Disk space:** 500 ms snapshots = 172,800 rows per day. A 100,000-row synthetic file of the same shape is 21.5 MB, i.e. about 215 bytes
per row, so about **40 MB per day** is a rough estimate (not measured on real data; check with `du -sh data/raw/lob`).
Feature files add roughly 1.5 to 3 GB for a week of data (estimate). Keep 5 GB free.

**Keep the laptop awake.** Sleep creates gaps in the data and a gap inside an episode looks like a price jump to the agent.
Use `caffeinate`, keep the charger plugged in, and start the collector in its own Terminal window.

First a 1-minute test:

```bash
source .venv/bin/activate
python scripts/collect_binance_lob.py --symbol BTCUSDT --depth 5 --interval-ms 500 --duration-minutes 1 --out-dir data/raw/lob/test_run
```

Expected output (about 2 rows per second; an 18-second test produced 34 rows):

```text
collecting BTCUSDT depth=5 every 500ms mode=ws
flushed 120 rows -> data/raw/lob/test_run/lob_part_BTCUSDT_<timestamp>_0000.parquet
```

(Your row count will be about 120 after one minute.)

Then the real collection. `--duration-minutes 10080` is 7 days. You can stop early with Ctrl-C (the last rows are saved):

```bash
caffeinate -dimsu python scripts/collect_binance_lob.py --symbol BTCUSDT --depth 5 --interval-ms 500 \
    --duration-minutes 10080 --out-dir data/raw/lob/btcusdt_run1
```

It writes a new file every 60 seconds, so a crash loses at most one minute. If it stops, start it again with the **same** `--out-dir`.
You then have a gap; try to avoid several gaps.

**Merge the parts into one clean file.** Always pass `--out data/processed/lob_binance.parquet`.
(The default output `data/processed/lob.parquet` is the *synthetic demo* file; do not overwrite it.)
Use a separate `--out-dir` per collection so old test files do not get merged by accident.

```bash
python scripts/merge_lob_parts.py --parts-dir data/raw/lob/btcusdt_run1 --out data/processed/lob_binance.parquet
```

Expected: `merged -> data/processed/lob_binance.parquet: {... 'rows': N, 'parts': M}`.
The merge removes crossed books, negative sizes and exact duplicates and sorts by time.

If the collector cannot connect, Binance may not be reachable from your network or region. Check Binance's terms for your location.
Try `--mode rest` (polling, slower but simpler).

### 4b. Public Binance data (optional, depth 1 only)

Binance publishes free daily files, but only the best bid/ask (depth 1). Use this only if you cannot collect live data.
Set `data.depth: 1` in your config.

```bash
python scripts/download_binance_public.py --symbol BTCUSDT --start 2023-06-01 --end 2023-06-07 \
    --kinds bookTicker,aggTrades --trades-market futures/um --max-rows 1000000
```

This writes `data/raw/lob/binance_public/binance_BTCUSDT_bookTicker_aggTrades_<start>_<end>.parquet`. Point `data.lob_path` at it.
`--max-rows` limits the raw rows read per day (1,000,000 rows is only the first hours of a busy day); remove it for full days (large downloads).
Quotes with only one level mean the Transformer and the agent see less information than with a depth-5 book.

### 4c. News: FNSPID (equity news, Hugging Face)

```bash
python scripts/prepare_fnspid.py --tickers AAPL --max-rows 5000
```

It streams the big FNSPID file over HTTP and stops after 5000 matching articles. Output: `data/raw/news/fnspid_AAPL.parquet`.
Useful options: `--start 2020-01-01 --end 2020-12-31`, `--tickers AAPL,MSFT`, `--max-scan-mb 3000` (stop scanning after this many MB).
A test with `AAPL` returned 441 articles from March to June 2020 in the first 5 MB scanned.
The data licence is CC BY-NC 4.0 (non-commercial; credit the authors).

**Mode B in one sentence for your report:** the news are real but for AAPL in 2020; the order book is real but for BTCUSDT now.
Sentiment is therefore validated as a pipeline, and the ablations show whether the *extra features change the numbers*, not whether news causes profit.

No news file? The pipeline still runs: `run_all.py` skips the two sentiment variants and records them as N/A with the reason.

---

## 5. Configure

```bash
cp configs/real_binance.yaml configs/my_run.yaml
```

Edit `configs/my_run.yaml` in any text editor. Use that file in all following commands.

### Every setting that matters

| Setting | Meaning | Advice |
|---|---|---|
| `project.seed` | base random seed | keep 42. Training seeds come from `experiment.seeds` |
| `project.device` | where the Transformer runs | `auto` picks CUDA, Apple MPS or CPU |
| `data.lob_path` | merged order-book file | `data/processed/lob_binance.parquet` |
| `data.news_path` | news file | `data/raw/news/fnspid_AAPL.parquet`; if the file does not exist, news is skipped |
| `data.symbol` | label for the instrument | `BTCUSDT` |
| `data.depth` | order-book levels in the file | 5 for the live collector, 1 for public data |
| `data.tick_size` | smallest price step | **look it up, do not guess** (below) |
| `data.split` | train / validation / test fractions by day | 0.70 / 0.15 / 0.15 |
| `sentiment.*` | FinBERT model, batch size, lookback windows (5, 30, 120 min), max age (240 min) | keep defaults. Lower `batch_size` if memory is tight |
| `transformer.sequence_length` | how many past snapshots the encoder sees | 32 to 64 |
| `transformer.d_model`, `nhead`, `num_layers`, `dim_feedforward` | encoder size | keep small (32/2/1/64 or 64/4/2/128) |
| `transformer.horizons_minutes` | prediction horizons | [1, 5] |
| `transformer.epochs`, `patience` | training length, early stopping | epochs 20 to 30 with patience 3 to 5 for real data (the shipped value 2 is a smoke-test value) |
| `transformer.window_stride` | use every Nth window | 5 to 20 for millions of rows |
| `transformer.class_weights` | balance the up/flat/down classes | true |
| `features.label_threshold_bps` | what counts as "up" or "down" (basis points of price) | pick near a typical 1-minute move so "flat" is not 90% of labels |
| `features.vol_window`, `return_lags`, `clip_value` | feature windows and clipping | defaults are fine. Watch the "clipped" warning |
| `simulator.episode_minutes` | length of one episode | 20 (= 2400 steps at 500 ms) |
| `simulator.action_interval_ms` | time between decisions | 500, matching the data |
| `simulator.total_volume_per_side` | order size per side, split over two levels | **scale to the instrument** (below) |
| `simulator.max_inventory` | hard position limit | **scale to the instrument** |
| `simulator.min_spread_ticks`, `max_spread_ticks`, `max_mid_offset_ticks` | allowed quote range | 1 to 8 ticks, offset up to 4 |
| `simulator.latency_steps` | delay before a new order can fill | 0 (optimistic) or 1 (one snapshot, more cautious) |
| `simulator.fee_per_unit`, `rebate_per_unit` | cost / rebate per filled unit, in quote currency | see fees below |
| `reward.type` | `truncated_inventory` or `quadratic` | keep truncated |
| `reward.inventory_threshold`, `inventory_penalty` | penalty starts above this inventory | threshold about 40% of `max_inventory` |
| `reward.cancellation_penalty` | cost per canceled unit | small; raises it if the agent churns quotes |
| `sac.total_timesteps` | training steps per run | table below |
| `sac.learning_starts`, `buffer_size`, `batch_size` | warm-up steps, replay memory, mini-batch | buffer at least as large as the timesteps if RAM allows |
| `sac.eval_freq`, `n_eval_episodes` | validation check every N steps, on how many windows | 10,000 and 5 to 10 |
| `sac.device` | where SAC trains | `cpu` is usually fastest for these small networks |
| `evaluation.adverse_horizon_steps`, `adverse_threshold_ticks` | adverse-selection definition | 10 steps and 1 tick |
| `baselines.gamma`, `kappa` | Avellaneda-Stoikov parameters | untuned defaults; do not claim they are optimal |
| `experiment.variants` | which models to train | all four |
| `experiment.seeds` | one training run per seed | `[0, 1, 2, 3, 4]` (5 seeds). 3 is the minimum for a spread |
| `experiment.n_test_episodes` | number of fixed test windows | 30 (needs about 10 hours of test data at 20 minutes each) |
| `experiments.*` | old on/off flags | not used by the new runner; ignore |

**Tick size.** Look it up from Binance (no account needed):

```bash
curl -s "https://api.binance.com/api/v3/exchangeInfo?symbol=BTCUSDT" | python3 -c "import sys,json; print([f for f in json.load(sys.stdin)['symbols'][0]['filters'] if f['filterType']=='PRICE_FILTER'])"
```

Expected: `[{'filterType': 'PRICE_FILTER', 'minPrice': '0.01000000', 'maxPrice': '1000000.00000000', 'tickSize': '0.01000000'}]`
so `data.tick_size: 0.01` for BTCUSDT. For another pair, change `symbol=` and read its `tickSize`.

**Order sizes.** The shipped `total_volume_per_side: 20` and `max_inventory: 100` were chosen for the synthetic data. BTC order books
quote sizes of a fraction of a coin up to a few coins per level. Check typical sizes in your data, e.g.:

```bash
python -c "import pandas as pd; d=pd.read_parquet('data/processed/lob_binance.parquet'); print(d[['bid_size_1','ask_size_1']].describe())"
```

In an 18-second live test the level-1 sizes averaged about 2.5 (bid) and 7.5 (ask) BTC, with a minimum near 0.16, so the shipped 20 per side
is larger than the visible size. Choose a per-side volume that is small compared with the visible size (so the "you do not move the market" assumption is less wrong)
and a `max_inventory` of a few times that volume. Then scale `reward.inventory_threshold` with it.

**Fees.** The simulator charges `fee_per_unit` for every unit filled. A percentage fee becomes `fee_rate x price`. Illustration only:
a maker fee of 0.01% at a price near 60,000 is `0.0001 x 60000 = 6.0` per unit. Look up the real fee tier of the venue you are modeling;
this repo does not know it. `rebate_per_unit` works the same way. Zero fees make market making look better than it is.

**Training steps to hours.** Measured on an Apple-silicon laptop, CPU, network 256x256, batch 256: about **300 steps per second**
for every variant. Treat these as rough (your machine will differ):

| `sac.total_timesteps` | one run | 20 runs (4 variants x 5 seeds) |
|---:|---:|---:|
| 100,000 | 6 min | 1.9 h |
| 200,000 | 11 min | 3.7 h |
| 300,000 | 17 min | 5.6 h |
| 500,000 | 28 min | 9.3 h |
| 1,000,000 | 56 min | 18.5 h |

Add the validation checks (about 5 million environment steps for the 20-run, 200k setup, roughly 15 more minutes at an estimated 6,000 steps/s)
and the Transformer. A good first plan: 200k to 300k steps, 3 to 5 seeds.

**Validation windows.** `sac.n_eval_episodes` windows are taken from the validation split and `experiment.n_test_episodes` from the test split,
evenly spaced. A warning `windows of N steps do not fit without overlap` means your split is too short for that many windows: collect
more days or lower the number.

---

## 6. Build features, train, run the experiment

Replace `my_run.yaml` with your file name. First check the data and plan:

```bash
source .venv/bin/activate
python scripts/prepare_data.py --config configs/my_run.yaml
python scripts/run_all.py --config configs/my_run.yaml --dry-run
python scripts/run_experiments.py --config configs/my_run.yaml --dry-run | tail -3
```

Expected (numbers depend on your config):

```text
20 SAC runs, 4,000,000 total training timesteps (+ ~5,040,000 validation and ~1,584,000 test env steps).
```

**Option 1: everything in one command** (real FinBERT, resumable):

```bash
caffeinate -dimsu python scripts/run_all.py --config configs/my_run.yaml --resume
```

**Option 2: step by step** (recommended the first time, so you see each result):

```bash
python scripts/build_features.py --config configs/my_run.yaml --stage base          # FinBERT runs here (first time downloads ~440 MB)
python scripts/train_transformer.py --config configs/my_run.yaml                     # watch val_loss; see the accuracy vs majority-class line
python scripts/build_features.py --config configs/my_run.yaml --stage fusion
python scripts/run_experiments.py --config configs/my_run.yaml --experiment-id real_v1 --resume
```

The Transformer log prints validation accuracy next to the majority-class accuracy. If they are equal, the Transformer learned nothing
beyond "predict the most common class". That is a real finding; do not hide it.

**Monitor.** Training prints a line per finished run. In another Terminal:

```bash
ls artifacts/evaluation | head -30            # one folder per finished run
tail -f artifacts/logs/pipeline_status.json   # only after run_all.py; shows finished stages
```

Start the dashboard (`make demo`) while training runs: finished runs appear after a refresh.

**Resume.** If the laptop sleeps or you press Ctrl-C, re-run the same command with `--resume`. A run is skipped only if it is
complete and matches the current experiment id, `--fast` mode, step budget, config and features. Pitfalls:

- Changing **any** config value that matters (reward, simulator, sac, features, windows) makes old runs "not matching", so they are retrained. That is intended.
- The same run folders are used by every experiment ("latest wins"). Running `--fast` or another experiment id afterwards overwrites runs of the same name.
- Do not edit the data file between runs. A changed file changes the feature hash and everything is retrained.
- Do not change `--experiment-id` while resuming; use the same id.
- Without `--resume`, everything is retrained and overwritten.

**GPU / MPS.** SAC with these small networks is usually fastest on `sac.device: cpu`. The Transformer and FinBERT can use a GPU:
add `--device mps` (Apple) or `--device cuda` to `train_transformer.py` / `build_features.py`. `auto` chooses for you.

---

## 7. Read the results

1. Open the dashboard (`make demo`) and look at Model Comparison, Episode Replay and Training.
2. Generate the report: `python scripts/generate_report.py --experiment-id real_v1` and open `artifacts/reports/FINAL_PROJECT_REPORT.md`.

**Metrics in plain words**

| Metric | Meaning |
|---|---|
| EPnL | average profit per test episode (in quote currency, e.g. USDT) |
| MAP | mean absolute position: how much inventory you carry on average |
| PnL/MAP | profit per unit of average risk |
| Fill count | number of executions in all test windows |
| Adverse selection | share of fills followed by a price move against you (within the configured horizon) |
| Cancel ratio | canceled volume / placed volume |
| Max drawdown | worst fall of the profit curve inside an episode |
| Sharpe-like | mean episode PnL / its standard deviation (only a rough guide with few episodes) |
| Fill rate | filled volume / placed volume |
| Spread capture vs inventory PnL | profit from buying below and selling above the mid, vs profit from the price drifting while holding inventory |

**What would worry me**

- **Zero or very few fills.** The numbers are then not about trading. Check order size, spread limits and the fill rule.
- **Inventory pinned at the limit.** The agent just holds a position and is betting on the price direction.
- **Negative PnL with a high fill rate.** Classic adverse selection: you trade, then the price moves against you.
- **Cancel ratio near 1.** Every quote is re-priced every step. Real exchanges cost or limit that. Compare with the heuristic.
- **Huge std across seeds, or only 2 or 3 seeds.** A difference smaller than the spread is not a difference.
- **Spread-capture PnL negative while inventory PnL positive.** Profit came from price drift, not from market making.
- **Validation reward much better than test result.** Overfitting to the validation windows or to the training period.
- **Transformer accuracy equal to the majority-class rate.** The encoder added noise, not signal.
- **Sentiment variants equal to LOB-only.** Expected in Mode B; do not claim news helps.
- **The learned agent loses to the heuristic.** Report it as it is.
- **Perfectly smooth, always-rising equity curve.** Suspect a bug or leakage and investigate.
- **A `clipped` warning for many columns.** The test period looks different from the training period; features saturate.

Allowed language: "the experiment compares...", "we observed...". Avoid "outperforms" or "significantly better" unless a proper test
supports it. The Welch p-values here are exploratory (few seeds, same test windows, no correction for many comparisons).

---

## 8. Final report

```bash
python scripts/generate_report.py --experiment-id real_v1 --config configs/my_run.yaml
```

Files:

- `artifacts/reports/FINAL_PROJECT_REPORT.md`: 24 sections.
- `artifacts/reports/figures/*.png`: PnL, inventory, adverse selection, learning curves.

It reads only the database and the run folders, and prints `N/A (reason)` where something is missing. You can convert it to PDF/Word with any
Markdown tool. Read it fully and edit the discussion in your own words. Check that the data-mode line says what you actually did.

---

## 9. Publish to GitHub (step by step)

1. In a browser go to github.com, sign in, click **New repository**. Name it (for example `llm-market-making`), choose Public or Private,
   **do not** add a README, .gitignore or licence (the repo must be empty). Click **Create repository**.
2. In Terminal, in the project folder:

```bash
git status                       # check what will be committed
git add -A
git status                       # data/, artifacts/ and .env must NOT be listed (they are git-ignored)
git commit -m "Complete project"
git remote add origin https://github.com/<your-user>/<your-repo>.git
git branch -M main
git push -u origin main
```

3. Authentication. GitHub no longer accepts your account password on the command line. Use one of:
   - **Personal Access Token:** GitHub > Settings > Developer settings > Personal access tokens > Fine-grained tokens > Generate.
     Pick your repo, permission "Contents: Read and write". When `git push` asks for a password, paste the token (the username is your GitHub name).
   - **GitHub CLI:** `brew install gh`, then `gh auth login` and follow the prompts, then `git push -u origin main`.

**Never commit:** `data/` (large, licensed), `artifacts/` (large, regenerated), `.env`, any token, `frontend/.env.local`, `.venv`.
They are already in `.gitignore`. If you accidentally pushed a secret, revoke it on GitHub at once; deleting the commit is not enough.
If you want to show a report in the repo, copy `FINAL_PROJECT_REPORT.md` and `figures/` into `docs/` and commit those.

---

## 10. Presenting and final checklist

### Demo script (about 5 minutes; works without retraining)

1. State the claim carefully: "A research simulator for a multi-level market-making agent with FinBERT and Transformer features. Historical replay, not live trading."
2. `make demo` (or open the saved dashboard). Show **Overview** and the data-mode label.
3. **Episode Replay:** pick one test episode of `sac_full_seed0` (or your best run); play it; point at quotes, fills, inventory and PnL.
4. Switch "compare with" to `heuristic` on the **same** episode.
5. **Behavior:** cancel ratio, fill rate, adverse selection, where the PnL came from.
6. **Model Comparison:** mean ± std over seeds, N/A rows, the "exploratory" note.
7. **Training:** the validation curve and the chosen checkpoint.
8. Say the limits out loud: snapshot fills are approximations, Mode B news are not matched, few seeds.

Do not say "profitable", "outperforms" or "better" unless your numbers and the report's own discussion support it.

### Done checklist

- [ ] `pytest -q` all green
- [ ] Real order-book data collected and merged (at least 7 days), news file prepared
- [ ] `configs/my_run.yaml`: tick size verified, sizes scaled, 3 to 5 seeds
- [ ] `run_experiments.py --dry-run` looked sane
- [ ] Real run finished; every run folder exists
- [ ] Report generated and read; data-mode line is correct; no smoke-test banner on real results
- [ ] You can explain the corrections: IMM is already multi-level; FinBERT is a sentiment classifier; SAC is not assumed better than TD3
- [ ] You can explain Mode B and the fill approximation
- [ ] Repo pushed; `data/`, `artifacts/`, `.env` not in it
- [ ] Demo rehearsed with the saved results

---

## 11. Troubleshooting

| Message or symptom | Cause | Fix |
|---|---|---|
| `ERROR: port 8000 is already in use` | another program (often Docker) uses it | `API_PORT=8765 make demo`. By hand: `uvicorn market_maker.api.main:app --port 8765` and `VITE_API_URL=http://localhost:8765` in `frontend/.env.local`. See who uses it: `lsof -i :8000` |
| `ERROR: frontend dependencies are not installed` | no `node_modules` | `make install-frontend` |
| `ModuleNotFoundError: No module named 'market_maker'` | virtual environment not active | `source .venv/bin/activate` (and `pip install -e .` if still failing) |
| `ValueError: Chronological split produced empty partition(s): ['test']` | fewer than 4 UTC days of data | collect more days (7 recommended) or add data |
| `ValueError: split has N rows, fewer than one episode` | a split shorter than `episode_minutes` | more data or a smaller `simulator.episode_minutes` |
| `windows of N steps do not fit without overlap` (warning) | test/validation split too short for that many windows | more data or fewer `n_test_episodes` / `n_eval_episodes` |
| `RuntimeError: Stale base features for SAC_...` | data, feature settings or the news file changed since the features were built | rebuild: `build_features.py --stage base`, retrain the Transformer, then `--stage fusion`; or `run_experiments.py ... --force-features` (then retrain the Transformer) |
| `Transformer checkpoint was trained on base features ..., current base features are ...` | stale Transformer after rebuilding base features | `python scripts/train_transformer.py --config ...` then `build_features.py --stage fusion` |
| `FileNotFoundError: LOB data not found at ...` | wrong `data.lob_path` or you never merged | check the path; run the merge command (section 4a) or `make sample-data` for synthetic data |
| `No cached sentiment ... and no news file ...` | sentiment variants but no news | provide the news file, or use `run_all.py` (it skips sentiment variants), or `--no-sentiment`, or `--stub-sentiment` for a smoke test |
| FinBERT download is slow or fails | first use downloads about 440 MB from Hugging Face | check the internet, retry; the model is cached afterwards |
| `Not run / insufficient data` in the report | that section had no results (empty DB, run did not finish) | run the experiment, then regenerate the report. Check `--db-path` / `--artifacts-root` |
| Report says `N/A (not run)` for sentiment variants | no news file at the time of the run | expected in news-optional mode; the reason is listed under the table |
| Report shows the smoke-test banner on real data | experiment was run with `--stub-sentiment` or marked synthetic | rerun without the stub flag; pass `--data-mode mode_b` |
| Report shows no banner on synthetic data | custom config name (data mode is inferred from the name `fast_demo.yaml`) | pass `--data-mode synthetic_smoke_test` to `run_experiments.py` / `run_all.py` |
| `resume: re-running X (metadata ... differs ...)` | config, budget, features or experiment id changed | expected; the run is retrained |
| `no runnable variants left` | all variants need news and there is none | provide news or include `SAC_LOB_ONLY` / `SAC_LOB_TRANSFORMER` |
| Collector cannot connect / HTTP error | network or Binance availability in your region | check the network, try `--mode rest`; respect Binance's terms |
| Merged data contains tiny test files | old `lob_part_*.parquet` files in the same folder | use a fresh `--out-dir` per collection |
| `clip` warning ("is clipped for X% of test rows") | test data far from training data | check the period; consider fewer raw price-level features or more training data |
| `Stage 'X' FAILED with exit code N` (run_all) | the stage's own error is printed above it | fix it, then `python scripts/run_all.py ... --from-stage X --resume` |

---

## 12. Glossary

- **LOB (limit order book):** the list of waiting buy orders (bids) and sell orders (asks) at each price.
- **Level 1 / level 2:** the best price on a side, and the next price behind it. The agent quotes both on each side.
- **Midprice (mid):** (best bid + best ask) / 2.
- **Spread:** best ask minus best bid. The market maker earns part of it when both sides trade.
- **Tick size:** the smallest allowed price step.
- **Inventory:** how much of the asset you hold (negative = short).
- **MAP:** mean absolute position, average |inventory|.
- **Adverse selection:** your order is filled mostly when the price is about to move against you.
- **Mark-to-market (MtM):** valuing the inventory at the current mid.
- **Episode:** one replayed time window (for example 20 minutes) that the agent plays from start to end.
- **Reward vs PnL:** the reward used for learning includes penalties; PnL is the money result. Report PnL and risk, not just reward.
- **SAC (Soft Actor-Critic):** a reinforcement-learning method for continuous actions that also rewards keeping its choices varied.
- **TD3:** a related method used by the original IMM paper. Not run here.
- **FinBERT:** a pretrained language model that labels financial text as positive, negative or neutral. It is a classifier, not a chatbot.
- **Transformer encoder:** a neural network that reads the last N market snapshots and summarizes them.
- **Frozen:** its weights do not change while the agent trains.
- **Ablation:** removing or adding one part (sentiment, Transformer) to see what changes.
- **Baseline:** a simple non-learning strategy to compare against.
- **Seed:** the number that fixes random choices. Different seeds give different trained agents; the spread between them shows how noisy training is.
- **Training / validation / test:** training teaches the agent; validation picks the best checkpoint and settings; the test period is looked at once, at the end.
- **Overfitting:** doing well on data the model has seen and badly on new data.
- **Look-ahead (leakage):** using information from the future. The news alignment forbids it: only articles with `published_at <= t` are used at time `t`.
- **Mode B:** real order book + real news that are not from the same asset and time.
- **Smoke test:** a run that only checks the software works.
