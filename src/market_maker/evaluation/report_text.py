"""Static methodology text for the final report.

Everything here is methodology that does not depend on measured results (spec sections 1.1, 33, Appendix A/B).
Measured numbers never live in this file; they are filled in by `report.py` from the DB / run artifacts.
"""
from __future__ import annotations

SMOKE_BANNER = "SOFTWARE SMOKE TEST ONLY - NOT A FINANCIAL PERFORMANCE RESULT"
FILL_DISCLAIMER = ("Fills and queue priority are approximated from historical snapshots/aggregated trades and are "
                   "not equivalent to a full message-level exchange reconstruction.")
MISSING = "Not run / insufficient data"

IMM_CITATION = ("IMM paper (IJCAI 2024): \"IMM: An Imitative Reinforcement Learning Approach with Predictive "
                "Representation Learning for Automatic Market Making\"")

BASE_PAPER = f"""The base paper of this project is the {IMM_CITATION}. The facts below are the ones this project relies on
(project specification, Appendix A):

- It addresses automatic market making with **multi-price-level** policies (orders are stacked on several price levels per side).
- It introduces a **stable reference-price** formulation.
- Its state contains market variables, predictive signals and private/account/order information.
- Its action describes the desired quote midprice, the spread, and the bid/ask volume-distribution parameters.
- It uses **predictive signals at multiple horizons**. LightGBM is used to generate such signals; it is not the whole state representation.
- Its representation-learning unit is a **temporal convolution + spatial attention (TCSA)** network.
- It uses a **signal-based expert strategy** and combines a **TD3-style** off-policy RL learner with an **imitation / behavior-cloning** loss.
- Its reward considers PnL, inventory control and an optional compensation / transaction incentive.
- Its experiments use 5-depth historical futures LOB data sampled at 500 ms, two quote levels on each side, and risk/profit metrics
  (Episodic PnL, Mean Absolute Position, Return Per Trade, PnL/MAP).

This project is **not** a reproduction of IMM: it does not use the same dataset, simulator semantics, hyperparameters or
training method, and it does not implement IMM's imitation-learning objective."""

CORRECTIONS = [
    ("IMM is a single-price-level method.",
     "IMM is already a multi-price-level market-making method (multi-price-level order stacking, two bid and two ask levels in "
     "its experiments).",
     "This project inherits two-level quoting from IMM and does not claim multi-price-level quoting as a contribution."),
    ("IMM uses LightGBM as the whole state representation.",
     "LightGBM generates multi-granularity predictive signals; the state representation also uses a temporal convolution + "
     "spatial attention network (TCSA).",
     "Our Transformer encoder plays the role of the temporal representation; it is a different architecture, not a re-implementation of TCSA."),
    ("IMM is a generic actor-critic system.",
     "The IMM base RL learner is TD3 combined with an imitation / behavior-cloning objective on expert demonstrations.",
     "This project uses SAC without imitation learning. The report therefore describes the work as extending IMM's multi-level "
     "structure, not its full imitation objective."),
    ("FinBERT is a generative LLM.",
     "FinBERT is a pretrained BERT-based financial sentiment classifier.",
     "It is used only as a frozen sentiment encoder/classifier, never as a generative trading agent (see Appendix B wording in Section 9)."),
    ("FinBERT sentiment predicts prices.",
     "Sentiment is a text-derived context feature. Whether it helps trading must be validated experimentally.",
     "The `SAC_LOB_SENTIMENT` and `SAC_FULL` ablations exist to test this; no benefit is assumed."),
    ("SAC is better than TD3.",
     "SAC is a reasonable off-policy maximum-entropy alternative for continuous actions; superiority must be established by experiments.",
     "TD3 was not run in this project (optional extension), so no SAC-vs-TD3 statement is made."),
    ("Headlines can be matched to the nearest market state.",
     "Historical news is asynchronous and sparse relative to 500 ms LOB data; future headlines must never be aligned to earlier states.",
     "Only articles with published_at <= t are visible at time t (as-of backward join), covered by a mandatory leakage test."),
    ("A snapshot LOB dataset is an exchange simulator.",
     "Without message-level order events and queue identifiers, queue position and fills are approximations.",
     f"All fills here are snapshot approximations. {FILL_DISCLAIMER}"),
]

ARCH_MERMAID = """```mermaid
flowchart TD
    A[Historical LOB / Trades] --> B[Data Validation + Normalization]
    N[Timestamped Financial News] --> C[News Validation + Asset Mapping]

    B --> D[LOB Feature Engineering]
    C --> E[FinBERT Batch Sentiment]
    D --> F[Temporal Window Builder]
    F --> G[Transformer Temporal Encoder]

    E --> H[As-of Time Alignment + Sentiment Aggregation]
    D --> I[Feature Fusion]
    G --> I
    H --> I

    I --> J[Gymnasium Market-Making Environment]
    J --> K[SAC Agent]
    K --> L[Quote Action: Mid Offset, Spread, Bid Split, Ask Split]
    L --> J

    J --> M[Rewards + Next State + Fill Events]
    M --> K

    J --> O[Evaluation Engine]
    O --> P[Metrics / Trades / Episodes / Config]
    P --> Q[FastAPI Results API]
    P --> R[React Dashboard]
    P --> S[Automated Markdown Report]
```"""

ARCH_NOTES = """The system is a modular monolith (one Python package, `market_maker`), not a set of networked services.
FinBERT is never run inside the RL loop: sentiment is scored once, cached to Parquet, aligned to market timestamps and
aggregated; Transformer embeddings are precomputed with a frozen checkpoint; the RL environment consumes numeric arrays only.

| Stage | Module / script | Output |
|---|---|---|
| Validate + split | `data/validators.py`, `data/splits.py`, `scripts/prepare_data.py` | chronological train/validation/test |
| Base features | `features/pipeline.py`, `scripts/build_features.py` | `artifacts/features/lob_<split>.parquet`, scaled `.npy` arrays, `scaler.joblib`, `feature_meta.json` |
| Transformer | `models/transformer_train.py`, `scripts/train_transformer.py` | `artifacts/transformer/checkpoint.pt`, `training_curve.csv` |
| Fusion | `features/pipeline.py` (`--stage fusion`) | `artifacts/features/<variant>_<split>.npy` |
| Environment | `env/market_making_env.py`, `env/execution.py`, `env/accounting.py`, `env/reward.py` | Gymnasium env |
| Training + experiment | `training/train_sac.py`, `training/experiment.py`, `scripts/run_experiments.py` | `artifacts/rl/<variant>/seed<N>/`, run folders, SQLite rows, experiment summary |
| Serving | `api/main.py` (FastAPI, read-only), `frontend/` (React) | dashboard |
| Report | `evaluation/report.py`, `scripts/generate_report.py` | this document |"""

FINBERT_METHOD = """- Model: `{model_name}` through Hugging Face Transformers, run in `model.eval()` / `torch.no_grad()`. It is **not fine-tuned** in this project.
- Output per text item: P(positive), P(negative), P(neutral); scalar score `sentiment_score = P(positive) - P(negative)`. Results are cached to Parquet so the RL code never re-runs FinBERT.
- **No look-ahead.** For a market state at time `t` only articles with `published_at <= t` are visible (as-of backward join; no nearest-neighbor matching). A mandatory unit test places a strongly positive article at `t+1` and verifies it cannot influence the state at `t`.
- Aggregation windows (config `sentiment.lookback_minutes`): {lookbacks} minutes. For each window: exponentially weighted mean score, maximum absolute score and article count; plus minutes since the latest article (capped at `sentiment.max_age_minutes` = {max_age}) and a `has_recent_news` validity mask. With no recent news the sentiment is 0, the count is 0 and the mask is false, so the policy can tell "neutral article" from "no article".
- Sentiment features feed only the `SAC_LOB_SENTIMENT` and `SAC_FULL` variants.
- FinBERT sentiment does not guarantee price relevance; whether it helps is tested by the ablations, not assumed.

**Appendix B wording.** In this project, "LLM-enhanced" refers to the use of a pretrained transformer-based financial language model (FinBERT) to extract sentiment features from financial text. FinBERT is used as a sentiment encoder/classifier rather than as a generative trading agent."""

TRANSFORMER_METHOD = """- Input: a sliding window of standardized market features, shape `[sequence_length, feature_dim]` (sequence length {seq}).
- Architecture: linear input projection to `d_model` = {d_model}, positional encoding, {layers} TransformerEncoder layer(s), {heads} attention heads, feed-forward width {ff}, dropout {dropout}, layer normalization, pooled final representation.
- Supervised pretraining objective: for each horizon (config `transformer.horizons_minutes` = {horizons} minutes) classify the future midprice move as down / flat / up with a threshold of {thr} basis points of the midprice; the loss is the sum of the per-horizon cross-entropies (optional class weights: {cw}). A label is ignored when the row found for `t + horizon` lies later than `label_max_gap_factor` times the horizon (session gaps).
- Early stopping on the validation loss; the test split is never touched during Transformer training. The scaler used for the inputs is fit on the train split only.
- Output to RL: the pooled embedding and the per-horizon class probabilities (plus a `transformer_signal_valid` flag for rows without a full window). The Transformer is **frozen** during SAC training, so representation learning and RL are not trained jointly (a simplification relative to IMM).
- Embeddings of the training split are in-sample for the Transformer; validation and test embeddings are out-of-sample. Transformer features may overfit."""

ENV_METHOD = """`MarketMakingEnv` (Gymnasium) replays one contiguous historical segment per episode (random start in the training split during training; fixed, evenly spaced windows for validation and test).

**Action (4 continuous values in [-1, 1]).**

| Index | Meaning | Mapping |
|---|---|---|
| 0 | quote-center offset | `offset_ticks = a0 * max_mid_offset_ticks` ({max_off} ticks), added to the reference price |
| 1 | quoted spread | `spread_ticks = min_spread_ticks + (a1 + 1)/2 * (max_spread_ticks - min_spread_ticks)` ({min_sp} to {max_sp} ticks) |
| 2 | bid volume split | `(a2 + 1)/2` of the per-side volume goes to bid level 1, the rest to bid level 2 |
| 3 | ask volume split | `(a3 + 1)/2` of the per-side volume goes to ask level 1, the rest to ask level 2 |

Two bid levels and two ask levels are produced; level 2 is one tick behind level 1. All prices are rounded to the tick size ({tick}); bid level 1 is always strictly below ask level 1. The per-side volume is {vol}. The reference price mode is `{ref}`.

**Execution (snapshot approximation).** A resting bid is fill-eligible on the next snapshot if the next best ask is at or below its price, or if an aggregate sell trade at or below its price is observed (trade volume caps the fill); asks are symmetric. Fills never exceed the outstanding volume and partial fills leave the remainder resting. {disclaimer}

**Quote persistence.** Orders are kept per (side, level). An unchanged price keeps its remaining volume (no cancel); a changed price cancels the remainder and places a new order; a smaller desired volume cancels only the difference; a larger one tops up. Placed and canceled volume are logged every step.

**Latency.** `latency_steps` = {latency}: new or changed orders become fill-eligible only after that many snapshots.

**Fees and rebates.** `fee_per_unit` = {fee}, `rebate_per_unit` = {rebate} (applied per filled unit in the accounting).

**Private state.** The observation is `[market features | private state]`; the private block (6 values, always appended by the environment) is: inventory / max inventory, tanh(mark-to-market PnL / `pnl_scale`), remaining-episode fraction, last quoted level-1 bid distance, ask distance and spread (in ticks normalized by `max_spread_ticks`).

**Inventory hard limit.** Desired volumes are clamped each step so that even if every outstanding order filled, |inventory| <= `max_inventory` ({max_inv}).

**Accounting.** Buy fill: inventory up, cash down; sell fill: inventory down, cash up; inventory is marked at the reference price (midprice)."""

LIMITATIONS = [
    "Historical replay cannot model the effect of the agent's own orders on the market: the market path is fixed regardless of what the agent quotes.",
    "Snapshot data cannot reconstruct exact queue priority. Fill timing and fill probability are approximations.",
    "News and LOB frequencies differ drastically (sparse, asynchronous articles versus 500 ms snapshots).",
    "Publication timestamps can contain delays or ambiguity, so the as-of alignment may still be optimistic or pessimistic by an unknown amount.",
    "FinBERT sentiment does not guarantee price relevance.",
    "Transformer features may overfit; the frozen encoder's training-split embeddings are in-sample.",
    "RL results can have high variance across random seeds; a small number of seeds gives low statistical power.",
    "SAC versus TD3 superiority is domain-dependent; TD3 was not run here, so no comparison is made.",
    "Backtest PnL may differ materially from live execution.",
    "Fees, rebates, slippage and latency strongly affect market-making economics; the values used here are configuration assumptions, not calibrated exchange terms.",
]
LIMITATIONS_EXTRA = [
    "Mode B data (real crypto order book + real equity news from FNSPID) is validated separately: the news and the order book come from different assets and periods and are not causally matched, so no causal sentiment-to-PnL claim is possible from such a run.",
    "IMM's imitation-learning objective, stable reference price and TCSA encoder are not implemented; this is an extension of IMM's multi-level structure, not a reproduction.",
    "The Transformer is frozen during SAC training (no joint representation learning).",
    "The Avellaneda-Stoikov baseline parameters (gamma, kappa) and the heuristic baseline are not tuned; baselines are evaluated once on the same fixed test windows, so they have no seed spread.",
    "All seeds share the same fixed test windows, so the Welch tests only compare seed-to-seed training variability on those windows, not variability across market regimes.",
    "Episode windows are contiguous row ranges and are not checked against gaps in the collected data (for example when the collector was interrupted); a gap inside an episode appears to the agent as a price jump. Transformer labels do ignore targets across gaps.",
    "The configured `adverse_selection_penalty` is not applied by the reward function (the optional term is not implemented); the adverse-selection metric is evaluated only after the fact.",
    "Metric definitions are approximations: the adverse-selection ratio uses a fixed horizon and tick threshold from the config and is not claimed to be equivalent to the IMM paper's metric.",
]

FUTURE_SCOPE = [
    "True message-level LOB replay and exact queue-position modeling.",
    "IMM's stable reference price, expert demonstration generation and an SAC + behavior-cloning objective.",
    "A TD3 baseline with an identical observation and action space.",
    "More seeds with confidence intervals, longer data periods and hyperparameter sweeps (on the validation split only).",
    "Matched news and LOB data for the same asset and period (Mode A), so that the sentiment channel can be evaluated causally.",
    "Calibrated fees, rebates and latency; multi-asset policies; explainability with feature attribution; offline RL variants.",
]

REFERENCES = [
    f"{IMM_CITATION}. Bibliographic details (authors, pages) should be copied from the paper PDF supplied with the project.",
    "Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063.",
    "Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. arXiv:1801.01290.",
    "Fujimoto, S., van Hoof, H., Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods (TD3). ICML 2018.",
    "Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS 2017.",
    "Raffin, A. et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. Journal of Machine Learning Research 22(268).",
    "Avellaneda, M., Stoikov, S. (2008). High-frequency trading in a limit order book. Quantitative Finance 8(3).",
    "Dong, Z. et al. (2024). FNSPID: A Comprehensive Financial News Dataset in Time Series. arXiv:2402.06698.",
]

VARIANT_LABELS = {
    "HEURISTIC": "Heuristic",
    "AVELLANEDA_STOIKOV": "Avellaneda-Stoikov",
    "SAC_LOB_ONLY": "SAC LOB Only",
    "SAC_LOB_TRANSFORMER": "SAC + Transformer",
    "SAC_LOB_SENTIMENT": "SAC + FinBERT",
    "SAC_FULL": "SAC Full",
}
VARIANT_ORDER = list(VARIANT_LABELS)
BASELINE_VARIANTS = ["HEURISTIC", "AVELLANEDA_STOIKOV"]

VARIANT_DESCRIPTIONS = {
    "HEURISTIC": "Symmetric fixed-spread two-level quoting with an inventory-skew offset; no learned model (baseline A).",
    "AVELLANEDA_STOIKOV": "Avellaneda-Stoikov (2008) reservation price and optimal spread mapped onto the same four-dimensional action; gamma/kappa from the config, untuned.",
    "SAC_LOB_ONLY": "SAC on LOB features + private state (no Transformer, no sentiment).",
    "SAC_LOB_TRANSFORMER": "SAC on LOB features + frozen Transformer embedding and trend probabilities + private state.",
    "SAC_LOB_SENTIMENT": "SAC on LOB features + FinBERT sentiment aggregates + private state.",
    "SAC_FULL": "SAC on LOB + Transformer + FinBERT sentiment + private state.",
}
