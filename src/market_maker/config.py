"""Configuration loader using Pydantic models."""
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
    lr: float = 1e-3
    weight_decay: float = 1e-2
    patience: int = 3  # early stopping on validation loss (epochs without improvement)
    num_workers: int = 0
    window_stride: int = 1  # stride between consecutive training/validation windows (rows)
    class_weights: bool = False  # weight CE by inverse train label frequency (flat class usually dominates)
    embed_batch_size: int = 1024  # batch size when exporting embeddings for every row


class FeaturesConfig(BaseModel):
    vol_window: int = 20  # rolling window (snapshots) of the rolling_vol feature
    return_lags: list[int] = [1, 5]
    clip_value: float = 10.0  # standardized features are clipped to +-clip_value
    label_threshold_bps: float = 5.0  # Transformer up/down label threshold (basis points of mid price)


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
    rebate_per_unit: float = 0.0
    pnl_scale: float = 50.0  # tanh scale for the mark-to-market PnL entry of the observation


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
    net_arch: list[int] = [256, 256]
    eval_freq: int = 5_000  # steps between deterministic validation evaluations
    n_eval_episodes: int = 5  # fixed validation windows per evaluation
    checkpoint_freq: int = 0  # optional extra periodic checkpoints every N steps (0 = off)
    device: str = "auto"  # SB3 device ("cpu" is usually fastest for small MLPs)


class EvaluationConfig(BaseModel):
    adverse_horizon_steps: int = 10
    adverse_threshold_ticks: float = 1.0


class BaselinesConfig(BaseModel):
    gamma: float = 0.1  # Avellaneda-Stoikov risk aversion
    kappa: float = 100.0  # arrival-intensity decay per price unit; spread term ~ 2/kappa
    volatility_window: int = 20  # rolling window (snapshots) of the rolling_vol feature


class ExperimentsConfig(BaseModel):
    run_heuristic: bool = True
    run_lob_only: bool = True
    run_lob_transformer: bool = True
    run_lob_sentiment: bool = True
    run_full: bool = True
    run_td3: bool = False


class ExperimentConfig(BaseModel):
    name: str = "experiment"
    experiment_id: str | None = None  # defaults to `name`
    variants: list[str] = ["SAC_LOB_ONLY", "SAC_LOB_TRANSFORMER", "SAC_LOB_SENTIMENT", "SAC_FULL"]
    seeds: list[int] = [0, 1, 2]
    n_val_episodes: int | None = None  # None -> sac.n_eval_episodes
    n_test_episodes: int = 20
    window_selection: str = "evenly_spaced"  # deterministic evenly spaced val/test windows


class AppConfig(BaseModel):
    project: ProjectConfig
    data: DataConfig
    sentiment: SentimentConfig
    transformer: TransformerConfig
    simulator: SimulatorConfig
    reward: RewardConfig
    sac: SacConfig
    experiments: ExperimentsConfig
    evaluation: EvaluationConfig = EvaluationConfig()
    baselines: BaselinesConfig = BaselinesConfig()
    features: FeaturesConfig = FeaturesConfig()
    experiment: ExperimentConfig = ExperimentConfig()

    @property
    def episode_steps(self) -> int:
        return int((self.simulator.episode_minutes * 60 * 1000) / self.simulator.action_interval_ms)

    @property
    def experiment_id(self) -> str:
        return self.experiment.experiment_id or self.experiment.name


def load_config(path: str) -> AppConfig:
    """Load configuration from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
