"""Pydantic response models for the read-only results API."""
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str


class RunList(BaseModel):
    run_ids: list[str]


class RunDetail(BaseModel):
    run_id: str
    variant: str
    seed: int
    model_path: str
    created_at: str
    data_mode: Optional[str] = None


class EpisodePnl(BaseModel):
    episode: int
    pnl: float


class FillRow(BaseModel):
    episode: int
    side: str
    price: float
    volume: float


class EquityPoint(BaseModel):
    episode: int
    cumulative_pnl: float


class RunSummaryRow(BaseModel):
    run_id: str
    variant: str
    seed: int
    data_mode: Optional[str] = None
    created_at: str
    mean_pnl: Optional[float] = None
    map: Optional[float] = None
    max_drawdown: Optional[float] = None
    n_fills: Optional[float] = None
    sharpe_like: Optional[float] = None
    pnl_over_map: Optional[float] = None
    metrics: dict[str, float] = Field(default_factory=dict)


class RunsSummary(BaseModel):
    runs: list[RunSummaryRow]


class ReplayStep(BaseModel):
    step: int
    timestamp: Optional[str] = None
    mid: float
    bid_1: float
    bid_2: float
    ask_1: float
    ask_2: float
    bid_vol_1: float
    bid_vol_2: float
    ask_vol_1: float
    ask_vol_2: float
    inventory: float
    mtm_pnl: float
    placed_volume: float
    canceled_volume: float


class ReplayFill(BaseModel):
    side: str
    price: float
    volume: float
    step: Optional[int] = None
    timestamp: Optional[str] = None


class ReplaySummary(BaseModel):
    """Whole-episode aggregates computed over ALL steps (not the downsampled subset)."""
    n_fills: int
    filled_volume: float
    final_mtm_pnl: Optional[float] = None
    max_abs_inventory: Optional[float] = None
    avg_quoted_spread: Optional[float] = None
    cancel_volume: float
    placed_volume: float


class Replay(BaseModel):
    run_id: str
    episode: int
    n_steps_total: int
    n_points: int
    downsampled: bool
    stride: int
    steps: list[ReplayStep]
    fills: list[ReplayFill]
    summary: ReplaySummary


class TrainingPoint(BaseModel):
    timesteps: float
    train_reward_mean: Optional[float] = None
    val_reward_mean: Optional[float] = None
    val_pnl_mean: Optional[float] = None
    val_map: Optional[float] = None
    ent_coef: Optional[float] = None
    critic_loss: Optional[float] = None
    actor_loss: Optional[float] = None
    is_best: bool = False


class TrainingCurve(BaseModel):
    run_id: str
    points: list[TrainingPoint]


class MetricAggregate(BaseModel):
    mean: float
    std: Optional[float] = None
    n_seeds: int


class VariantAggregate(BaseModel):
    variant: str
    n_seeds: int
    metrics: dict[str, MetricAggregate]


class PairwiseComparison(BaseModel):
    variant: str
    metric: str
    n_a: Optional[int] = None
    n_b: Optional[int] = None
    mean_diff: Optional[float] = None
    t: Optional[float] = None
    p_value: Optional[float] = None
    cohens_d: Optional[float] = None
    note: Optional[str] = None


class ExperimentListItem(BaseModel):
    experiment_id: str
    data_mode: Optional[str] = None
    created_utc: Optional[str] = None
    n_variants: int
    has_summary_file: bool


class ExperimentList(BaseModel):
    experiments: list[ExperimentListItem]


class ExperimentDetail(BaseModel):
    experiment_id: str
    data_mode: Optional[str] = None
    created_utc: Optional[str] = None
    seeds: list[int] = Field(default_factory=list)
    variants: list[VariantAggregate]
    reference_variant: str = "SAC_LOB_ONLY"
    comparisons: list[PairwiseComparison] = Field(default_factory=list)
    comparison_note: Optional[str] = None
    comparison_label: str = "exploratory, small n"


class PnlAttribution(BaseModel):
    run_id: str
    spread_capture_pnl: Optional[float] = None
    inventory_pnl: Optional[float] = None
    fees_paid: Optional[float] = None
    rebates: Optional[float] = None
    total: Optional[float] = None


class SignalsUnavailable(BaseModel):
    available: bool = False
    reason: str


class SignalsPayload(BaseModel):
    available: bool = True
    split: str
    n_rows_total: int
    n_points: int
    stride: int
    timestamps: list[str]
    midprice: Optional[list[Optional[float]]] = None
    series: dict[str, list[Optional[float]]]
    notes: list[str] = Field(default_factory=list)


JsonAny = Any
