"""Build MarketMakingEnv instances from the feature artifacts of a variant + split."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import gymnasium as gym

from market_maker.env.market_making_env import MarketMakingEnv
from market_maker.features.pipeline import columns_path, lob_split_path, variant_path
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)


def evenly_spaced_starts(n_rows: int, episode_length: int, n: int) -> list[int]:
    """`n` deterministic episode start indices spread evenly over a split of `n_rows` rows.

    Valid starts are 0..n_rows-episode_length-1 (a full episode must fit). Windows do not overlap whenever
    n * episode_length <= n_rows - 1; otherwise they are still evenly spaced but overlap (a warning is logged)."""
    max_start = n_rows - episode_length - 1
    if max_start < 0:
        raise ValueError(f"split has {n_rows} rows, fewer than one episode ({episode_length} steps)")
    if n <= 1:
        return [0]
    if n * episode_length > n_rows - 1:
        logger.warning("%d windows of %d steps do not fit without overlap in %d rows", n, episode_length, n_rows)
    return [int(round(x)) for x in np.linspace(0, max_start, n)]


class RandomStartWrapper(gym.Wrapper):
    """Training-time wrapper: every reset draws a start index from its OWN seeded RNG, so a run is fully
    reproducible from its seed (the base env reseeds from OS entropy when reset() gets no seed)."""

    def __init__(self, env, seed: int):
        super().__init__(env)
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = dict(options or {})
        if "start_index" not in options:
            options["start_index"] = int(self._rng.integers(0, max(1, self.env.unwrapped._max_start + 1)))
        return self.env.reset(seed=seed, options=options)


def load_split_arrays(artifacts_dir: str, variant: str, split: str, mmap: bool = True):
    feats = np.load(variant_path(artifacts_dir, variant, split), mmap_mode="r" if mmap else None)
    with open(columns_path(artifacts_dir, variant)) as f:
        cols = json.load(f)
    lob = pd.read_parquet(lob_split_path(artifacts_dir, split))
    if len(lob) != len(feats):
        raise ValueError(f"{variant}/{split}: {len(feats)} feature rows vs {len(lob)} LOB rows (stale artifacts?)")
    return feats, cols, lob


def make_env(cfg, variant: str, split: str, artifacts_dir: str, mode: str = "eval", seed: int = 0):
    """mode='train': random start windows (seeded wrapper); mode='eval': raw env, callers pass
    reset(options={'start_index': i}) with fixed windows (see evenly_spaced_starts)."""
    feats, cols, lob = load_split_arrays(artifacts_dir, variant, split)
    sim = cfg.simulator
    env = MarketMakingEnv(
        fused_features=feats, raw_lob_df=lob, feature_columns=cols, tick_size=cfg.data.tick_size,
        levels_per_side=sim.levels_per_side, min_spread_ticks=sim.min_spread_ticks,
        max_spread_ticks=sim.max_spread_ticks, max_mid_offset_ticks=sim.max_mid_offset_ticks,
        total_volume_per_side=sim.total_volume_per_side, max_inventory=sim.max_inventory,
        fee_per_unit=sim.fee_per_unit, rebate_per_unit=sim.rebate_per_unit, latency_steps=sim.latency_steps,
        pnl_scale=sim.pnl_scale, reward_config=cfg.reward.model_dump(),
        reference_price_mode=sim.reference_price_mode, episode_length_steps=cfg.episode_steps,
    )
    if mode == "train":
        return RandomStartWrapper(env, seed)
    if mode != "eval":
        raise ValueError(f"mode must be 'train' or 'eval', got {mode}")
    return env
