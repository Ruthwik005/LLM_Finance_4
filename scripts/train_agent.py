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
        fee_per_unit=cfg.simulator.fee_per_unit, rebate_per_unit=cfg.simulator.rebate_per_unit,
        latency_steps=cfg.simulator.latency_steps, pnl_scale=cfg.simulator.pnl_scale,
        reward_config=cfg.reward.model_dump(), reference_price_mode=cfg.simulator.reference_price_mode,
        episode_length_steps=episode_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default="SAC_LOB_ONLY")
    parser.add_argument("--out", default="artifacts/rl/model.zip")
    args = parser.parse_args()

    if args.variant != "SAC_LOB_ONLY":
        raise NotImplementedError(
            f"Variant {args.variant} requires precomputed sentiment/transformer artifacts. "
            f"Only SAC_LOB_ONLY is currently trainable end-to-end. "
            f"Other variants (SAC_LOB_TRANSFORMER, SAC_LOB_SENTIMENT, SAC_FULL) require "
            f"wiring in sentiment/transformer artifacts from Tasks 5/7."
        )

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
