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
