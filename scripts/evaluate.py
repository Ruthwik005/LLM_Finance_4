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
from market_maker.evaluation.metrics import summarize_run
from market_maker.evaluation.run_metadata import build_run_metadata
from market_maker.agents.avellaneda_stoikov import AvellanedaStoikovMarketMaker
from market_maker.db.session import get_engine, get_session, init_db
from market_maker.db.writer import write_run_to_db
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default="heuristic_baseline")
    parser.add_argument("--variant", default="HEURISTIC",
                         help="HEURISTIC, AVELLANEDA_STOIKOV, or a SAC_* variant label (SAC requires --model-path)")
    parser.add_argument("--model-path", default=None,
                         help="Path to a trained SAC model (e.g. artifacts/rl/model.zip). "
                              "If omitted, falls back to the heuristic market maker.")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--data-mode", choices=["mode_b", "synthetic_smoke_test"], default=None,
                         help="Explicitly label the data mode for this run's metadata/report banner. "
                              "If omitted, inferred from the config filename (configs/fast_demo.yaml "
                              "is the config populated by `make sample-data` and defaults to "
                              "synthetic_smoke_test; any other config defaults to mode_b).")
    args = parser.parse_args()

    if args.data_mode is not None:
        data_mode = args.data_mode
    else:
        data_mode = "synthetic_smoke_test" if os.path.basename(args.config) == "fast_demo.yaml" else "mode_b"

    cfg = load_config(args.config)
    df = load_lob(cfg.data.lob_path)
    df, _ = validate_lob(df)
    splits = chronological_split(df, "timestamp", **cfg.data.split.model_dump())

    test_feats = compute_lob_features(splits["test"], depth=cfg.data.depth, return_lags=[1, 5],
                                       vol_window=cfg.baselines.volatility_window)
    feature_cols = [c for c in test_feats.columns if c != "timestamp"]
    fused = test_feats[feature_cols].to_numpy(dtype=np.float32)
    episode_steps = int((cfg.simulator.episode_minutes * 60 * 1000) / cfg.simulator.action_interval_ms)

    env = MarketMakingEnv(
        fused_features=fused, raw_lob_df=splits["test"].reset_index(drop=True), feature_columns=feature_cols,
        tick_size=cfg.data.tick_size, levels_per_side=cfg.simulator.levels_per_side,
        min_spread_ticks=cfg.simulator.min_spread_ticks, max_spread_ticks=cfg.simulator.max_spread_ticks,
        max_mid_offset_ticks=cfg.simulator.max_mid_offset_ticks,
        total_volume_per_side=cfg.simulator.total_volume_per_side, max_inventory=cfg.simulator.max_inventory,
        fee_per_unit=cfg.simulator.fee_per_unit, rebate_per_unit=cfg.simulator.rebate_per_unit,
        latency_steps=cfg.simulator.latency_steps, pnl_scale=cfg.simulator.pnl_scale, reward_config=cfg.reward.model_dump(),
        reference_price_mode=cfg.simulator.reference_price_mode, episode_length_steps=episode_steps,
    )
    variant = args.variant.upper()
    if args.model_path:
        from stable_baselines3 import SAC
        model = SAC.load(args.model_path)
        model_path = args.model_path
        result = run_backtest(env, model, n_episodes=args.n_episodes, is_heuristic=False,
                               max_inventory=cfg.simulator.max_inventory)
    else:
        if variant == "AVELLANEDA_STOIKOV":
            agent = AvellanedaStoikovMarketMaker(
                gamma=cfg.baselines.gamma, kappa=cfg.baselines.kappa,
                rolling_vol_index=feature_cols.index("rolling_vol"), midprice_index=feature_cols.index("midprice"),
                tick_size=cfg.data.tick_size, min_spread_ticks=cfg.simulator.min_spread_ticks,
                max_spread_ticks=cfg.simulator.max_spread_ticks,
                max_mid_offset_ticks=cfg.simulator.max_mid_offset_ticks, episode_length_steps=episode_steps)
        elif variant == "HEURISTIC":
            agent = HeuristicMarketMaker()
        else:
            raise SystemExit(f"Variant {args.variant} needs --model-path (trained SAC model).")
        model_path = "N/A"
        result = run_backtest(env, agent, n_episodes=args.n_episodes, is_heuristic=True,
                               max_inventory=cfg.simulator.max_inventory)
    summary = summarize_run(result, cfg)

    run_dir = f"artifacts/evaluation/{args.run_id}"
    write_run_artifacts(run_dir, config=cfg.model_dump(),
                         metadata=build_run_metadata(cfg, data_mode=data_mode, model_path=model_path, variant=variant,
                                                      splits=splits),
                         backtest_result=result, metrics=summary)

    engine = get_engine(f"sqlite:///{os.environ.get('MARKET_MAKER_DB_PATH', 'artifacts/market_maker.db')}")
    init_db(engine)
    session = get_session(engine)
    try:
        write_run_to_db(session, run_id=args.run_id, variant=variant, seed=cfg.project.seed,
                         model_path=model_path, config=cfg.model_dump(), backtest_result=result, metrics=summary)
    finally:
        session.close()

    logger.info("Evaluation complete for run_id=%s. Summary: %s", args.run_id, summary)


if __name__ == "__main__":
    main()
