"""Evaluate one agent on the fixed TEST windows with the scaled feature artifacts.

HEURISTIC / AVELLANEDA_STOIKOV need no model. SAC_* variants load --model-path (default
artifacts/rl/<variant>/seed<seed>/best_model.zip). Writes run artifacts + DB rows like run_experiments does.
"""
import argparse
import os
from market_maker.config import load_config
from market_maker.agents.heuristic import HeuristicMarketMaker
from market_maker.agents.avellaneda_stoikov import AvellanedaStoikovMarketMaker
from market_maker.evaluation.backtest import run_backtest, write_run_artifacts
from market_maker.evaluation.metrics import summarize_run
from market_maker.evaluation.run_metadata import build_run_metadata
from market_maker.db.session import get_engine, get_session, init_db
from market_maker.db.writer import write_run_to_db
from market_maker.features.pipeline import SPLITS, ensure_features, load_meta, lob_split_path
from market_maker.training.env_factory import evenly_spaced_starts, make_env
from market_maker.training.experiment import infer_data_mode
from market_maker.utils.logging import get_logger
import pandas as pd

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None, help="default: heuristic_baseline for HEURISTIC, else <variant>[_seed<N>]")
    parser.add_argument("--variant", default="HEURISTIC",
                        help="HEURISTIC, AVELLANEDA_STOIKOV, or SAC_LOB_ONLY|SAC_LOB_TRANSFORMER|SAC_LOB_SENTIMENT|SAC_FULL")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-episodes", type=int, default=None, help="default: experiment.n_test_episodes")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--data-mode", choices=["mode_b", "synthetic_smoke_test"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    variant = args.variant.upper()
    seed = cfg.project.seed if args.seed is None else args.seed
    data_mode = args.data_mode or infer_data_mode(args.config)
    feat_variant = variant if variant.startswith("SAC_") else "SAC_LOB_ONLY"
    ensure_features(cfg, args.artifacts_root, feat_variant)
    env = make_env(cfg, feat_variant, "test", args.artifacts_root, "eval")
    n = args.n_episodes or cfg.experiment.n_test_episodes
    starts = evenly_spaced_starts(len(env.raw_lob_df), cfg.episode_steps, n)
    L = cfg.episode_steps

    if variant.startswith("SAC_"):
        from stable_baselines3 import SAC
        model_path = args.model_path or os.path.join(args.artifacts_root, "rl", variant.lower(), f"seed{seed}",
                                                     "best_model.zip")
        agent, heur = SAC.load(model_path, device=cfg.sac.device), False
        run_id = args.run_id or f"{variant.lower()}_seed{seed}"
    elif variant == "AVELLANEDA_STOIKOV":
        agent, heur, model_path = AvellanedaStoikovMarketMaker(
            gamma=cfg.baselines.gamma, kappa=cfg.baselines.kappa, tick_size=cfg.data.tick_size,
            min_spread_ticks=cfg.simulator.min_spread_ticks, max_spread_ticks=cfg.simulator.max_spread_ticks,
            max_mid_offset_ticks=cfg.simulator.max_mid_offset_ticks, episode_length_steps=L,
            vol_window=cfg.baselines.volatility_window), True, "N/A"
        run_id = args.run_id or "avellaneda_stoikov"
    elif variant == "HEURISTIC":
        agent, heur, model_path = HeuristicMarketMaker(), True, "N/A"
        run_id = args.run_id or "heuristic_baseline"
    else:
        raise SystemExit(f"unknown variant {args.variant}")

    result = run_backtest(env, agent, n_episodes=len(starts), is_heuristic=heur,
                          max_inventory=cfg.simulator.max_inventory, start_indices=starts)
    summary = summarize_run(result, cfg)
    frames = {s: pd.read_parquet(lob_split_path(args.artifacts_root, s), columns=["timestamp"]) for s in SPLITS}
    md = build_run_metadata(cfg, data_mode=data_mode, model_path=model_path, variant=variant, splits=frames)
    md.update({"seed": seed, "test_windows": starts})
    write_run_artifacts(os.path.join(args.artifacts_root, "evaluation", run_id), config=cfg.model_dump(),
                        metadata=md, backtest_result=result, metrics=summary)

    db_path = args.db_path or os.environ.get("MARKET_MAKER_DB_PATH") or os.path.join(args.artifacts_root, "market_maker.db")
    engine = get_engine(f"sqlite:///{db_path}")
    init_db(engine)
    session = get_session(engine)
    try:
        write_run_to_db(session, run_id=run_id, variant=variant, seed=seed, model_path=model_path,
                        config=cfg.model_dump(), backtest_result=result, metrics=summary)
    finally:
        session.close()
    logger.info("Evaluation complete for run_id=%s. Summary: %s", run_id, summary)


if __name__ == "__main__":
    main()
