"""Train ONE variant/seed with the same machinery as run_experiments (validation-based best checkpoint).

Features are built on demand for variants that need no Transformer (SAC_LOB_ONLY; SAC_LOB_SENTIMENT needs
sentiment). `--out` additionally copies the best model to a flat path (e.g. artifacts/rl/model.zip).
"""
import argparse
import os
import shutil
from market_maker.config import load_config
from market_maker.features.pipeline import ensure_features
from market_maker.training.env_factory import evenly_spaced_starts, make_env
from market_maker.training.train_sac import train_sac_agent
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default="SAC_LOB_ONLY")
    parser.add_argument("--seed", type=int, default=None, help="default: project.seed")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--out", default=None, help="optional extra copy of the best model (e.g. artifacts/rl/model.zip)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    variant = args.variant.upper()
    seed = cfg.project.seed if args.seed is None else args.seed
    ensure_features(cfg, args.artifacts_root, variant)
    train_env = make_env(cfg, variant, "train", args.artifacts_root, "train", seed=seed)
    val_env = make_env(cfg, variant, "validation", args.artifacts_root, "eval")
    n_val = cfg.experiment.n_val_episodes or cfg.sac.n_eval_episodes
    val_starts = evenly_spaced_starts(len(val_env.raw_lob_df), cfg.episode_steps, n_val)
    out_dir = os.path.join(args.artifacts_root, "rl", variant.lower(), f"seed{seed}")
    res = train_sac_agent(train_env, val_env, cfg.sac, seed=seed, out_dir=out_dir, val_starts=val_starts,
                          total_timesteps=args.total_timesteps, max_inventory=cfg.simulator.max_inventory)
    logger.info("Best model (step %s, val reward %.4f): %s", res.best_step, res.best_val_reward, res.best_model_path)
    out = args.out or os.path.join(args.artifacts_root, "rl", "model.zip")  # flat copy: Makefile compatibility
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    shutil.copyfile(res.best_model_path, out)
    logger.info("Copied best model to %s", out)


if __name__ == "__main__":
    main()
