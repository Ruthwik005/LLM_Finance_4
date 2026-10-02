"""Run variants x seeds (train on train split, select checkpoint on validation, report on fixed test windows),
plus HEURISTIC / AVELLANEDA_STOIKOV baselines, and aggregate mean +- std across seeds.

Requires feature artifacts (scripts/build_features.py, and scripts/train_transformer.py for the Transformer
variants). Missing LOB-only/sentiment features are built on demand; Transformer variants need the checkpoint.
"""
import argparse
import json
from market_maker.config import load_config
from market_maker.training.experiment import infer_data_mode, plan_runs, run_experiment
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--experiment-id", default=None)
    p.add_argument("--fast", action="store_true", help="tiny budgets (smoke test): 600 steps, 2 val / 3 test windows")
    p.add_argument("--dry-run", action="store_true", help="print the plan and estimated totals, train nothing")
    p.add_argument("--resume", action="store_true", help="skip runs whose artifacts/evaluation/<run_id>/ is complete")
    p.add_argument("--artifacts-root", default="artifacts")
    p.add_argument("--db-path", default=None, help="SQLite file (default $MARKET_MAKER_DB_PATH or <artifacts>/market_maker.db)")
    p.add_argument("--data-mode", choices=["mode_b", "synthetic_smoke_test"], default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    variants = [v.upper() for v in (args.variants or cfg.experiment.variants)]
    seeds = args.seeds if args.seeds is not None else cfg.experiment.seeds
    if args.dry_run:
        plan = plan_runs(cfg, variants, seeds, fast=args.fast)
        print(json.dumps(plan, indent=2))
        print(f"{len(plan['runs'])} SAC runs, {plan['total_train_timesteps']:,} total training timesteps "
              f"(+ ~{plan['total_validation_env_steps']:,} validation and ~{plan['total_test_env_steps']:,} test env steps).")
        return
    summary = run_experiment(cfg, artifacts_dir=args.artifacts_root, db_path=args.db_path, variants=variants,
                             seeds=seeds, experiment_id=args.experiment_id, fast=args.fast, resume=args.resume,
                             data_mode=args.data_mode or infer_data_mode(args.config), config_path=args.config)
    logger.info("Experiment %s finished. Summary: %s/evaluation/%s_summary.json", summary["experiment_id"],
                args.artifacts_root, summary["experiment_id"])
    for v, ms in summary["aggregates"].items():
        m = ms.get("mean_pnl")
        if m:
            logger.info("%-22s mean_pnl %.4f +- %s (n=%d)", v, m["mean"], m["std"], m["n_seeds"])


if __name__ == "__main__":
    main()
