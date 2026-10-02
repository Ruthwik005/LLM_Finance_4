"""Run the whole pipeline: prepare_data -> build_features -> train_transformer -> run_experiments -> generate_report.

Examples
  python scripts/run_all.py --config configs/fast_demo.yaml --fast --stub-sentiment      # synthetic smoke test
  python scripts/run_all.py --config configs/real_binance.yaml --dry-run                  # show what would run
  python scripts/run_all.py --config configs/real_binance.yaml --resume                   # real run (long)
  python scripts/run_all.py --config configs/real_binance.yaml --from-stage run_experiments --resume

Real FinBERT is used unless --stub-sentiment is passed (then results are labeled SOFTWARE SMOKE TEST ONLY).
If no news file exists, sentiment variants are skipped and recorded as N/A (news-optional).
--fast only shrinks the experiment (600 SAC steps, few windows); the Transformer uses the config's epochs.
"""
import argparse
import sys

from market_maker.config import load_config
from market_maker.orchestration import STAGES, PipelineError, run_plan
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--fast", action="store_true", help="smoke-test budgets in run_experiments (experiment id gets a _fast suffix)")
    p.add_argument("--stub-sentiment", action="store_true",
                   help="SMOKE TESTS ONLY: synthetic stub scores instead of FinBERT; marks the run synthetic_smoke_test")
    p.add_argument("--from-stage", choices=STAGES, default=None, help="start at this stage and run all later ones")
    p.add_argument("--only-stage", choices=STAGES, default=None, help="run just this stage")
    p.add_argument("--dry-run", action="store_true", help="print the commands, execute nothing")
    p.add_argument("--resume", action="store_true", help="pass --resume to run_experiments (skip complete, matching runs)")
    p.add_argument("--force-features", action="store_true", help="pass --force-features to run_experiments")
    p.add_argument("--artifacts-root", default="artifacts")
    p.add_argument("--db-path", default=None, help="SQLite file (default <artifacts-root>/market_maker.db)")
    p.add_argument("--experiment-id", default=None)
    p.add_argument("--variants", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--data-mode", choices=["mode_b", "synthetic_smoke_test"], default=None)
    args = p.parse_args()
    try:
        code = run_plan(load_config(args.config), args)
    except (PipelineError, FileNotFoundError) as e:
        logger.error("%s", e)
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
