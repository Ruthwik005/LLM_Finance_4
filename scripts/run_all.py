import argparse
import os
import subprocess
import sys
from market_maker.config import load_config
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)

STAGES = [
    ("prepare_data", [sys.executable, "scripts/prepare_data.py", "--config"]),
    ("run_finbert", [sys.executable, "scripts/run_finbert.py", "--config"]),
    ("train_transformer", [sys.executable, "scripts/train_transformer.py", "--config"]),
    ("train_agent", [sys.executable, "scripts/train_agent.py", "--config"]),
    ("evaluate", [sys.executable, "scripts/evaluate.py", "--config"]),
    ("generate_report", [sys.executable, "scripts/generate_report.py"]),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    for stage_name, base_cmd in STAGES:
        if stage_name == "run_finbert" and not os.path.exists(cfg.data.news_path):
            logger.info("Skipping stage 'run_finbert': no news file at %s (news-optional mode).",
                        cfg.data.news_path)
            continue
        cmd = base_cmd + ([args.config] if base_cmd[-1] == "--config" else [])
        logger.info("Running stage: %s (%s)", stage_name, " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error("Stage '%s' failed with exit code %d", stage_name, result.returncode)
            sys.exit(result.returncode)
    logger.info("run_all.py completed all stages successfully.")


if __name__ == "__main__":
    main()
