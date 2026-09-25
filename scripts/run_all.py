import argparse
import subprocess
import sys
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)

STAGES = [
    ("prepare_data", ["python", "scripts/prepare_data.py", "--config"]),
    ("run_finbert", ["python", "scripts/run_finbert.py", "--config"]),
    ("train_transformer", ["python", "scripts/train_transformer.py", "--config"]),
    ("train_agent", ["python", "scripts/train_agent.py", "--config"]),
    ("evaluate", ["python", "scripts/evaluate.py", "--config"]),
    ("generate_report", ["python", "scripts/generate_report.py"]),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    for stage_name, base_cmd in STAGES:
        cmd = base_cmd + ([args.config] if base_cmd[-1] == "--config" else [])
        logger.info("Running stage: %s (%s)", stage_name, " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error("Stage '%s' failed with exit code %d", stage_name, result.returncode)
            sys.exit(result.returncode)
    logger.info("run_all.py completed all stages successfully.")


if __name__ == "__main__":
    main()
