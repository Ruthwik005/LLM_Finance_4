"""Generate artifacts/reports/FINAL_PROJECT_REPORT.md (+ figures/) from the SQLite DB and run artifacts.

Sources: --db-path / $MARKET_MAKER_DB_PATH, --artifacts-root / $MARKET_MAKER_ARTIFACTS_ROOT (flags win).
Missing data is rendered as "N/A (reason)" or "Not run / insufficient data"; nothing is invented.
"""
import argparse
import os
from market_maker.evaluation.report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", default=None, help="default: $MARKET_MAKER_ARTIFACTS_ROOT or artifacts")
    parser.add_argument("--db-path", default=None, help="default: $MARKET_MAKER_DB_PATH or <artifacts-root>/market_maker.db")
    parser.add_argument("--runs-root", default=None, help="default: <artifacts-root>/evaluation")
    parser.add_argument("--out", default=None, help="default: <artifacts-root>/reports/FINAL_PROJECT_REPORT.md")
    parser.add_argument("--experiment-id", default=None, help="default: the most recent experiment in the DB")
    parser.add_argument("--config", default=None, help="optional config path used only for reproduction commands / as config fallback")
    args = parser.parse_args()
    root = args.artifacts_root or os.environ.get("MARKET_MAKER_ARTIFACTS_ROOT") or "artifacts"
    out = args.out or os.path.join(root, "reports", "FINAL_PROJECT_REPORT.md")
    generate_report(runs_root=args.runs_root or os.path.join(root, "evaluation"), output_path=out, artifacts_root=root,
                    db_path=args.db_path, experiment_id=args.experiment_id, config_path=args.config)
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
