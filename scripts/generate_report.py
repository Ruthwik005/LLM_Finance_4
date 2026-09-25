import argparse
from market_maker.evaluation.report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="artifacts/evaluation")
    parser.add_argument("--out", default="artifacts/reports/FINAL_PROJECT_REPORT.md")
    args = parser.parse_args()
    generate_report(runs_root=args.runs_root, output_path=args.out)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
