import os
from market_maker.evaluation.backtest import write_run_artifacts
from market_maker.evaluation.report import generate_report


def test_generate_report_includes_actual_metrics(tmp_path):
    run_dir = str(tmp_path / "run_a")
    write_run_artifacts(run_dir, config={"seed": 1}, metadata={"seed": 1, "model_path": "N/A"},
                         backtest_result={"episode_pnls": [5.0], "inventories": [], "fills": [[]], "orders": [[]]},
                         metrics={"mean_pnl": 5.0, "map": 2.0})
    out_path = str(tmp_path / "FINAL_PROJECT_REPORT.md")
    text = generate_report(runs_root=str(tmp_path), output_path=out_path)
    assert os.path.exists(out_path)
    assert "5.0" in text
    assert "Abstract" in text


def test_generate_report_marks_missing_data_explicitly(tmp_path):
    empty_root = str(tmp_path / "empty")
    os.makedirs(empty_root)
    out_path = str(tmp_path / "REPORT.md")
    text = generate_report(runs_root=empty_root, output_path=out_path)
    assert "Not run / insufficient data" in text
