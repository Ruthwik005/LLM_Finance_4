import json
import os
import socket
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from market_maker.config import load_config
from market_maker.orchestration import STAGES, PipelineError, build_plan, run_plan, select_stages

ROOT = Path(__file__).resolve().parents[1]


def _args(tmp_path, **kw):
    base = dict(config="configs/fast_demo.yaml", fast=True, stub_sentiment=False, from_stage=None, only_stage=None,
                dry_run=False, resume=False, force_features=False, artifacts_root=str(tmp_path / "art"), db_path=None,
                experiment_id=None, variants=None, seeds=None, data_mode=None)
    base.update(kw)
    return Namespace(**base)


def _cfg(tmp_path, news=True):
    cfg = load_config(str(ROOT / "configs/fast_demo.yaml"))
    p = tmp_path / "news.parquet"
    if news:
        p.write_bytes(b"x")
    return cfg.model_copy(update={"data": cfg.data.model_copy(update={"news_path": str(p)})})


def test_stage_order_and_selection():
    assert STAGES == ["prepare_data", "build_features", "train_transformer", "run_experiments", "generate_report"]
    assert select_stages(None, None) == STAGES
    assert select_stages("run_experiments", None) == ["run_experiments", "generate_report"]
    assert select_stages(None, "generate_report") == ["generate_report"]
    with pytest.raises(PipelineError):
        select_stages("prepare_data", "generate_report")


def test_real_finbert_is_default_and_stub_is_explicit(tmp_path):
    cfg = _cfg(tmp_path)
    real = build_plan(cfg, _args(tmp_path))
    flat = " ".join(" ".join(c) for cmds in real["stages"].values() for c in cmds)
    assert "--stub-sentiment" not in flat and "--no-sentiment" not in flat
    assert real["data_mode"] is None and not real["skipped_variants"]
    stub = build_plan(cfg, _args(tmp_path, stub_sentiment=True))
    assert "--stub-sentiment" in stub["stages"]["build_features"][0]
    assert "synthetic_smoke_test" in stub["stages"]["run_experiments"][0]


def test_commands_use_current_interpreter_and_flags(tmp_path):
    plan = build_plan(_cfg(tmp_path), _args(tmp_path, resume=True, db_path="x.db", seeds=[0, 1]))
    for cmds in plan["stages"].values():
        for c in cmds:
            assert c[0] == sys.executable
    exp = plan["stages"]["run_experiments"][0]
    assert "--fast" in exp and "--resume" in exp and "--db-path" in exp and exp[exp.index("--experiment-id") + 1] == "fast_demo_fast"
    assert exp[exp.index("--seeds") + 1:exp.index("--seeds") + 3] == ["0", "1"]
    assert plan["stages"]["train_transformer"][1][-5:-4] != []  # fusion command follows the Transformer training
    assert "fusion" in plan["stages"]["train_transformer"][1]


def test_no_news_skips_sentiment_variants_with_reason(tmp_path):
    plan = build_plan(_cfg(tmp_path, news=False), _args(tmp_path))
    assert set(plan["skipped_variants"]) == {"SAC_LOB_SENTIMENT", "SAC_FULL"}
    assert "no news file" in plan["skipped_variants"]["SAC_FULL"]
    assert "--no-sentiment" in plan["stages"]["build_features"][0]
    assert plan["variants"] == ["SAC_LOB_ONLY", "SAC_LOB_TRANSFORMER"]
    assert "SAC_FULL" not in plan["stages"]["run_experiments"][0]


def test_no_runnable_variants_is_an_error(tmp_path):
    with pytest.raises(PipelineError):
        build_plan(_cfg(tmp_path, news=False), _args(tmp_path, variants=["SAC_FULL"]))


def test_dry_run_executes_nothing(tmp_path, capsys):
    code = run_plan(_cfg(tmp_path), _args(tmp_path, dry_run=True))
    assert code == 0
    out = capsys.readouterr().out
    assert "dry run" in out and "run_experiments.py" in out
    assert not os.path.exists(tmp_path / "art")


def test_failure_stops_with_nonzero_and_records_status(tmp_path):
    import yaml
    raw = yaml.safe_load(open(ROOT / "configs/fast_demo.yaml"))
    raw["data"]["lob_path"] = str(tmp_path / "missing.parquet")
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))
    code = run_plan(load_config(str(cfg_path)), _args(tmp_path, config=str(cfg_path)))
    assert code != 0
    st = json.load(open(tmp_path / "art" / "logs" / "pipeline_status.json"))
    assert st["stages"]["prepare_data"]["status"] == "failed"
    assert "build_features" not in st["stages"]


def test_cli_dry_run_subprocess():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/run_all.py"), "--config", str(ROOT / "configs/fast_demo.yaml"),
                        "--fast", "--stub-sentiment", "--dry-run"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "[generate_report]" in r.stdout


def test_launch_demo_dry_run_and_busy_port():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/launch_demo.py"), "--dry-run"], capture_output=True, text=True,
                       env=dict(os.environ, API_PORT="8123"))
    assert r.returncode == 0 and "--port 8123" in r.stdout and "VITE_API_URL=http://localhost:8123" in r.stdout
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    port = s.getsockname()[1]
    try:
        r = subprocess.run([sys.executable, str(ROOT / "scripts/launch_demo.py"), "--api-only"], capture_output=True, text=True,
                           env=dict(os.environ, API_PORT=str(port)))
    finally:
        s.close()
    assert r.returncode == 2 and "already in use" in r.stderr and "VITE_API_URL" in r.stderr
