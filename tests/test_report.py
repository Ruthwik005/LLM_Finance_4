import json
import os
import re

import numpy as np
import pytest

from market_maker.db.session import get_engine, get_session, init_db
from market_maker.db.writer import write_experiment_aggregates, write_run_to_db
from market_maker.evaluation.backtest import write_run_artifacts
from market_maker.evaluation.report import SECTIONS, fmt_num, generate_report
from market_maker.evaluation.report_text import FILL_DISCLAIMER, MISSING, SMOKE_BANNER

FORBIDDEN = ["outperforms", "outperform ", "significantly better", "faster convergence", "higher profitability",
             "lower adverse selection", "better inventory control", "outperforms IMM", "statistically significant",
             "proves profitability", "guaranteed"]


def _steps(n=6, mid0=100.0):
    out = []
    for t in range(n):
        out.append({"t": t, "timestamp": f"2024-01-01T00:00:0{t}+00:00", "mid": mid0 + t * 0.01, "next_mid": mid0 + (t + 1) * 0.01,
                    "bid_prices": [mid0 - 0.01, mid0 - 0.02], "ask_prices": [mid0 + 0.01, mid0 + 0.02],
                    "bid_volumes": [5.0, 5.0], "ask_volumes": [5.0, 5.0], "inventory": float(t % 3), "cash": 0.0,
                    "mtm_pnl": 0.1 * t, "fills": [], "placed_volume": 1.0, "canceled_volume": 0.5})
    return out


def _result(pnl):
    return {"episode_pnls": [pnl, pnl], "inventories": [np.array([0.0, 1.0]), np.array([0.0, 2.0])],
            "fills": [[{"side": "bid", "price": 99.99, "volume": 1.0, "step": 1}], []], "orders": [[], []],
            "steps": [_steps(), _steps()]}


def build_populated(tmp_path, data_mode="mode_b", with_curve=True):
    art = tmp_path / "artifacts"
    ev = art / "evaluation"
    db = str(art / "market_maker.db")
    os.makedirs(ev, exist_ok=True)
    engine = get_engine(f"sqlite:///{db}")
    init_db(engine)
    session = get_session(engine)
    cfg = {"simulator": {"max_inventory": 100, "max_mid_offset_ticks": 4, "min_spread_ticks": 1, "max_spread_ticks": 8,
                         "total_volume_per_side": 20, "latency_steps": 0, "fee_per_unit": 0.0, "rebate_per_unit": 0.0,
                         "episode_minutes": 5, "action_interval_ms": 500, "reference_price_mode": "midprice"},
           "data": {"symbol": "TEST", "depth": 5, "tick_size": 0.01, "lob_path": "x.parquet", "news_path": "n.parquet"},
           "reward": {"type": "truncated_inventory", "inventory_threshold": 40, "inventory_penalty": 0.001,
                      "cancellation_penalty": 0.0001, "scale": 1.0, "adverse_selection_penalty": 0.0},
           "sac": {"learning_rate": 0.0003, "net_arch": [64, 64]}, "transformer": {}, "sentiment": {}, "features": {}, "evaluation": {}}
    runs = [("heuristic", "HEURISTIC", 0, 1.5)] + [(f"sac_lob_only_seed{s}", "SAC_LOB_ONLY", s, 2.0 + s) for s in range(3)] \
        + [(f"sac_full_seed{s}", "SAC_FULL", s, 3.0 + 0.5 * s) for s in range(3)]
    per_variant: dict = {}
    for rid, variant, seed, pnl in runs:
        metrics = {"mean_pnl": pnl, "map": 1.0 + seed, "pnl_over_map": pnl / (1.0 + seed), "n_fills": 5.0 + seed,
                   "adverse_selection_ratio": 0.3 + 0.05 * seed, "cancel_ratio": 0.5, "max_drawdown": 0.7,
                   "sharpe_like": 0.2, "fill_rate": 0.01, "avg_quoted_spread": 2.0, "spread_capture_pnl": -0.1,
                   "inventory_pnl": 0.2, "fees_paid": 0.0, "max_abs_inventory": 3.0}
        md = {"seed": seed, "variant": variant, "data_mode": data_mode, "git_commit": "abc123", "model_path": "N/A",
              "total_timesteps": 600, "config_hash": "cfghash", "python_version": "3.11.0",
              "date_ranges": {"train": {"start": "2024-01-01", "end": "2024-01-04", "rows": 100},
                              "validation": {"start": "2024-01-05", "end": "2024-01-05", "rows": 20},
                              "test": {"start": "2024-01-06", "end": "2024-01-06", "rows": 20}},
              "dataset_fingerprints": {"lob_sha256": "deadbeef", "news_sha256": "cafe"}}
        res = _result(pnl)
        write_run_artifacts(str(ev / rid), config=cfg, metadata=md, backtest_result=res, metrics=metrics)
        if with_curve and variant != "HEURISTIC":
            with open(ev / rid / "training_curve.csv", "w") as f:
                f.write("timesteps,train_reward_mean,val_reward_mean,val_pnl_mean,val_map,ent_coef,critic_loss,actor_loss,is_best\n"
                        f"300,nan,{-1.0 - seed},0,0,nan,nan,nan,0\n600,nan,{-0.5 - seed},0,0,nan,nan,nan,1\n")
        write_run_to_db(session, rid, variant, seed, "N/A", cfg, res, metrics)
        per_variant.setdefault(variant, []).append(metrics)
    rows = []
    for v, ms in per_variant.items():
        for k in ms[0]:
            vals = np.array([m[k] for m in ms])
            rows.append({"variant": v, "metric": k, "mean": float(vals.mean()),
                         "std": float(vals.std(ddof=1)) if len(vals) > 1 else None, "n_seeds": len(vals)})
    write_experiment_aggregates(session, "exp1", rows)
    session.close()
    summary = {"experiment_id": "exp1", "created_utc": "2026-01-01T00:00:00+00:00", "data_mode": data_mode, "fast_mode": False,
               "seeds": [0, 1, 2], "n_test_windows": 2, "test_windows": [0, 10], "n_val_windows": 1, "episode_steps": 6,
               "aggregates": {}, "comparisons_vs_SAC_LOB_ONLY": {"SAC_FULL": {"mean_pnl": {
                   "n_a": 3, "n_b": 3, "mean_diff": 1.0, "t": 2.0, "p_value": 0.2, "cohens_d": 1.5, "note": "x"}}}, "timings": {}}
    json.dump(summary, open(ev / "exp1_summary.json", "w"))
    return str(art), db, rows


def _gen(art, db, tmp_path, **kw):
    out = str(tmp_path / "out" / "FINAL_PROJECT_REPORT.md")
    return generate_report(runs_root=os.path.join(art, "evaluation"), output_path=out, artifacts_root=art, db_path=db, **kw), out


def _section(text, i):
    m = re.search(rf"^## {i}\. .*?$(.*?)(?=^## \d+\. |\Z)", text, re.S | re.M)
    return m.group(1).strip()


def test_all_24_sections_present_and_nonempty(tmp_path):
    art, db, _ = build_populated(tmp_path)
    text, _ = _gen(art, db, tmp_path)
    for i, name in enumerate(SECTIONS, 1):
        assert f"## {i}. {name}" in text
        assert len(_section(text, i)) > 20, name
    assert len(SECTIONS) == 24


def test_populated_report_has_no_not_run_placeholders_and_matches_db(tmp_path):
    art, db, rows = build_populated(tmp_path)
    text, out = _gen(art, db, tmp_path)
    assert MISSING not in text
    results = _section(text, 16)
    by = {(r["variant"], r["metric"]): r for r in rows}
    full = by[("SAC_FULL", "mean_pnl")]
    assert f"{fmt_num(full['mean'])} ± {fmt_num(full['std'])} (n=3)" in results
    assert fmt_num(by[("HEURISTIC", "mean_pnl")]["mean"]) in results
    assert f"{fmt_num(by[('SAC_LOB_ONLY', 'n_fills')]['mean'], 'count')} ± " in results
    # variants not run get N/A with a reason
    assert "N/A (not run)" in results and "variant not run in experiment `exp1`" in results
    for fig in ["pnl_comparison", "inventory_comparison", "adverse_selection_comparison", "learning_curves"]:
        assert f"figures/{fig}.png" in text
        assert os.path.exists(os.path.join(os.path.dirname(out), "figures", f"{fig}.png"))


def test_mandatory_text_and_forbidden_phrases(tmp_path):
    art, db, _ = build_populated(tmp_path)
    text, _ = _gen(art, db, tmp_path)
    assert FILL_DISCLAIMER in text
    assert "Mode B" in text and "FNSPID" in text and "not causally matched" in text
    assert "```mermaid" in text
    assert "sentiment encoder/classifier rather than as a generative trading agent" in text
    for ref in ["1908.10063", "1801.01290", "2402.06698", "Avellaneda", "Raffin", "Vaswani", "Fujimoto"]:
        assert ref in text
    low = text.lower()
    for phrase in FORBIDDEN:
        assert phrase.lower() not in low, phrase
    # discussion states the exploratory p-value and that differences are within noise
    disc = _section(text, 20)
    assert "p = 0.200" in disc and "within seed-to-seed noise" in disc


def test_synthetic_banner_and_mode_b_line(tmp_path):
    art, db, _ = build_populated(tmp_path, data_mode="synthetic_smoke_test")
    text, _ = _gen(art, db, tmp_path)
    assert SMOKE_BANNER in text
    assert "Data mode: Mode B — real crypto LOB" not in text
    assert FILL_DISCLAIMER in text
    assert "carry no financial meaning" in _section(text, 20)


def test_real_mode_has_no_smoke_banner(tmp_path):
    art, db, _ = build_populated(tmp_path, data_mode="mode_b")
    text, _ = _gen(art, db, tmp_path)
    assert SMOKE_BANNER not in text
    assert "Data mode: Mode B — real crypto LOB" in text


def test_empty_db_gives_honest_na_report(tmp_path):
    art = tmp_path / "artifacts"
    os.makedirs(art / "evaluation")
    engine = get_engine(f"sqlite:///{art / 'market_maker.db'}")
    init_db(engine)
    text, _ = _gen(str(art), str(art / "market_maker.db"), tmp_path)
    for i, name in enumerate(SECTIONS, 1):
        assert f"## {i}. {name}" in text and len(_section(text, i)) > 10
    assert MISSING in _section(text, 16)
    assert MISSING in _section(text, 17)
    assert FILL_DISCLAIMER in text
    assert "outperform" not in text.lower()


def test_missing_db_file_does_not_crash_and_is_not_created(tmp_path):
    art = tmp_path / "artifacts"
    os.makedirs(art / "evaluation")
    db = str(art / "nope.db")
    text, _ = _gen(str(art), db, tmp_path)
    assert not os.path.exists(db)
    assert "not found" in text and MISSING in text


def test_run_folders_only_fallback(tmp_path):
    run_dir = str(tmp_path / "run_a")
    write_run_artifacts(run_dir, config={"seed": 1}, metadata={"seed": 1, "model_path": "N/A"},
                        backtest_result={"episode_pnls": [5.0], "inventories": [], "fills": [[]], "orders": [[]]},
                        metrics={"mean_pnl": 5.0, "map": 2.0})
    out = str(tmp_path / "REPORT.md")
    text = generate_report(runs_root=str(tmp_path), output_path=out, db_path=str(tmp_path / "none.db"))
    assert "run_a" in text and "5.0000" in text and "Abstract" in text
    assert "N/A (undefined or not recorded)" in text


def test_env_vars_select_sources(tmp_path, monkeypatch):
    art, db, rows = build_populated(tmp_path)
    monkeypatch.setenv("MARKET_MAKER_DB_PATH", db)
    monkeypatch.setenv("MARKET_MAKER_ARTIFACTS_ROOT", art)
    out = str(tmp_path / "env_report.md")
    text = generate_report(runs_root=None, output_path=out)
    assert "exp1" in text and "SAC Full" in text


def test_table_cells_match_formatter():
    assert fmt_num(None) == "N/A" and fmt_num(float("nan")) == "N/A"
    assert fmt_num(1234.567) == "1234.6" and fmt_num(0.5) == "0.5000" and fmt_num(5, "count") == "5.0"
