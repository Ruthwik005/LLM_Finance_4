import json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from market_maker.api import schemas as S
from market_maker.db.session import get_engine, get_session, init_db
from market_maker.db.writer import write_run_to_db, write_experiment_aggregates
import market_maker.api.main as api_main


def _step(t, mid, inv, mtm, fills=()):
    return {"t": t, "timestamp": f"2024-01-01T00:00:{t % 60:02d}+00:00", "mid": mid,
            "bid_prices": [mid - 1, mid - 2], "ask_prices": [mid + 1, mid + 2],
            "bid_volumes": [1.0, 1.0], "ask_volumes": [1.0, 1.0], "inventory": inv, "cash": 0.0, "mtm_pnl": mtm,
            "placed_volume": 2.0, "canceled_volume": 1.0, "fills": list(fills)}


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = get_engine(f"sqlite:///{tmp_path}/t.db")
    init_db(engine)
    monkeypatch.setattr(api_main, "ENGINE", engine)
    monkeypatch.setenv("MARKET_MAKER_ARTIFACTS_ROOT", str(tmp_path / "art"))
    s = get_session(engine)
    n = 500
    steps = [_step(t, 100 + 0.01 * t, 0.0, 0.1 * t) for t in range(n)]
    fills = [{"side": "bid", "price": 100.5, "volume": 2.0, "step": 7}, {"side": "ask", "price": 101.0, "volume": 1.0, "step": 333},
             {"side": "bid", "price": 102.0, "volume": 1.0, "step": 499}]
    write_run_to_db(s, "run_a", "SAC_LOB_ONLY", 0, "m.zip", {}, {
        "episode_pnls": [1.0, 2.0], "inventories": [], "fills": [fills, []], "orders": [[], []], "steps": [steps, []]},
        {"mean_pnl": 1.5, "map": 3.0, "max_drawdown": 0.5, "n_fills": 3, "spread_capture_pnl": -1.0, "inventory_pnl": 2.5,
         "fees_paid": 0.1, "rebates_received": 0.0, "total_pnl": 3.0})
    write_run_to_db(s, "run_old", "HEURISTIC", 0, "N/A", {}, {"episode_pnls": [0.5], "inventories": [],
                    "fills": [[{"side": "bid", "price": 1.0, "volume": 1.0}]], "orders": [[]]}, {"mean_pnl": 0.5})
    s.close()
    return tmp_path, TestClient(api_main.app)


def test_existing_shapes_validate(env):
    _, c = env
    S.RunList(**c.get("/runs").json())
    d = S.RunDetail(**c.get("/runs/run_a").json())
    assert d.variant == "SAC_LOB_ONLY"
    assert c.get("/runs/run_a/metrics").json()["mean_pnl"] == 1.5
    assert c.get("/runs/run_a/fills").json()[0].keys() == {"episode", "side", "price", "volume"}


def test_runs_summary(env):
    _, c = env
    r = c.get("/runs/summary").json()
    S.RunsSummary(**r)
    row = {x["run_id"]: x for x in r["runs"]}
    assert row["run_a"]["mean_pnl"] == 1.5 and row["run_a"]["map"] == 3.0 and row["run_a"]["n_fills"] == 3
    assert row["run_old"]["map"] is None


def test_replay_downsamples_but_keeps_fills(env):
    _, c = env
    r = c.get("/runs/run_a/episodes/0/replay?max_points=40").json()
    S.Replay(**r)
    assert r["n_steps_total"] == 500 and r["downsampled"] and r["n_points"] < 80
    steps = [s["step"] for s in r["steps"]]
    assert steps == sorted(steps)
    assert {7, 333, 499} <= set(steps)
    assert {f["step"] for f in r["fills"]} == {7, 333, 499}
    assert r["fills"][0]["timestamp"] is not None
    sm = r["summary"]  # computed over all 500 steps, not the sampled subset
    assert sm["n_fills"] == 3 and sm["filled_volume"] == 4.0 and sm["cancel_volume"] == 500.0
    assert sm["avg_quoted_spread"] == pytest.approx(2.0) and sm["final_mtm_pnl"] == pytest.approx(49.9)
    full = c.get("/runs/run_a/episodes/0/replay?max_points=5000").json()
    assert not full["downsampled"] and full["n_points"] == 500


def test_replay_404_and_empty(env):
    _, c = env
    assert c.get("/runs/nope/episodes/0/replay").status_code == 404
    assert c.get("/runs/run_a/episodes/9/replay").status_code == 404
    old = c.get("/runs/run_old/episodes/0/replay").json()  # old run: episode exists, no steps
    assert old["summary"]["final_mtm_pnl"] is None and old["steps"] == [] and old["n_points"] == 0 and len(old["fills"]) == 1 and old["fills"][0]["step"] is None
    assert c.get("/runs/run_a/episodes/1/replay").json()["steps"] == []


def test_training_curve_new_and_old(env):
    root, c = env
    d = root / "art" / "evaluation"
    (d / "run_a").mkdir(parents=True)
    (d / "run_a" / "training_curve.csv").write_text(
        "timesteps,train_reward_mean,val_reward_mean,val_pnl_mean,val_map,ent_coef,critic_loss,actor_loss,is_best\n"
        "300,nan,-1.5,0.07,0.29,0.5,1.0,-2.0,1\n600,-1.0,-1.4,0.1,0.3,nan,nan,nan,0\n")
    r = c.get("/runs/run_a/training-curve").json()
    S.TrainingCurve(**r)
    p = r["points"]
    assert len(p) == 2 and p[0]["is_best"] is True and p[0]["train_reward_mean"] is None and p[1]["ent_coef"] is None
    (d / "run_old").mkdir()
    (d / "run_old" / "training_curve.csv").write_text("step,reward\n100,0.5\n200,0.7\n")
    p = c.get("/runs/run_old/training-curve").json()["points"]
    assert [x["timesteps"] for x in p] == [100.0, 200.0] and p[1]["train_reward_mean"] == 0.7 and p[0]["val_reward_mean"] is None
    assert c.get("/runs/run_a/training-curve").status_code == 200
    assert c.get("/runs/zzz/training-curve").status_code == 404


def test_training_curve_missing_file_is_empty(env):
    _, c = env
    assert c.get("/runs/run_old/training-curve").json()["points"] == []


def test_experiments(env):
    root, c = env
    assert c.get("/experiments").json()["experiments"] == []
    assert c.get("/experiments/nope").status_code == 404
    s = get_session(api_main.ENGINE)
    write_experiment_aggregates(s, "exp1", [
        {"variant": "SAC_LOB_ONLY", "metric": "mean_pnl", "mean": 1.0, "std": 0.5, "n_seeds": 3},
        {"variant": "SAC_FULL", "metric": "mean_pnl", "mean": 2.0, "std": None, "n_seeds": 1}])
    s.close()
    ev = root / "art" / "evaluation"
    ev.mkdir(parents=True)
    (ev / "exp1_summary.json").write_text(json.dumps({
        "experiment_id": "exp1", "data_mode": "synthetic_smoke_test", "seeds": [0, 1, 2],
        "comparisons_vs_SAC_LOB_ONLY": {"SAC_FULL": {"mean_pnl": {"n_a": 3, "n_b": 3, "mean_diff": 1.0, "t": NAN, "p_value": 0.4, "cohens_d": 0.7, "note": "x"}}},
        "comparison_note": "Exploratory comparison with a small number of seeds"}))
    lst = c.get("/experiments").json()["experiments"]
    assert lst[0]["experiment_id"] == "exp1" and lst[0]["has_summary_file"]
    r = c.get("/experiments/exp1").json()
    S.ExperimentDetail(**r)
    v = {x["variant"]: x for x in r["variants"]}
    assert v["SAC_FULL"]["metrics"]["mean_pnl"]["std"] is None and v["SAC_LOB_ONLY"]["n_seeds"] == 3
    assert r["comparisons"][0]["t"] is None and r["comparisons"][0]["p_value"] == 0.4
    assert "xploratory" in r["comparison_note"] and r["comparison_label"] == "exploratory, small n"


NAN = float("nan")


def test_pnl_attribution(env):
    _, c = env
    r = c.get("/runs/run_a/pnl-attribution").json()
    S.PnlAttribution(**r)
    assert r["spread_capture_pnl"] == -1.0 and r["inventory_pnl"] == 2.5 and r["total"] == 3.0 and r["rebates"] == 0.0
    o = c.get("/runs/run_old/pnl-attribution").json()
    assert o["spread_capture_pnl"] is None and o["total"] is None
    assert c.get("/runs/zzz/pnl-attribution").status_code == 404


def test_signals_unavailable_then_available(env):
    root, c = env
    r = c.get("/signals").json()
    assert r["available"] is False and "reason" in r
    f = root / "art" / "features"
    f.mkdir(parents=True)
    n = 300
    ts = pd.date_range("2024-01-01", periods=n, freq="s", tz="UTC")
    pd.DataFrame({"timestamp": ts, "bid_price_1": 99.0, "ask_price_1": 101.0}).to_parquet(f / "lob_test.parquet")
    cols = ["midprice", "sentiment_ewm_5", "article_count_5", "transformer_prob_h1_down", "transformer_prob_h1_flat",
            "transformer_prob_h1_up", "transformer_signal_valid"]
    arr = np.zeros((n, len(cols)), dtype=np.float32)
    arr[:, 1] = 0.5
    np.save(f / "sac_full_test.npy", arr)
    (f / "sac_full_columns.json").write_text(json.dumps(cols))
    r = c.get("/signals?split=test&max_points=50").json()
    S.SignalsPayload(**r)
    assert r["available"] and r["n_rows_total"] == n and r["n_points"] <= 60
    assert len(r["timestamps"]) == r["n_points"] and set(r["series"]) >= {"sentiment_ewm_5", "article_count_5", "transformer_prob_h1_up"}
    assert all(len(v) == r["n_points"] for v in r["series"].values())
    assert r["midprice"][0] == 100.0
    assert "midprice" not in r["series"]
    assert c.get("/signals?split=bogus").status_code == 422
    assert c.get("/signals?split=train").json()["available"] is False
