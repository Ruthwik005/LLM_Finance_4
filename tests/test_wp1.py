import json
import os
import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from market_maker.agents.avellaneda_stoikov import AvellanedaStoikovMarketMaker
from market_maker.data.sample_data import generate_synthetic_lob
from market_maker.db.models import Step
from market_maker.db.session import get_engine, get_session, init_db
from market_maker.db.writer import write_run_to_db
from market_maker.env.market_making_env import MarketMakingEnv, PRIVATE_STATE_DIM
from market_maker.evaluation.backtest import run_backtest, write_run_artifacts
from market_maker.evaluation.metrics import summarize_run
from market_maker.agents.heuristic import HeuristicMarketMaker
from market_maker.utils.io import file_sha256

REWARD = {"type": "truncated_inventory", "inventory_threshold": 40, "inventory_penalty": 0.001,
          "cancellation_penalty": 0.0001, "scale": 1.0}
ACT = np.array([0.0, -1.0, 1.0, -1.0], dtype=np.float32)  # bid L1 20 @99.99, ask L2 20 @100.02


def _env(rows, ep_len, **kw):
    n = len(rows)
    df = pd.DataFrame({"bid_price_1": [r[0] for r in rows], "ask_price_1": [r[1] for r in rows],
                       "trade_price": [np.nan] * n, "trade_side": [None] * n, "trade_size": [0.0] * n,
                       "timestamp": pd.date_range("2024-01-01", periods=n, freq="500ms", tz="UTC")})
    args = dict(fused_features=np.zeros((n, 3), dtype=np.float32), raw_lob_df=df, feature_columns=["a", "b", "c"],
                tick_size=0.01, levels_per_side=2, min_spread_ticks=2, max_spread_ticks=8, max_mid_offset_ticks=4,
                total_volume_per_side=20.0, max_inventory=100.0, fee_per_unit=0.0, reward_config=REWARD,
                reference_price_mode="midprice", episode_length_steps=ep_len)
    args.update(kw)
    return MarketMakingEnv(**args)


FLAT = (99.99, 100.01)


def test_observation_has_private_state_and_inventory_visible():
    env = _env([FLAT] * 6, 3)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3 + PRIVATE_STATE_DIM,) and obs.dtype == np.float32
    assert env.observation_space.shape == obs.shape
    np.testing.assert_allclose(obs[3:], [0, 0, 1, 0, 0, 0])
    obs, *_ = env.step(ACT)
    assert obs[3 + 2] == pytest.approx(2 / 3)  # remaining fraction
    assert obs[3 + 3] == pytest.approx(1 / 8)  # bid distance 1 tick / max_spread_ticks
    assert obs[3 + 4] == pytest.approx(1 / 8)
    assert obs[3 + 5] == pytest.approx(2 / 8)


def test_inventory_fraction_in_observation_after_fill():
    rows = [FLAT, (99.0, 99.5), (99.0, 99.5), (99.0, 99.5), (99.0, 99.5)]
    env = _env(rows, 3)
    env.reset(seed=0)
    obs, *_ = env.step(ACT)
    assert obs[3] == pytest.approx(0.2)  # inventory 20 / 100


def test_check_env_with_private_state():
    check_env(_env(list(map(tuple, np.tile([99.99, 100.01], (10, 1)))), 4), skip_render_check=True)


def test_quote_persistence_no_cancel_when_unchanged():
    env = _env([FLAT] * 6, 4)
    env.reset(seed=0)
    _, _, _, _, i1 = env.step(ACT)
    assert i1["placed_volume"] == 40 and i1["canceled_volume"] == 0
    _, r2, _, _, i2 = env.step(ACT)
    assert i2["placed_volume"] == 0 and i2["canceled_volume"] == 0
    assert i2["reward_components"]["cancellation_penalty"] == 0.0
    assert i2["step_log"]["bid_volumes"] == [20.0, 0.0]
    # a wider spread changes both prices: cancel 40, place 40
    _, _, _, _, i3 = env.step(np.array([0.0, 1.0, 1.0, -1.0], dtype=np.float32))
    assert i3["canceled_volume"] == 40 and i3["placed_volume"] == 40
    assert i3["reward_components"]["cancellation_penalty"] == pytest.approx(0.0001 * 40)


def test_partial_fill_remainder_persists():
    rows = [FLAT, FLAT, FLAT, FLAT, FLAT]
    env = _env(rows, 3)
    env.raw_lob_df.loc[1, "trade_price"] = 99.0
    env.raw_lob_df.loc[1, "trade_side"] = "sell"
    env.raw_lob_df.loc[1, "trade_size"] = 5.0
    env.reset(seed=0)
    _, _, _, _, i1 = env.step(ACT)
    assert i1["inventory"] == 5.0
    _, _, _, _, i2 = env.step(ACT)
    assert i2["step_log"]["bid_volumes"][0] == pytest.approx(15.0)  # remainder persists, not re-placed
    assert i2["placed_volume"] == 0 and i2["canceled_volume"] == 0


def test_latency_delays_fill_eligibility():
    # later rows keep mid=100 (so quotes are unchanged) but the ask touches our bid -> fill evidence
    rows = [FLAT, (100.01, 99.99), (100.01, 99.99), (100.01, 99.99), (100.01, 99.99)]
    e0 = _env(rows, 3, latency_steps=0)
    e0.reset(seed=0)
    assert e0.step(ACT)[4]["inventory"] == 20.0
    e1 = _env(rows, 3, latency_steps=1)
    e1.reset(seed=0)
    assert e1.step(ACT)[4]["inventory"] == 0.0
    assert e1.step(ACT)[4]["inventory"] == 20.0  # eligible one step later, same resting order


def test_fees_and_rebates_tracked_separately():
    rows = [FLAT, (99.0, 99.5), (99.0, 99.5), (99.0, 99.5), (99.0, 99.5)]
    env = _env(rows, 3, fee_per_unit=0.01, rebate_per_unit=0.02)
    env.reset(seed=0)
    _, _, _, _, info = env.step(ACT)
    assert info["fees_paid"] == pytest.approx(0.2)
    assert info["rebates_received"] == pytest.approx(0.4)
    assert info["cash"] == pytest.approx(-99.99 * 20 - 0.2 + 0.4)


def test_step_log_fields():
    rows = [FLAT, (99.0, 99.5), (99.0, 99.5), (99.0, 99.5), (99.0, 99.5)]
    env = _env(rows, 3)
    env.reset(seed=0)
    log = env.step(ACT)[4]["step_log"]
    assert log["t"] == 0 and log["timestamp"].startswith("2024-01-01T00:00:00")
    assert log["mid"] == pytest.approx(100.0) and log["next_mid"] == pytest.approx(99.25)
    assert log["bid_prices"] == pytest.approx([99.99, 99.98]) and log["ask_prices"] == pytest.approx([100.01, 100.02])
    assert log["ask_volumes"] == [0.0, 20.0]
    assert log["fills"][0]["side"] == "bid" and log["inventory"] == 20.0
    assert log["placed_volume"] == 40.0


# ---------------- metrics (hand-computed) ----------------
def _step(t, fills, placed, canceled, mtm):
    return {"t": t, "timestamp": None, "mid": 100.0, "next_mid": 100.0 if t == 0 else 99.9,
            "bid_prices": [99.98, 99.97], "ask_prices": [100.02, 100.03], "bid_volumes": [10.0, 0.0],
            "ask_volumes": [10.0, 0.0], "inventory": 1.0, "cash": 0.0, "mtm_pnl": mtm, "fills": fills,
            "placed_volume": placed, "canceled_volume": canceled}


CFG = {"data": {"tick_size": 0.01}, "simulator": {"fee_per_unit": 0.01, "rebate_per_unit": 0.0},
       "evaluation": {"adverse_horizon_steps": 2, "adverse_threshold_ticks": 1.0}}


def _result(pnls=(0.5,)):
    fills = [{"side": "bid", "price": 99.98, "volume": 2.0, "mid": 100.0},
             {"side": "ask", "price": 100.02, "volume": 1.0, "mid": 100.0}]
    steps = [[_step(0, fills, 20.0, 0.0, 0.04), _step(1, [], 0.0, 10.0, 0.1), _step(2, [], 0.0, 0.0, 0.0)]]
    return {"episode_pnls": list(pnls), "inventories": [np.array([2.0, -3.0, 1.0])], "fills": [[]], "orders": [[]],
            "steps": steps}


def test_summarize_run_hand_computed():
    m = summarize_run(_result(), CFG)
    assert m["n_fills"] == 2
    assert m["spread_capture_pnl"] == pytest.approx(0.06)       # 2*0.02 + 1*0.02
    assert m["fees_paid"] == pytest.approx(0.03)                # 0.01 * 3 units
    assert m["inventory_pnl"] == pytest.approx(0.5 - 0.06 + 0.03)
    assert m["cancel_ratio"] == pytest.approx(10 / 20)
    assert m["fill_rate"] == pytest.approx(3 / 20)
    assert m["avg_quoted_spread"] == pytest.approx(4.0)         # (100.02-99.98)/0.01
    assert m["avg_quote_distance"] == pytest.approx(2.0)
    assert m["max_drawdown"] == pytest.approx(0.1)              # curve [0,.04,.1,0]
    assert m["max_abs_inventory"] == 3.0
    assert m["map"] == pytest.approx(2.0)
    assert m["pnl_over_map"] == pytest.approx(0.25)
    # buy: 99.9-100=-0.1 < -0.01 adverse; sell: -0.1 not > 0.01 -> 1 of 2
    assert m["adverse_selection_ratio"] == pytest.approx(0.5)
    assert "sharpe_like" not in m  # single episode


def test_sharpe_like_and_omission_when_std_zero():
    assert summarize_run(_result((1.0, 3.0)), CFG)["sharpe_like"] == pytest.approx(2.0)
    assert "sharpe_like" not in summarize_run(_result((2.0, 2.0)), CFG)


def test_zero_fill_and_zero_map_edge_cases():
    r = _result()
    for s in r["steps"][0]:
        s["fills"] = []
    r["inventories"] = [np.zeros(3)]
    m = summarize_run(r, CFG)
    assert m["n_fills"] == 0 and m["spread_capture_pnl"] == 0 and m["fees_paid"] == 0
    assert m["inventory_pnl"] == pytest.approx(0.5)
    assert "adverse_selection_ratio" not in m and "pnl_over_map" not in m
    assert all(np.isfinite(v) for v in m.values())


# ---------------- Avellaneda-Stoikov ----------------
def _as(**kw):
    a = dict(gamma=0.1, kappa=1.5, rolling_vol_index=1, midprice_index=0, tick_size=0.01, min_spread_ticks=1,
             max_spread_ticks=8, max_mid_offset_ticks=4, episode_length_steps=100)
    a.update(kw)
    return AvellanedaStoikovMarketMaker(**a)


def _obs(vol=0.0005, remaining=1.0):
    return np.array([100.0, vol] + [0, 0, remaining, 0, 0, 0], dtype=np.float32)


def test_as_valid_actions_and_skews_against_inventory():
    agent = _as(gamma=1.0)
    flat = agent.predict(_obs(), inventory=0.0, max_inventory=100.0)
    long_ = agent.predict(_obs(), inventory=50.0, max_inventory=100.0)
    short = agent.predict(_obs(), inventory=-50.0, max_inventory=100.0)
    for a in (flat, long_, short):
        assert a.shape == (4,) and np.all(np.abs(a) <= 1.0)
    assert flat[0] == 0.0
    assert long_[0] < 0 < short[0]  # long -> quote lower to sell
    assert long_[1] >= flat[1]       # spread grows with risk term


def test_as_no_skew_at_horizon_end():
    assert _as(gamma=1.0).predict(_obs(remaining=0.0), 50.0, 100.0)[0] == 0.0


# ---------------- artifacts / DB ----------------
def _full_run():
    lob = generate_synthetic_lob(n_rows=500, depth=2, symbol="BTCUSDT", seed=5)
    env = MarketMakingEnv(fused_features=np.random.randn(len(lob), 4).astype(np.float32), raw_lob_df=lob,
                          feature_columns=list("abcd"), tick_size=0.01, levels_per_side=2, min_spread_ticks=1,
                          max_spread_ticks=8, max_mid_offset_ticks=4, total_volume_per_side=20.0,
                          max_inventory=100.0, fee_per_unit=0.0, reward_config=REWARD,
                          reference_price_mode="midprice", episode_length_steps=20)
    return run_backtest(env, HeuristicMarketMaker(), 2, True, 100.0)


def test_backtest_has_steps_and_artifacts_written(tmp_path):
    res = _full_run()
    assert len(res["steps"]) == 2 and len(res["steps"][0]) == 20
    m = summarize_run(res, CFG)
    d = str(tmp_path / "r")
    write_run_artifacts(d, {"a": 1}, {"seed": 1, "git_commit": None}, res, m)
    for f in ["orders.csv", "quotes.csv", "training_curve.csv"]:
        assert os.path.getsize(os.path.join(d, f)) > 0
    assert len(open(os.path.join(d, "quotes.csv")).readlines()) == 1 + 40
    assert len(open(os.path.join(d, "training_curve.csv")).readlines()) == 1  # header only
    for p in ["equity_curve", "episode_pnl", "inventory", "adverse_selection"]:
        assert os.path.getsize(os.path.join(d, "plots", p + ".png")) > 0


def test_training_curve_written_when_supplied(tmp_path):
    d = str(tmp_path / "r")
    write_run_artifacts(d, {}, {}, _full_run(), {}, training_curve=[(100, 1.5), (200, 2.5)])
    assert len(open(os.path.join(d, "training_curve.csv")).readlines()) == 3


def test_file_sha256(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert file_sha256(str(p)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert file_sha256(str(tmp_path / "missing")) is None


def test_db_steps_rows_replace_on_rerun(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path}/t.db")
    init_db(engine)
    s = get_session(engine)
    res = _full_run()
    for _ in range(2):
        write_run_to_db(s, "r1", "HEURISTIC", 1, "N/A", {}, res, {"mean_pnl": 1.0})
    assert s.query(Step).filter_by(run_id="r1").count() == 40
    row = s.query(Step).filter_by(run_id="r1", episode=1, step=3).one()
    assert row.bid_1 > 0 and row.timestamp is not None
