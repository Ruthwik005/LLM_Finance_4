import json
import os
import numpy as np
import pandas as pd
import pytest

from market_maker.db.models import ExperimentAggregate, Metric, Run
from market_maker.db.session import get_engine, get_session
from market_maker.training.env_factory import evenly_spaced_starts, make_env
from market_maker.training.experiment import (aggregate_runs, compare_to_reference, mean_std, run_experiment,
                                               welch_compare)
from market_maker.training.train_sac import BestCheckpointTracker, ValidationCheckpointCallback, train_sac_agent


def test_env_reset_start_index():
    from test_wp1 import _env  # noqa  (reuse helper env with deterministic rows)
    env = _env([(99.99, 100.01)] * 50, 10)
    env.reset(options={"start_index": 7})
    assert env._data_idx == 7
    env.reset(options={"start_index": env._max_start})
    with pytest.raises(ValueError):
        env.reset(options={"start_index": env._max_start + 1})
    with pytest.raises(ValueError):
        env.reset(options={"start_index": -1})
    env.reset()  # still random without options
    assert 0 <= env._data_idx <= env._max_start


def test_evenly_spaced_starts_no_overlap_and_bounds():
    s = evenly_spaced_starts(10_000, 100, 5)
    assert s[0] == 0 and s[-1] == 10_000 - 100 - 1 and len(s) == 5
    assert all(b - a >= 100 for a, b in zip(s, s[1:]))
    assert evenly_spaced_starts(10_000, 100, 5) == s  # deterministic
    with pytest.raises(ValueError):
        evenly_spaced_starts(50, 100, 3)
    assert evenly_spaced_starts(500, 100, 1) == [0]


def test_best_checkpoint_tracker_and_callback_with_mocked_eval(tmp_path):
    t = BestCheckpointTracker()
    assert [t.update(s, i) for i, s in enumerate([1.0, 0.5, 2.0, 2.0, float("nan"), 3.0])] == \
        [True, False, True, False, False, True]
    assert t.best_score == 3.0 and t.best_step == 5

    scores = iter([1.0, 3.0, 2.0])
    saves = []

    class FakeModel:
        ep_info_buffer = [{"r": 1.0}, {"r": 3.0}]

        def save(self, path):
            saves.append(os.path.basename(path))

    cb = ValidationCheckpointCallback(lambda m: {"val_reward_mean": next(scores), "val_pnl_mean": 0.0, "val_map": 1.0},
                                      eval_freq=100, out_dir=str(tmp_path))
    cb.model = FakeModel()
    for step in (100, 200, 300):
        cb.num_timesteps = step
        cb._on_step()
    cb._on_training_end()  # last eval already at 300 -> no extra eval
    assert saves == ["best_model", "best_model"]  # saved at steps 100 and 200 only
    assert cb.tracker.best_step == 200 and [r["is_best"] for r in cb.rows] == [1, 1, 0]
    assert cb.rows[0]["train_reward_mean"] == 2.0
    rows = pd.read_csv(tmp_path / "training_curve.csv")
    assert list(rows["timesteps"]) == [100, 200, 300] and "val_pnl_mean" in rows.columns


def test_welch_and_aggregation():
    a, b = [1.0, 2.0, 3.0, 4.0], [0.0, 0.5, 1.0, 1.5]
    r = welch_compare(a, b)
    from scipy import stats
    assert r["p_value"] == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue)
    assert r["cohens_d"] > 0 and "not a claim of significance" in r["note"]
    assert welch_compare([1.0], b)["p_value"] is None
    assert mean_std([2.0, 4.0]) == (3.0, pytest.approx(np.sqrt(2)))
    assert mean_std([2.0])[1] is None
    agg = aggregate_runs({"SAC_LOB_ONLY": [{"mean_pnl": 1.0}, {"mean_pnl": 3.0}],
                          "SAC_FULL": [{"mean_pnl": 2.0, "sharpe_like": 1.0}, {"mean_pnl": 6.0}]})
    assert agg["SAC_LOB_ONLY"]["mean_pnl"]["std"] == pytest.approx(np.sqrt(2))
    assert agg["SAC_FULL"]["sharpe_like"]["n_seeds"] == 1
    cmp = compare_to_reference(agg)
    assert "SAC_FULL" in cmp and "mean_pnl" in cmp["SAC_FULL"]
    assert "significant" not in json.dumps(cmp).replace("not a claim of significance", "")


def test_make_env_train_and_eval_modes(wp2_built):
    cfg, art = wp2_built
    env = make_env(cfg, "SAC_FULL", "train", art, "train", seed=3)
    obs, _ = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]
    starts_a = [env.unwrapped._data_idx]
    for _ in range(3):
        env.reset()
        starts_a.append(env.unwrapped._data_idx)
    env2 = make_env(cfg, "SAC_FULL", "train", art, "train", seed=3)
    env2.reset()
    starts_b = [env2.unwrapped._data_idx]
    for _ in range(3):
        env2.reset()
        starts_b.append(env2.unwrapped._data_idx)
    assert starts_a == starts_b  # seeded, reproducible training windows


def test_sac_training_deterministic_and_saves_best(wp2_copy, tmp_path):
    cfg, art = wp2_copy
    outs = []
    for k in range(2):
        tr_env = make_env(cfg, "SAC_LOB_ONLY", "train", art, "train", seed=5)
        val_env = make_env(cfg, "SAC_LOB_ONLY", "validation", art, "eval")
        starts = evenly_spaced_starts(len(val_env.raw_lob_df), cfg.episode_steps, 2)
        res = train_sac_agent(tr_env, val_env, cfg.sac, seed=5, out_dir=str(tmp_path / f"m{k}"), val_starts=starts)
        outs.append(res)
    assert os.path.exists(outs[0].best_model_path) and os.path.exists(outs[0].last_model_path)
    assert [r["val_reward_mean"] for r in outs[0].curve] == [r["val_reward_mean"] for r in outs[1].curve]
    assert len(outs[0].curve) == 2  # eval at 150 and 300
    c = pd.read_csv(outs[0].curve_path)
    assert {"timesteps", "train_reward_mean", "val_reward_mean", "val_pnl_mean", "val_map"} <= set(c.columns)


def test_run_experiment_end_to_end(wp2_copy, tmp_path):
    cfg, art = wp2_copy
    db = str(tmp_path / "t.db")
    summary = run_experiment(cfg, artifacts_dir=art, db_path=db, variants=["SAC_LOB_ONLY", "SAC_FULL"], seeds=[0, 1],
                             experiment_id="smoke", fast=True, data_mode="synthetic_smoke_test")
    ev = os.path.join(art, "evaluation")
    for rid in ["sac_lob_only_seed0", "sac_lob_only_seed1", "sac_full_seed0", "sac_full_seed1", "heuristic",
                "avellaneda_stoikov"]:
        assert os.path.exists(os.path.join(ev, rid, "metrics.json")), rid
    assert os.path.exists(os.path.join(art, "rl", "sac_full", "seed1", "best_model.zip"))
    md = json.load(open(os.path.join(ev, "sac_full_seed1", "metadata.json")))
    assert md["seed"] == 1 and md["variant"] == "SAC_FULL" and md["data_mode"] == "synthetic_smoke_test"
    curve = pd.read_csv(os.path.join(ev, "sac_full_seed1", "training_curve.csv"))
    assert "val_reward_mean" in curve.columns
    assert os.path.exists(os.path.join(ev, "smoke_summary.json")) and os.path.exists(os.path.join(ev, "smoke_summary.csv"))
    s = summary["aggregates"]["SAC_FULL"]["mean_pnl"]
    assert s["n_seeds"] == 2 and s["std"] is not None
    assert summary["aggregates"]["HEURISTIC"]["mean_pnl"]["std"] is None
    assert "comparisons_vs_SAC_LOB_ONLY" in summary and summary["comparisons_vs_SAC_LOB_ONLY"]["SAC_FULL"]
    # same fixed test windows for every agent
    for rid in ["heuristic", "sac_full_seed0"]:
        assert json.load(open(os.path.join(ev, rid, "metadata.json")))["test_windows"] == summary["test_windows"]

    session = get_session(get_engine(f"sqlite:///{db}"))
    assert {r.run_id for r in session.query(Run).all()} >= {"sac_full_seed0", "heuristic", "avellaneda_stoikov"}
    rows = session.query(ExperimentAggregate).filter_by(experiment_id="smoke").all()
    n1 = len(rows)
    assert n1 > 0 and {r.variant for r in rows} >= {"SAC_LOB_ONLY", "SAC_FULL", "HEURISTIC"}
    session.close()
    # resume skips completed runs and re-writes aggregates (replace, not append)
    mtime = os.path.getmtime(os.path.join(art, "rl", "sac_full", "seed1", "best_model.zip"))
    run_experiment(cfg, artifacts_dir=art, db_path=db, variants=["SAC_LOB_ONLY", "SAC_FULL"], seeds=[0, 1],
                   experiment_id="smoke", fast=True, resume=True, data_mode="synthetic_smoke_test")
    assert os.path.getmtime(os.path.join(art, "rl", "sac_full", "seed1", "best_model.zip")) == mtime
    session = get_session(get_engine(f"sqlite:///{db}"))
    assert session.query(ExperimentAggregate).filter_by(experiment_id="smoke").count() == n1
    session.close()
