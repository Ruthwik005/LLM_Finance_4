"""Multi-variant x multi-seed experiment runner with aggregation.

For every (variant, seed): train SAC (train split, validation-based best checkpoint) -> evaluate `best_model.zip`
on fixed TEST windows -> run artifacts + DB rows. HEURISTIC and AVELLANEDA_STOIKOV are evaluated once on the
same test windows. Per variant the key metrics are aggregated across seeds (mean +- std, ddof=1) into
artifacts/evaluation/<experiment_id>_summary.{json,csv} and the DB table `experiment_aggregates`.
"""
from __future__ import annotations
import csv
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from market_maker.agents.avellaneda_stoikov import AvellanedaStoikovMarketMaker
from market_maker.agents.heuristic import HeuristicMarketMaker
from market_maker.db.session import get_engine, get_session, init_db
from market_maker.db.writer import write_experiment_aggregates, write_run_to_db
from market_maker.evaluation.backtest import run_backtest, write_run_artifacts
from market_maker.evaluation.metrics import summarize_run
from market_maker.evaluation.run_metadata import build_run_metadata
from market_maker.features.pipeline import SPLITS, ensure_features, load_meta, lob_split_path
from market_maker.training.env_factory import evenly_spaced_starts, make_env
from market_maker.training.train_sac import train_sac_agent
from market_maker.utils.logging import get_logger

logger = get_logger(__name__)

BASELINES = ["HEURISTIC", "AVELLANEDA_STOIKOV"]
REFERENCE_VARIANT = "SAC_LOB_ONLY"
COMPARE_METRICS = ["mean_pnl", "pnl_over_map", "sharpe_like", "map", "max_drawdown"]
EXPLORATORY_NOTE = ("Exploratory comparison with a small number of seeds; Welch p-values and effect sizes are "
                    "descriptive only and are not a claim of significance.")
FAST_BUDGET = dict(total_timesteps=600, eval_freq=300, learning_starts=100, buffer_size=2000, batch_size=64)


def infer_data_mode(config_path: str) -> str:
    return "synthetic_smoke_test" if os.path.basename(config_path or "") == "fast_demo.yaml" else "mode_b"


def run_id_for(variant: str, seed: int | None = None) -> str:
    return f"{variant.lower()}_seed{seed}" if seed is not None else variant.lower()


# ---------------------------------------------------------------------------
# Aggregation / statistics (pure functions)
# ---------------------------------------------------------------------------
def mean_std(values: list[float]) -> tuple[float, float | None]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), (float(arr.std(ddof=1)) if len(arr) > 1 else None)


def aggregate_runs(per_variant: dict[str, list[dict]]) -> dict[str, dict[str, dict]]:
    """{variant: [metrics per seed]} -> {variant: {metric: {mean, std, n_seeds, values}}} (metrics may be missing
    in some seeds; n_seeds counts the seeds that defined the metric)."""
    out: dict[str, dict[str, dict]] = {}
    for variant, runs in per_variant.items():
        keys = sorted({k for m in runs for k, v in m.items() if isinstance(v, (int, float)) and np.isfinite(v)})
        out[variant] = {}
        for k in keys:
            vals = [float(m[k]) for m in runs if k in m and m[k] is not None and np.isfinite(m[k])]
            mean, std = mean_std(vals)
            out[variant][k] = {"mean": mean, "std": std, "n_seeds": len(vals), "values": vals}
    return out


def welch_compare(a: list[float], b: list[float]) -> dict:
    """Welch's t-test (unequal variances) of a vs b + Cohen's d. Needs >= 2 values per group, else p/d are None."""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    res = {"n_a": int(len(a)), "n_b": int(len(b)),
           "mean_diff": float(a.mean() - b.mean()) if len(a) and len(b) else None,
           "t": None, "p_value": None, "cohens_d": None, "note": EXPLORATORY_NOTE}
    if len(a) < 2 or len(b) < 2:
        return res
    t, p = stats.ttest_ind(a, b, equal_var=False)
    res["t"] = None if not np.isfinite(t) else float(t)
    res["p_value"] = None if not np.isfinite(p) else float(p)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    res["cohens_d"] = float((a.mean() - b.mean()) / sp) if sp > 0 else None
    return res


def compare_to_reference(agg: dict, reference: str = REFERENCE_VARIANT) -> dict:
    out: dict = {}
    if reference not in agg:
        return out
    for variant, metrics in agg.items():
        if variant == reference or variant in BASELINES:
            continue
        out[variant] = {m: welch_compare(metrics[m]["values"], agg[reference][m]["values"])
                        for m in COMPARE_METRICS if m in metrics and m in agg[reference]}
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def plan_runs(cfg, variants, seeds, fast: bool = False) -> dict:
    sac = apply_fast(cfg.sac) if fast else cfg.sac
    n_val = _n_val(cfg, fast)
    steps_per_eval_ep = cfg.episode_steps
    n_evals = int(np.ceil(sac.total_timesteps / sac.eval_freq)) + 1
    runs = [{"run_id": run_id_for(v, s), "variant": v, "seed": s} for v in variants for s in seeds]
    train_steps = len(runs) * sac.total_timesteps
    eval_steps = len(runs) * n_evals * n_val * steps_per_eval_ep
    return {"runs": runs, "total_train_timesteps": train_steps, "total_validation_env_steps": eval_steps,
            "total_test_env_steps": (len(runs) + 2) * (_n_test(cfg, fast)) * steps_per_eval_ep,
            "baselines": BASELINES, "sac_total_timesteps": sac.total_timesteps, "eval_freq": sac.eval_freq}


def apply_fast(sac_cfg):
    return sac_cfg.model_copy(update=FAST_BUDGET)


def _n_val(cfg, fast):
    if fast:
        return 2
    return cfg.experiment.n_val_episodes or cfg.sac.n_eval_episodes


def _n_test(cfg, fast):
    return 3 if fast else cfg.experiment.n_test_episodes


def _evaluate_agent(env, agent, is_heuristic, cfg, starts):
    result = run_backtest(env, agent, n_episodes=len(starts), is_heuristic=is_heuristic,
                          max_inventory=cfg.simulator.max_inventory, start_indices=starts)
    return result, summarize_run(result, cfg)


def run_experiment(cfg, *, artifacts_dir: str = "artifacts", db_path: str | None = None, variants=None, seeds=None,
                   experiment_id: str | None = None, fast: bool = False, resume: bool = False,
                   data_mode: str | None = None, config_path: str = "") -> dict:
    variants = [v.upper() for v in (variants or cfg.experiment.variants)]
    seeds = list(seeds if seeds is not None else cfg.experiment.seeds)
    exp_id = experiment_id or cfg.experiment_id
    data_mode = data_mode or infer_data_mode(config_path)
    sac_cfg = apply_fast(cfg.sac) if fast else cfg.sac
    n_val, n_test = _n_val(cfg, fast), _n_test(cfg, fast)
    L = cfg.episode_steps
    eval_root = os.path.join(artifacts_dir, "evaluation")
    db_path = db_path or os.environ.get("MARKET_MAKER_DB_PATH") or os.path.join(artifacts_dir, "market_maker.db")
    engine = get_engine(f"sqlite:///{db_path}")
    init_db(engine)

    for v in variants:
        ensure_features(cfg, artifacts_dir, v)
    ensure_features(cfg, artifacts_dir, REFERENCE_VARIANT)  # baselines evaluate on the LOB-only block
    meta = load_meta(artifacts_dir)
    split_frames = {s: pd.read_parquet(lob_split_path(artifacts_dir, s), columns=["timestamp"]) for s in SPLITS}

    test_env = make_env(cfg, REFERENCE_VARIANT, "test", artifacts_dir, "eval")
    test_starts = evenly_spaced_starts(len(test_env.raw_lob_df), L, n_test)
    val_probe = make_env(cfg, REFERENCE_VARIANT, "validation", artifacts_dir, "eval")
    val_starts = evenly_spaced_starts(len(val_probe.raw_lob_df), L, n_val)
    del val_probe

    session = get_session(engine)

    def _record(run_id, variant, seed, model_path, result, metrics, extra_meta, curve_csv=None):
        run_dir = os.path.join(eval_root, run_id)
        md = build_run_metadata(cfg, data_mode=data_mode, model_path=model_path, variant=variant, splits=split_frames)
        md.update({"seed": seed, "experiment_id": exp_id, "test_windows": test_starts, "val_windows": val_starts,
                   "fast_mode": fast, **extra_meta})
        write_run_artifacts(run_dir, config=cfg.model_dump(), metadata=md, backtest_result=result, metrics=metrics)
        if curve_csv and os.path.exists(curve_csv):
            with open(curve_csv) as src, open(os.path.join(run_dir, "training_curve.csv"), "w") as dst:
                dst.write(src.read())
        write_run_to_db(session, run_id=run_id, variant=variant, seed=seed, model_path=model_path,
                        config=cfg.model_dump(), backtest_result=result, metrics=metrics)

    def _completed(run_id):
        return os.path.exists(os.path.join(eval_root, run_id, "metrics.json")) and \
            os.path.exists(os.path.join(eval_root, run_id, "metadata.json"))

    def _load_metrics(run_id):
        with open(os.path.join(eval_root, run_id, "metrics.json")) as f:
            return json.load(f)

    timings: dict[str, dict] = {}
    baseline_metrics: dict[str, dict] = {}
    for b in BASELINES:
        rid = run_id_for(b)
        if resume and _completed(rid):
            baseline_metrics[b] = _load_metrics(rid)
            logger.info("resume: skipping completed baseline %s", rid)
            continue
        if b == "HEURISTIC":
            agent = HeuristicMarketMaker()
        else:
            agent = AvellanedaStoikovMarketMaker(
                gamma=cfg.baselines.gamma, kappa=cfg.baselines.kappa, rolling_vol_index=meta["rolling_vol_index"],
                midprice_index=meta["midprice_index"], tick_size=cfg.data.tick_size,
                min_spread_ticks=cfg.simulator.min_spread_ticks, max_spread_ticks=cfg.simulator.max_spread_ticks,
                max_mid_offset_ticks=cfg.simulator.max_mid_offset_ticks, episode_length_steps=L,
                feature_mean=meta["feature_mean"], feature_scale=meta["feature_scale"])
        result, metrics = _evaluate_agent(test_env, agent, True, cfg, test_starts)
        _record(rid, b, 0, "N/A", result, metrics, {})
        baseline_metrics[b] = metrics

    from stable_baselines3 import SAC
    per_variant: dict[str, list[dict]] = {v: [] for v in variants}
    for variant in variants:
        for seed in seeds:
            rid = run_id_for(variant, seed)
            if resume and _completed(rid):
                per_variant[variant].append(_load_metrics(rid))
                logger.info("resume: skipping completed run %s", rid)
                continue
            model_dir = os.path.join(artifacts_dir, "rl", variant.lower(), f"seed{seed}")
            train_env = make_env(cfg, variant, "train", artifacts_dir, "train", seed=seed)
            val_env = make_env(cfg, variant, "validation", artifacts_dir, "eval")
            t0 = time.time()
            tr = train_sac_agent(train_env, val_env, sac_cfg, seed=seed, out_dir=model_dir, val_starts=val_starts,
                                 max_inventory=cfg.simulator.max_inventory)
            train_s = time.time() - t0
            model = SAC.load(tr.best_model_path, device=sac_cfg.device)
            env = make_env(cfg, variant, "test", artifacts_dir, "eval")
            result, metrics = _evaluate_agent(env, model, False, cfg, test_starts)
            timings[rid] = {"train_seconds": train_s, "steps_per_second": sac_cfg.total_timesteps / max(train_s, 1e-9)}
            _record(rid, variant, seed, tr.best_model_path, result, metrics,
                    {"best_step": tr.best_step, "best_val_reward": tr.best_val_reward,
                     "total_timesteps": sac_cfg.total_timesteps, "train_seconds": train_s,
                     "last_model_path": tr.last_model_path, "checkpoint_selection": "best validation mean reward"},
                    curve_csv=tr.curve_path)
            per_variant[variant].append(metrics)
            logger.info("run %s done in %.1fs: mean_pnl=%s", rid, train_s, metrics.get("mean_pnl"))

    agg = aggregate_runs({**per_variant, **{b: [m] for b, m in baseline_metrics.items()}})
    comparisons = compare_to_reference({v: a for v, a in agg.items() if v in per_variant})
    summary = {
        "experiment_id": exp_id, "created_utc": datetime.now(timezone.utc).isoformat(), "data_mode": data_mode,
        "fast_mode": fast, "variants": variants, "seeds": seeds, "n_test_windows": len(test_starts),
        "test_windows": test_starts, "n_val_windows": len(val_starts), "episode_steps": L,
        "std_definition": "sample standard deviation across seeds (ddof=1); null when n_seeds < 2",
        "aggregates": {v: {m: {k: x for k, x in d.items() if k != "values"} for m, d in ms.items()}
                       for v, ms in agg.items()},
        "per_seed_values": {v: {m: d["values"] for m, d in ms.items()} for v, ms in agg.items()},
        "comparisons_vs_" + REFERENCE_VARIANT: comparisons, "comparison_note": EXPLORATORY_NOTE,
        "timings": timings,
    }
    os.makedirs(eval_root, exist_ok=True)
    with open(os.path.join(eval_root, f"{exp_id}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    rows = [{"variant": v, "metric": m, "mean": d["mean"], "std": d["std"], "n_seeds": d["n_seeds"]}
            for v, ms in agg.items() for m, d in ms.items()]
    with open(os.path.join(eval_root, f"{exp_id}_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "metric", "mean", "std", "n_seeds"])
        w.writeheader()
        w.writerows(rows)
    write_experiment_aggregates(session, exp_id, rows)
    session.close()
    return summary
