from __future__ import annotations
import csv
import glob
import json
import math
import os
from typing import Union
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from market_maker.api import schemas as S
from market_maker.db.session import get_engine, get_session, init_db
from market_maker.db.models import Run, Metric, Episode, Fill, Step, ExperimentAggregate

app = FastAPI(title="Market Maker Results API")
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"], allow_headers=["*"],
)

DB_PATH = os.environ.get("MARKET_MAKER_DB_PATH", "artifacts/market_maker.db")
ENGINE = get_engine(f"sqlite:///{DB_PATH}")
init_db(ENGINE)

COMPARISON_NOTE = ("Exploratory comparison with a small number of seeds; Welch p-values and effect sizes are "
                   "descriptive only and are not a claim of significance.")


def artifacts_root() -> str:
    """Resolved per call so tests/deployments can point MARKET_MAKER_ARTIFACTS_ROOT elsewhere."""
    return os.environ.get("MARKET_MAKER_ARTIFACTS_ROOT", "artifacts")


def _session():
    return get_session(ENGINE)


def _clean(x):
    """Recursively replace NaN/inf with None so the payload is valid JSON."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _read_data_mode(run_id: str) -> str | None:
    """data_mode lives in the run's file-artifact metadata.json (None if unavailable)."""
    path = os.path.join(artifacts_root(), "evaluation", run_id, "metadata.json")
    try:
        with open(path) as f:
            return json.load(f).get("data_mode")
    except (OSError, ValueError):
        return None


def _require_run(session, run_id: str) -> Run:
    run = session.query(Run).filter_by(run_id=run_id).one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run


@app.get("/health", response_model=S.Health)
def health():
    return {"status": "ok"}


@app.get("/runs", response_model=S.RunList)
def list_runs():
    session = _session()
    try:
        return {"run_ids": sorted(r.run_id for r in session.query(Run).all())}
    finally:
        session.close()


@app.get("/runs/summary", response_model=S.RunsSummary)
def runs_summary():
    session = _session()
    try:
        by_run: dict[str, dict[str, float]] = {}
        for m in session.query(Metric).all():
            by_run.setdefault(m.run_id, {})[m.name] = m.value
        rows = []
        for r in session.query(Run).order_by(Run.run_id).all():
            m = by_run.get(r.run_id, {})
            rows.append({
                "run_id": r.run_id, "variant": r.variant, "seed": r.seed, "data_mode": _read_data_mode(r.run_id),
                "created_at": r.created_at.isoformat(), "mean_pnl": m.get("mean_pnl"), "map": m.get("map"),
                "max_drawdown": m.get("max_drawdown"), "n_fills": m.get("n_fills"),
                "sharpe_like": m.get("sharpe_like"), "pnl_over_map": m.get("pnl_over_map"), "metrics": m,
            })
        return {"runs": rows}
    finally:
        session.close()


@app.get("/runs/{run_id}", response_model=S.RunDetail)
def run_detail(run_id: str):
    session = _session()
    try:
        run = _require_run(session, run_id)
        return {"run_id": run.run_id, "variant": run.variant, "seed": run.seed,
                "model_path": run.model_path, "created_at": run.created_at.isoformat(),
                "data_mode": _read_data_mode(run_id)}
    finally:
        session.close()


@app.get("/runs/{run_id}/metrics", response_model=dict[str, float])
def run_metrics(run_id: str):
    session = _session()
    try:
        _require_run(session, run_id)
        return {m.name: m.value for m in session.query(Metric).filter_by(run_id=run_id).all()}
    finally:
        session.close()


@app.get("/runs/{run_id}/episodes", response_model=list[S.EpisodePnl])
def run_episodes(run_id: str):
    session = _session()
    try:
        _require_run(session, run_id)
        return [{"episode": e.episode_index, "pnl": e.pnl}
                for e in session.query(Episode).filter_by(run_id=run_id).order_by(Episode.episode_index).all()]
    finally:
        session.close()


@app.get("/runs/{run_id}/fills", response_model=list[S.FillRow])
def run_fills(run_id: str):
    session = _session()
    try:
        _require_run(session, run_id)
        return [{"episode": f.episode_index, "side": f.side, "price": f.price, "volume": f.volume}
                for f in session.query(Fill).filter_by(run_id=run_id).order_by(Fill.id).all()]
    finally:
        session.close()


@app.get("/runs/{run_id}/equity-curve", response_model=list[S.EquityPoint])
def run_equity_curve(run_id: str):
    session = _session()
    try:
        _require_run(session, run_id)
        episodes = session.query(Episode).filter_by(run_id=run_id).order_by(Episode.episode_index).all()
        cumulative = 0.0
        curve = []
        for e in episodes:
            cumulative += e.pnl
            curve.append({"episode": e.episode_index, "cumulative_pnl": cumulative})
        return curve
    finally:
        session.close()


@app.get("/runs/{run_id}/pnl-attribution", response_model=S.PnlAttribution)
def pnl_attribution(run_id: str):
    session = _session()
    try:
        _require_run(session, run_id)
        m = {x.name: x.value for x in session.query(Metric).filter_by(run_id=run_id).all()}
        return {"run_id": run_id, "spread_capture_pnl": m.get("spread_capture_pnl"),
                "inventory_pnl": m.get("inventory_pnl"), "fees_paid": m.get("fees_paid"),
                "rebates": m.get("rebates_received"), "total": m.get("total_pnl")}
    finally:
        session.close()


def _downsample_indices(n: int, max_points: int, must_keep: set[int]) -> tuple[list[int], int]:
    stride = max(1, math.ceil(n / max_points)) if max_points > 0 else 1
    keep = {i for i in range(0, n, stride)}
    keep.add(n - 1)
    keep |= {i for i in must_keep if 0 <= i < n}
    return sorted(keep), stride


@app.get("/runs/{run_id}/episodes/{episode}/replay", response_model=S.Replay)
def episode_replay(run_id: str, episode: int, max_points: int = Query(2000, ge=10, le=50000)):
    session = _session()
    try:
        _require_run(session, run_id)
        ep_known = session.query(Episode).filter_by(run_id=run_id, episode_index=episode).first() is not None
        step_rows = session.query(Step).filter_by(run_id=run_id, episode=episode).order_by(Step.step).all()
        if not ep_known and not step_rows:
            raise HTTPException(status_code=404, detail=f"Episode {episode} not found for run '{run_id}'")
        fill_rows = session.query(Fill).filter_by(run_id=run_id, episode_index=episode).order_by(Fill.id).all()
        ts_by_step = {s.step: s.timestamp for s in step_rows}
        fills = [{"side": f.side, "price": f.price, "volume": f.volume, "step": f.step,
                  "timestamp": ts_by_step.get(f.step) if f.step is not None else None} for f in fill_rows]
        n = len(step_rows)
        fill_steps = {f["step"] for f in fills if f["step"] is not None}
        pos_of_step = {s.step: i for i, s in enumerate(step_rows)}
        keep, stride = _downsample_indices(n, max_points, {pos_of_step[t] for t in fill_steps if t in pos_of_step}) if n else ([], 1)
        steps = [{
            "step": s.step, "timestamp": s.timestamp, "mid": s.mid, "bid_1": s.bid_1, "bid_2": s.bid_2,
            "ask_1": s.ask_1, "ask_2": s.ask_2, "bid_vol_1": s.bid_vol_1, "bid_vol_2": s.bid_vol_2,
            "ask_vol_1": s.ask_vol_1, "ask_vol_2": s.ask_vol_2, "inventory": s.inventory, "mtm_pnl": s.mtm_pnl,
            "placed_volume": s.placed_volume, "canceled_volume": s.canceled_volume,
        } for s in (step_rows[i] for i in keep)]
        summary = {
            "n_fills": len(fills), "filled_volume": float(sum(f["volume"] for f in fills)),
            "final_mtm_pnl": step_rows[-1].mtm_pnl if n else None,
            "max_abs_inventory": max(abs(s.inventory) for s in step_rows) if n else None,
            "avg_quoted_spread": sum(s.ask_1 - s.bid_1 for s in step_rows) / n if n else None,
            "cancel_volume": float(sum(s.canceled_volume for s in step_rows)),
            "placed_volume": float(sum(s.placed_volume for s in step_rows)),
        }
        return {"run_id": run_id, "episode": episode, "n_steps_total": n, "n_points": len(steps),
                "downsampled": len(steps) < n, "stride": stride, "steps": steps, "fills": fills, "summary": summary}
    finally:
        session.close()


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


@app.get("/runs/{run_id}/training-curve", response_model=S.TrainingCurve)
def training_curve(run_id: str):
    session = _session()
    try:
        _require_run(session, run_id)
    finally:
        session.close()
    path = os.path.join(artifacts_root(), "evaluation", run_id, "training_curve.csv")
    points = []
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if "timesteps" in row:
                    ts = _num(row["timesteps"])
                    train = _num(row.get("train_reward_mean"))
                    val = _num(row.get("val_reward_mean"))
                elif "step" in row:  # old 2-column format: step,reward
                    ts, train, val = _num(row["step"]), _num(row.get("reward")), None
                else:
                    continue
                if ts is None:
                    continue
                points.append({
                    "timesteps": ts, "train_reward_mean": train, "val_reward_mean": val,
                    "val_pnl_mean": _num(row.get("val_pnl_mean")), "val_map": _num(row.get("val_map")),
                    "ent_coef": _num(row.get("ent_coef")), "critic_loss": _num(row.get("critic_loss")),
                    "actor_loss": _num(row.get("actor_loss")),
                    "is_best": str(row.get("is_best", "")).strip().lower() in ("1", "true", "1.0"),
                })
    except (OSError, csv.Error):
        points = []
    return {"run_id": run_id, "points": points}


def _experiment_summary_path(experiment_id: str) -> str:
    return os.path.join(artifacts_root(), "evaluation", f"{experiment_id}_summary.json")


def _load_summary(experiment_id: str) -> dict | None:
    try:
        with open(_experiment_summary_path(experiment_id)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


@app.get("/experiments", response_model=S.ExperimentList)
def list_experiments():
    session = _session()
    try:
        variants: dict[str, set] = {}
        for eid, var in session.query(ExperimentAggregate.experiment_id, ExperimentAggregate.variant).distinct().all():
            variants.setdefault(eid, set()).add(var)
    finally:
        session.close()
    files = {os.path.basename(p)[: -len("_summary.json")]: p
             for p in glob.glob(os.path.join(artifacts_root(), "evaluation", "*_summary.json"))}
    items = []
    for eid in sorted(set(variants) | set(files)):
        summ = _load_summary(eid) if eid in files else None
        items.append({
            "experiment_id": eid, "data_mode": (summ or {}).get("data_mode"),
            "created_utc": (summ or {}).get("created_utc"),
            "n_variants": len(variants.get(eid) or (summ or {}).get("aggregates", {}) or {}),
            "has_summary_file": summ is not None,
        })
    return {"experiments": items}


@app.get("/experiments/{experiment_id}", response_model=S.ExperimentDetail)
def experiment_detail(experiment_id: str):
    session = _session()
    try:
        rows = session.query(ExperimentAggregate).filter_by(experiment_id=experiment_id).all()
    finally:
        session.close()
    summary = _load_summary(experiment_id)
    if not rows and summary is None:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found")
    agg: dict[str, dict[str, dict]] = {}
    if rows:
        for r in rows:
            agg.setdefault(r.variant, {})[r.metric] = {"mean": r.mean, "std": r.std, "n_seeds": r.n_seeds}
    elif summary:
        for var, metrics in (summary.get("aggregates") or {}).items():
            agg[var] = {m: v for m, v in metrics.items() if isinstance(v, dict) and v.get("mean") is not None}
    variants = []
    for var, metrics in agg.items():
        metrics = _clean(metrics)
        metrics = {m: v for m, v in metrics.items() if v.get("mean") is not None}
        variants.append({"variant": var, "n_seeds": max((v["n_seeds"] for v in metrics.values()), default=0),
                         "metrics": metrics})
    comparisons, ref = [], "SAC_LOB_ONLY"
    if summary:
        for key, per_variant in summary.items():
            if key.startswith("comparisons_vs_"):
                ref = key[len("comparisons_vs_"):]
                for var, per_metric in (per_variant or {}).items():
                    for metric, c in (per_metric or {}).items():
                        comparisons.append(_clean({"variant": var, "metric": metric, **c}))
    return {
        "experiment_id": experiment_id, "data_mode": (summary or {}).get("data_mode"),
        "created_utc": (summary or {}).get("created_utc"), "seeds": (summary or {}).get("seeds") or [],
        "variants": variants, "reference_variant": ref, "comparisons": comparisons,
        "comparison_note": (summary or {}).get("comparison_note") or (COMPARISON_NOTE if comparisons else None),
    }


SIGNAL_PREFIXES = ("sentiment_ewm_", "sentiment_max_abs_", "article_count_", "minutes_since_last_article",
                   "has_recent_news", "transformer_prob_h", "transformer_signal_valid")


@app.get("/signals", response_model=Union[S.SignalsPayload, S.SignalsUnavailable])
def signals(split: str = "test", max_points: int = Query(2000, ge=10, le=50000)):
    import numpy as np
    import pandas as pd
    if split not in ("train", "validation", "test"):
        raise HTTPException(status_code=422, detail="split must be train, validation or test")
    feats = os.path.join(artifacts_root(), "features")
    lob_p = os.path.join(feats, f"lob_{split}.parquet")
    npy_p = os.path.join(feats, f"sac_full_{split}.npy")
    col_p = os.path.join(feats, "sac_full_columns.json")
    missing = [os.path.basename(p) for p in (lob_p, npy_p, col_p) if not os.path.exists(p)]
    if missing:
        return {"available": False, "reason": f"Feature artifacts not built yet (missing: {', '.join(missing)}). "
                                              "Run scripts/build_features.py."}
    try:
        lob = pd.read_parquet(lob_p)
        arr = np.load(npy_p, mmap_mode="r")
        with open(col_p) as f:
            cols = json.load(f)
        if arr.shape[0] != len(lob) or arr.shape[1] != len(cols):
            return {"available": False, "reason": "Feature artifacts are inconsistent (row/column mismatch)."}
        wanted = [c for c in cols if c.startswith(SIGNAL_PREFIXES)]
        if not wanted:
            return {"available": False, "reason": "No sentiment or transformer columns in this feature set."}
        n = len(lob)
        keep, stride = _downsample_indices(n, max_points, set())
        idx = np.array(keep)
        notes = []
        series: dict[str, np.ndarray] = {}
        # sentiment: raw (unstandardised) values if cached, else un-scale with the train-split scaler
        scaler = {}
        sc_p = os.path.join(feats, "extras_scaler.json")
        if os.path.exists(sc_p):
            with open(sc_p) as f:
                scaler = json.load(f)
        raw_sent_p = os.path.join(feats, f"sentiment_raw_{split}.npy")
        sent_cols_p = os.path.join(feats, "sentiment_columns.json")
        raw_sent, sent_cols = None, []
        if os.path.exists(raw_sent_p) and os.path.exists(sent_cols_p):
            with open(sent_cols_p) as f:
                sent_cols = json.load(f)
            raw_sent = np.load(raw_sent_p, mmap_mode="r")
            if raw_sent.shape != (n, len(sent_cols)):
                raw_sent = None
        t_cols = [c for c in cols if c.startswith("transformer_")]
        t_mean = (scaler.get("transformer") or {}).get("mean")
        t_scale = (scaler.get("transformer") or {}).get("scale")
        t_ok = t_mean is not None and len(t_mean) == len(t_cols) and len(t_scale) == len(t_cols)
        for c in wanted:
            j = cols.index(c)
            if c in sent_cols and raw_sent is not None:
                series[c] = np.asarray(raw_sent[idx, sent_cols.index(c)], dtype=float)
            elif c.startswith("transformer_") and t_ok:
                k = t_cols.index(c)
                series[c] = np.asarray(arr[idx, j], dtype=float) * t_scale[k] + t_mean[k]
            else:
                series[c] = np.asarray(arr[idx, j], dtype=float)
                notes.append(f"{c} is shown standardised (train-split z-score).")
        if raw_sent is not None:
            notes.append("Sentiment columns are the raw as-of-aligned values (before standardisation).")
        if t_ok:
            notes.append("Transformer probabilities recovered by inverting the train-split standardisation.")
        ts = pd.to_datetime(lob["timestamp"].iloc[idx], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
        mid = None
        if "midprice" in lob.columns:
            mid = lob["midprice"].iloc[idx].astype(float).tolist()
        elif {"bid_price_1", "ask_price_1"} <= set(lob.columns):
            mid = ((lob["bid_price_1"] + lob["ask_price_1"]) / 2).iloc[idx].astype(float).tolist()
        return _clean({"available": True, "split": split, "n_rows_total": n, "n_points": len(idx), "stride": stride,
                       "timestamps": ts, "midprice": mid,
                       "series": {k: v.tolist() for k, v in series.items()}, "notes": notes})
    except Exception as exc:  # corrupt/unreadable artifact: report, never invent data
        return {"available": False, "reason": f"Could not read feature artifacts: {type(exc).__name__}: {exc}"}
