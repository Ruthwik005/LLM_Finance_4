from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from market_maker.db.session import get_engine, get_session, init_db
from market_maker.db.models import Run, Metric, Episode, Fill

app = FastAPI(title="Market Maker Results API")
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:5173"],
    allow_methods=["GET"], allow_headers=["*"],
)

DB_PATH = os.environ.get("MARKET_MAKER_DB_PATH", "artifacts/market_maker.db")
ENGINE = get_engine(f"sqlite:///{DB_PATH}")
init_db(ENGINE)


def _session():
    return get_session(ENGINE)


def _require_run(session, run_id: str) -> Run:
    run = session.query(Run).filter_by(run_id=run_id).one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/runs")
def list_runs():
    session = _session()
    run_ids = [r.run_id for r in session.query(Run).all()]
    session.close()
    return {"run_ids": sorted(run_ids)}


@app.get("/runs/{run_id}")
def run_detail(run_id: str):
    session = _session()
    run = _require_run(session, run_id)
    result = {"run_id": run.run_id, "variant": run.variant, "seed": run.seed,
              "model_path": run.model_path, "created_at": run.created_at.isoformat()}
    session.close()
    return result


@app.get("/runs/{run_id}/metrics")
def run_metrics(run_id: str):
    session = _session()
    _require_run(session, run_id)
    metrics = {m.name: m.value for m in session.query(Metric).filter_by(run_id=run_id).all()}
    session.close()
    return metrics


@app.get("/runs/{run_id}/episodes")
def run_episodes(run_id: str):
    session = _session()
    _require_run(session, run_id)
    episodes = [{"episode": e.episode_index, "pnl": e.pnl}
                for e in session.query(Episode).filter_by(run_id=run_id).order_by(Episode.episode_index).all()]
    session.close()
    return episodes


@app.get("/runs/{run_id}/fills")
def run_fills(run_id: str):
    session = _session()
    _require_run(session, run_id)
    fills = [{"episode": f.episode_index, "side": f.side, "price": f.price, "volume": f.volume}
              for f in session.query(Fill).filter_by(run_id=run_id).all()]
    session.close()
    return fills


@app.get("/runs/{run_id}/equity-curve")
def run_equity_curve(run_id: str):
    session = _session()
    _require_run(session, run_id)
    episodes = session.query(Episode).filter_by(run_id=run_id).order_by(Episode.episode_index).all()
    session.close()
    cumulative = 0.0
    curve = []
    for e in episodes:
        cumulative += e.pnl
        curve.append({"episode": e.episode_index, "cumulative_pnl": cumulative})
    return curve
