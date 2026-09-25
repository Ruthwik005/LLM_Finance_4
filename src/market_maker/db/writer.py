from __future__ import annotations
import json
from market_maker.db.models import Run, Metric, Episode, Fill


def write_run_to_db(session, run_id: str, variant: str, seed: int, model_path: str,
                     config: dict, backtest_result: dict, metrics: dict) -> None:
    session.query(Fill).filter_by(run_id=run_id).delete()
    session.query(Episode).filter_by(run_id=run_id).delete()
    session.query(Metric).filter_by(run_id=run_id).delete()
    session.query(Run).filter_by(run_id=run_id).delete()

    session.add(Run(run_id=run_id, variant=variant, seed=seed, model_path=model_path,
                     config_json=json.dumps(config)))
    for name, value in metrics.items():
        session.add(Metric(run_id=run_id, name=name, value=float(value)))
    for i, pnl in enumerate(backtest_result["episode_pnls"]):
        session.add(Episode(run_id=run_id, episode_index=i, pnl=float(pnl)))
    for i, ep_fills in enumerate(backtest_result["fills"]):
        for fill in ep_fills:
            session.add(Fill(run_id=run_id, episode_index=i, side=fill["side"],
                              price=float(fill["price"]), volume=float(fill["volume"])))
    session.commit()
