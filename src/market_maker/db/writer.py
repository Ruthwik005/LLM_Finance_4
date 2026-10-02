from __future__ import annotations
import json
from market_maker.db.models import Run, Metric, Episode, Fill, Step


def write_run_to_db(session, run_id: str, variant: str, seed: int, model_path: str,
                     config: dict, backtest_result: dict, metrics: dict) -> None:
    session.query(Step).filter_by(run_id=run_id).delete()
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
    session.flush()
    rows = []
    for ep, ep_steps in enumerate(backtest_result.get("steps") or []):
        for s in ep_steps:
            rows.append(dict(
                run_id=run_id, episode=ep, step=int(s["t"]), timestamp=s.get("timestamp"), mid=float(s["mid"]),
                bid_1=s["bid_prices"][0], bid_2=s["bid_prices"][1], ask_1=s["ask_prices"][0], ask_2=s["ask_prices"][1],
                bid_vol_1=s["bid_volumes"][0], bid_vol_2=s["bid_volumes"][1],
                ask_vol_1=s["ask_volumes"][0], ask_vol_2=s["ask_volumes"][1],
                inventory=s["inventory"], cash=s["cash"], mtm_pnl=s["mtm_pnl"],
                placed_volume=s["placed_volume"], canceled_volume=s["canceled_volume"]))
    if rows:
        session.bulk_insert_mappings(Step, rows)
    session.commit()


def write_experiment_aggregates(session, experiment_id: str, rows: list[dict]) -> None:
    """Replace-on-rerun: rows = [{variant, metric, mean, std|None, n_seeds}, ...]."""
    from market_maker.db.models import ExperimentAggregate
    session.query(ExperimentAggregate).filter_by(experiment_id=experiment_id).delete()
    for r in rows:
        session.add(ExperimentAggregate(experiment_id=experiment_id, variant=r["variant"], metric=r["metric"],
                                        mean=float(r["mean"]), std=None if r["std"] is None else float(r["std"]),
                                        n_seeds=int(r["n_seeds"])))
    session.commit()
