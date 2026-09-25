from market_maker.db.models import Run, Metric, Episode, Fill
from market_maker.db.session import get_engine, get_session, init_db
from market_maker.db.writer import write_run_to_db


def _session(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path}/test.db")
    init_db(engine)
    return get_session(engine)


def test_write_run_to_db_creates_run_metric_episode_fill_rows(tmp_path):
    session = _session(tmp_path)
    backtest_result = {
        "episode_pnls": [1.0, 2.0],
        "inventories": [],
        "fills": [[{"side": "bid", "price": 100.0, "volume": 1.0}], []],
        "orders": [[], []],
    }
    write_run_to_db(session, run_id="run_1", variant="SAC_LOB_ONLY", seed=42, model_path="N/A",
                     config={"seed": 42}, backtest_result=backtest_result, metrics={"mean_pnl": 1.5})

    assert session.query(Run).filter_by(run_id="run_1").one().variant == "SAC_LOB_ONLY"
    assert session.query(Metric).filter_by(run_id="run_1", name="mean_pnl").one().value == 1.5
    assert session.query(Episode).filter_by(run_id="run_1").count() == 2
    assert session.query(Fill).filter_by(run_id="run_1").count() == 1


def test_write_run_to_db_replaces_existing_run_rows_on_rerun(tmp_path):
    session = _session(tmp_path)
    backtest_result = {"episode_pnls": [1.0], "inventories": [], "fills": [[]], "orders": [[]]}
    write_run_to_db(session, run_id="run_1", variant="SAC_LOB_ONLY", seed=42, model_path="N/A",
                     config={}, backtest_result=backtest_result, metrics={"mean_pnl": 1.0})
    write_run_to_db(session, run_id="run_1", variant="SAC_LOB_ONLY", seed=42, model_path="N/A",
                     config={}, backtest_result=backtest_result, metrics={"mean_pnl": 9.0})

    assert session.query(Run).filter_by(run_id="run_1").count() == 1
    assert session.query(Metric).filter_by(run_id="run_1", name="mean_pnl").one().value == 9.0
    assert session.query(Episode).filter_by(run_id="run_1").count() == 1  # not duplicated
