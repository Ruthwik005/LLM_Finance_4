from fastapi.testclient import TestClient
from market_maker.db.session import get_engine, get_session, init_db
from market_maker.db.writer import write_run_to_db
import market_maker.api.main as api_main


def _seed_db(tmp_path, run_id="run_test"):
    engine = get_engine(f"sqlite:///{tmp_path}/test.db")
    init_db(engine)
    session = get_session(engine)
    write_run_to_db(
        session, run_id=run_id, variant="SAC_LOB_ONLY", seed=42, model_path="N/A", config={"seed": 42},
        backtest_result={"episode_pnls": [1.0, 2.0], "inventories": [], "fills": [[{"side": "bid", "price": 100.0, "volume": 1.0}], []], "orders": [[], []]},
        metrics={"mean_pnl": 1.5},
    )
    session.close()
    return engine


def test_health_endpoint():
    client = TestClient(api_main.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_runs_and_run_detail_endpoints(tmp_path, monkeypatch):
    engine = _seed_db(tmp_path, "run_test")
    monkeypatch.setattr(api_main, "ENGINE", engine)
    client = TestClient(api_main.app)

    resp = client.get("/runs")
    assert resp.status_code == 200
    assert "run_test" in resp.json()["run_ids"]

    resp = client.get("/runs/run_test/metrics")
    assert resp.status_code == 200
    assert resp.json()["mean_pnl"] == 1.5

    resp = client.get("/runs/run_test/equity-curve")
    assert resp.status_code == 200
    assert resp.json()[-1]["cumulative_pnl"] == 3.0


def test_unknown_run_id_returns_404(tmp_path, monkeypatch):
    engine = get_engine(f"sqlite:///{tmp_path}/empty.db")
    init_db(engine)
    monkeypatch.setattr(api_main, "ENGINE", engine)
    client = TestClient(api_main.app)
    resp = client.get("/runs/does_not_exist/metrics")
    assert resp.status_code == 404
