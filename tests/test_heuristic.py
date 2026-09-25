import numpy as np
from market_maker.agents.heuristic import HeuristicMarketMaker


def test_heuristic_returns_valid_action_shape_and_range():
    agent = HeuristicMarketMaker()
    obs = np.zeros(8, dtype=np.float32)
    action = agent.predict(obs, inventory=0.0, max_inventory=100.0)
    assert action.shape == (4,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_heuristic_skews_mid_offset_against_long_inventory():
    agent = HeuristicMarketMaker()
    obs = np.zeros(8, dtype=np.float32)
    action_flat = agent.predict(obs, inventory=0.0, max_inventory=100.0)
    action_long = agent.predict(obs, inventory=80.0, max_inventory=100.0)
    assert action_long[0] < action_flat[0]  # long inventory -> skew quotes down to encourage selling
