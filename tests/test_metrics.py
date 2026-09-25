import numpy as np
from market_maker.evaluation.metrics import (
    mean_absolute_position, pnl_over_map, max_drawdown, adverse_selection_ratio, summarize_episode,
)


def test_mean_absolute_position():
    inv = np.array([1.0, -2.0, 3.0, -4.0])
    assert mean_absolute_position(inv) == 2.5


def test_pnl_over_map_normal_case():
    assert pnl_over_map(100.0, 4.0) == 25.0


def test_pnl_over_map_zero_map_returns_none_not_inf_or_crash():
    assert pnl_over_map(100.0, 0.0) is None


def test_max_drawdown_on_known_curve():
    equity = np.array([100.0, 110.0, 90.0, 95.0, 80.0])
    dd = max_drawdown(equity)
    assert dd == 30.0  # peak 110 -> trough 80


def test_adverse_selection_ratio_flags_bid_fill_followed_by_price_drop():
    midprice_series = np.array([100.0, 100.0, 100.0, 95.0, 95.0])
    fills = [{"side": "bid", "price": 100.0, "volume": 1.0}]
    ratio = adverse_selection_ratio(fills, midprice_series, fill_indices=[1], horizon_steps=2, threshold=0.02)
    assert ratio == 1.0


def test_adverse_selection_ratio_zero_when_no_fills():
    ratio = adverse_selection_ratio([], np.array([100.0, 100.0]), fill_indices=[], horizon_steps=2, threshold=0.02)
    assert ratio == 0.0


def test_summarize_episode_returns_mean_median_std():
    summary = summarize_episode(episode_pnls=[10.0, 20.0, 30.0], inventories=[np.array([1, 2]), np.array([0, 3]), np.array([-1, 1])])
    assert summary["mean_pnl"] == 20.0
    assert summary["median_pnl"] == 20.0
    assert summary["std_pnl"] > 0
