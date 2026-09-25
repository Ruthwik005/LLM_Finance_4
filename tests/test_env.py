import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env
from market_maker.data.sample_data import generate_synthetic_lob
from market_maker.env.market_making_env import MarketMakingEnv
from market_maker.env.accounting import AccountState


def _make_env(max_inventory=100.0, episode_length_steps=50):
    lob_df = generate_synthetic_lob(n_rows=1000, depth=2, symbol="BTCUSDT", seed=11)
    n = len(lob_df)
    feature_columns = [f"f{i}" for i in range(8)]
    fused = np.random.randn(n, 8).astype(np.float32)
    return MarketMakingEnv(
        fused_features=fused, raw_lob_df=lob_df, feature_columns=feature_columns,
        tick_size=0.01, levels_per_side=2, min_spread_ticks=1, max_spread_ticks=8,
        max_mid_offset_ticks=4, total_volume_per_side=20.0, max_inventory=max_inventory,
        fee_per_unit=0.0, reward_config={"type": "truncated_inventory", "inventory_threshold": 40,
                                          "inventory_penalty": 0.001, "cancellation_penalty": 0.0001, "scale": 1.0},
        reference_price_mode="midprice", episode_length_steps=episode_length_steps,
    )


def test_gymnasium_check_env_passes():
    env = _make_env()
    check_env(env.unwrapped if hasattr(env, "unwrapped") else env, skip_render_check=True)


def test_observation_always_within_observation_space():
    env = _make_env()
    obs, _ = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        if terminated or truncated:
            break


def test_inventory_never_exceeds_max_long_limit():
    env = _make_env(max_inventory=5.0)
    env.reset(seed=42)
    max_buy_action = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)  # push mid up, tight spread, all bid volume
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(max_buy_action)
        assert info["inventory"] <= 5.0 + 1e-6
        if terminated or truncated:
            break


def test_inventory_never_exceeds_max_short_limit():
    env = _make_env(max_inventory=5.0)
    env.reset(seed=42)
    max_sell_action = np.array([-1.0, -1.0, -1.0, 1.0], dtype=np.float32)
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(max_sell_action)
        assert info["inventory"] >= -5.0 - 1e-6
        if terminated or truncated:
            break


def test_episode_terminates_at_configured_length():
    env = _make_env(episode_length_steps=10)
    env.reset(seed=42)
    steps = 0
    for _ in range(100):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
        if terminated or truncated:
            break
    assert steps <= 10


def test_levels_per_side_other_than_two_raises():
    with pytest.raises(ValueError):
        _make_env_with_levels(levels_per_side=1)
    with pytest.raises(ValueError):
        _make_env_with_levels(levels_per_side=3)


def _make_env_with_levels(levels_per_side):
    lob_df = generate_synthetic_lob(n_rows=100, depth=2, symbol="BTCUSDT", seed=11)
    n = len(lob_df)
    feature_columns = [f"f{i}" for i in range(8)]
    fused = np.random.randn(n, 8).astype(np.float32)
    return MarketMakingEnv(
        fused_features=fused, raw_lob_df=lob_df, feature_columns=feature_columns,
        tick_size=0.01, levels_per_side=levels_per_side, min_spread_ticks=1, max_spread_ticks=8,
        max_mid_offset_ticks=4, total_volume_per_side=20.0, max_inventory=100.0,
        fee_per_unit=0.0, reward_config={"type": "truncated_inventory", "inventory_threshold": 40,
                                          "inventory_penalty": 0.001, "cancellation_penalty": 0.0001, "scale": 1.0},
        reference_price_mode="midprice", episode_length_steps=50,
    )


def test_reward_captures_realized_fill_edge_not_just_price_drift():
    # Directly demonstrate the accounting-level bug/fix: a naive "post-fill only"
    # delta_mtm at a constant reference price is zero even though a profitable
    # fill (buying below the reference price) just happened, whereas computing
    # prev_mtm BEFORE the fill (at the same price) correctly shows the edge.
    account = AccountState(cash=0.0, inventory=0.0, realized_pnl=0.0, fees=0.0,
                            filled_bid_volume=0.0, filled_ask_volume=0.0, turnover=0.0)
    ref_price = 100.0

    # Correct (fixed) ordering: snapshot mark-to-market BEFORE applying the fill.
    prev_mtm_fixed = account.mark_to_market(ref_price)

    # Buy 10 units at 95, which is 5 below the reference price -> realized edge of 50.
    account.apply_fill("bid", price=95.0, volume=10.0, fee_per_unit=0.0)

    next_ref_price = ref_price  # no price drift this step
    new_mtm = account.mark_to_market(next_ref_price)

    delta_mtm_fixed = new_mtm - prev_mtm_fixed
    assert delta_mtm_fixed == pytest.approx(50.0)

    # Buggy (old) ordering: snapshot mark-to-market AFTER applying the fill,
    # using the same (old) reference price -> misses the fill's edge entirely.
    prev_mtm_buggy = account.mark_to_market(ref_price)  # already post-fill here
    delta_mtm_buggy = new_mtm - prev_mtm_buggy
    assert delta_mtm_buggy == pytest.approx(0.0)
    assert delta_mtm_buggy != delta_mtm_fixed


def test_env_step_reward_reflects_realized_fill_edge_true_integration():
    """True integration test: builds a MarketMakingEnv with a hand-controlled two-row
    order book and calls env.step() directly, pinning down the numeric reward that
    depends on the prev_mtm-before-fills ordering fixed in market_making_env.py.

    Scenario (both rows have identical bid/ask -> zero reference-price drift across
    the step, isolating the realized fill edge):
      row0 (current): bid=99.99, ask=100.01 -> ref_price = 100.00
      row1 (next):    bid=99.99, ask=100.01 -> next_ref_price = 100.00 (no drift)
                       trade_price=99.0, trade_side="sell", trade_size=20

    Action is chosen (mid_offset=0, spread=min=2 ticks, bid_split=1.0, ask_split=0.0)
    so that:
      - bid_price_l1 = 99.99, bid_vol_l1 = total_volume_per_side (10), bid L2 vol = 0
      - ask_price_l1 = 100.01, ask_vol_l1 = 0, ask_vol_l2 = 10 (ask_price_l2 = 100.02)
    The next row's sell trade at 99.0 (size 20) crosses and fills the bid order at
    99.99 for the full 10 units (trade evidence caps/justifies the fill); the ask
    legs never fill (next_best_bid=99.99 does not reach 100.01/100.02, and the
    trade is a "sell", not a "buy", so it cannot justify an ask fill).

    Hand-calculation with the CORRECT (fixed) ordering — prev_mtm snapshotted
    BEFORE fills are applied:
      prev_mtm = mark_to_market(ref_price=100.0) on an empty account = 0.0
      after fill: cash = -99.99*10 = -999.9, inventory = 10
      new_mtm = mark_to_market(next_ref_price=100.0) = -999.9 + 10*100.0 = 0.1
      expected pnl = new_mtm - prev_mtm = 0.1

    Under the OLD (reverted) ordering — prev_mtm snapshotted AFTER fills, at the
    (unchanged) reference price — prev_mtm would already equal 0.1 (same cash/
    inventory, same price), so delta_mtm would be 0.1 - 0.1 = 0.0. This test
    would fail under that reversion since it asserts pnl != 0.
    """
    n_features = 4
    raw_lob_df = pd.DataFrame({
        "bid_price_1": [99.99, 99.99],
        "ask_price_1": [100.01, 100.01],
        "trade_price": [np.nan, 99.0],
        "trade_side": [None, "sell"],
        "trade_size": [0.0, 20.0],
    })
    fused_features = np.zeros((2, n_features), dtype=np.float32)

    env = MarketMakingEnv(
        fused_features=fused_features, raw_lob_df=raw_lob_df, feature_columns=[f"f{i}" for i in range(n_features)],
        tick_size=0.01, levels_per_side=2, min_spread_ticks=2, max_spread_ticks=8,
        max_mid_offset_ticks=4, total_volume_per_side=10.0, max_inventory=100.0,
        fee_per_unit=0.0, reward_config={"type": "truncated_inventory", "inventory_threshold": 40,
                                          "inventory_penalty": 0.001, "cancellation_penalty": 0.0001, "scale": 1.0},
        reference_price_mode="midprice", episode_length_steps=1,
    )
    env.reset(seed=42)
    assert env._data_idx == 0  # only one valid start index given the 2-row df + episode_length_steps=1

    action = np.array([0.0, -1.0, 1.0, -1.0], dtype=np.float32)
    _, reward, terminated, truncated, info = env.step(action)

    assert len(info["fills"]) == 1
    fill = info["fills"][0]
    assert fill["side"] == "bid"
    assert fill["price"] == pytest.approx(99.99)
    assert fill["volume"] == pytest.approx(10.0)

    assert info["reward_components"]["pnl"] == pytest.approx(0.1, abs=1e-3)
    assert info["reward_components"]["pnl"] != pytest.approx(0.0, abs=1e-3)
