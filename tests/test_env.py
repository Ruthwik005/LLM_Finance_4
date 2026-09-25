import numpy as np
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
